"""Anomaly detection trainer (autoencoder-based).

Key design decisions:
  - ConvAE/MemAE: MSE-only loss + memory sparsity (matching MemAE paper)
  - FutureFrameNet: Phased loss (MSE warmup → MSE+SSIM+GDL composite)
  - AMP (Automatic Mixed Precision) for ~2x GPU speedup
  - Gradient clipping for stability
  - Per-clip evaluation for UCSD datasets
"""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..evaluation.anomaly_metrics import evaluate_anomaly_detection


# ── Loss helpers ──────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
    return g / g.sum()


_ssim_kernel_cache = {}


def _ssim_loss(x: torch.Tensor, y: torch.Tensor,
               window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1 − SSIM as a differentiable loss."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = x.shape[1]
    cache_key = (ch, x.device, window_size)
    if cache_key not in _ssim_kernel_cache:
        k1d = _gaussian_kernel_1d(window_size, sigma).to(x.device)
        kernel = k1d.unsqueeze(0) * k1d.unsqueeze(1)
        _ssim_kernel_cache[cache_key] = kernel.expand(ch, 1, -1, -1).contiguous()
    kernel = _ssim_kernel_cache[cache_key]

    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=ch)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=ch)
    sig_xx = F.conv2d(x * x, kernel, padding=pad, groups=ch) - mu_x * mu_x
    sig_yy = F.conv2d(y * y, kernel, padding=pad, groups=ch) - mu_y * mu_y
    sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=ch) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_xx + sig_yy + C2))
    return 1.0 - ssim_map.mean()


def _gdl_loss(x: torch.Tensor, y: torch.Tensor, alpha: int = 2) -> torch.Tensor:
    """Gradient Difference Loss."""
    dx_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dx_y = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    dy_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dy_y = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    return (torch.abs(dx_x - dx_y).pow(alpha).mean() +
            torch.abs(dy_x - dy_y).pow(alpha).mean())


# ── Trainer ───────────────────────────────────────────────────────────────

class AnomalyTrainer(BaseTrainer):
    """
    Trainer for ConvAE / FutureFrameNet anomaly detection.

    Training strategy per model type:
      - ConvAE (reconstruction): MSE + memory sparsity loss only
        (matches the MemAE paper — no SSIM/GDL for reconstruction models)
      - FutureFrameNet (prediction): MSE warmup → MSE + SSIM + GDL composite
        (composite loss helps prediction models capture edge/structure detail)
    """

    def __init__(self, *args, warmup_epochs: int = 10,
                 use_amp: bool = True, grad_clip: float = 1.0,
                 sparsity_weight: float = 0.002,
                 data_root: str = None, ped: str = 'ped2',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.sparsity_weight = sparsity_weight
        self.data_root = data_root
        self.ped = ped
        self._current_epoch = 0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        n = 0
        is_prediction = hasattr(self.model, 'prediction_error')
        is_reconstruction = hasattr(self.model, 'memory')  # ConvAE has memory module

        # For prediction models: phased loss
        in_warmup = self._current_epoch < self.warmup_epochs
        self._current_epoch += 1

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                frames = batch[0].to(self.device, non_blocking=True)
            else:
                frames = batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if is_prediction and frames.dim() == 5:
                    # ── Prediction model (FutureFrameNet) ──
                    B, T, C, H, W = frames.shape
                    past = frames[:, :-1]
                    target = frames[:, -1]
                    x_in = past.reshape(B, (T - 1) * C, H, W)
                    recon = self.model(x_in)
                    loss_target = target
                else:
                    # ── Reconstruction model (ConvAE / ConvLSTMAE) ──
                    is_clip = (frames.dim() == 5)
                    if is_clip and not hasattr(self.model, 'temp_enc'):
                        B, T, C, H, W = frames.shape
                        frames = frames.reshape(B * T, C, H, W)
                    recon = self.model(frames)
                    loss_target = frames

                # MSE always active
                mse = F.mse_loss(recon, loss_target)

                if is_reconstruction:
                    # ConvAE/MemAE: MSE-only + sparsity (matching MemAE paper)
                    loss = mse
                    if hasattr(self.model, 'memory') and hasattr(self.model.memory, 'sparsity_loss'):
                        loss = loss + self.sparsity_weight * self.model.memory.sparsity_loss()
                elif is_prediction:
                    # FutureFrameNet: phased composite loss
                    if in_warmup:
                        loss = mse
                    else:
                        recon_4d = recon if recon.dim() == 4 else recon.view(-1, *recon.shape[-3:])
                        target_4d = loss_target if loss_target.dim() == 4 else loss_target.view(-1, *loss_target.shape[-3:])
                        ssim = _ssim_loss(recon_4d, target_4d)
                        gdl = _gdl_loss(recon_4d, target_4d)
                        loss = mse + ssim + 0.5 * gdl
                else:
                    loss = mse

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() * recon.shape[0]
            n += recon.shape[0]

        return {'recon_loss': total_loss / n}

    @torch.no_grad()
    def evaluate(self, loader):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = evaluate_anomaly_detection(
            self.model, None, loader, self.device,
            data_root=self.data_root, ped=self.ped,
            use_per_clip=(self.data_root is not None))
        return metrics
