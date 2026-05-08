"""Forecasting (GCN-GRU / AdaptiveNAS-GNN) trainer."""

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..evaluation.forecasting_metrics import evaluate_forecasting


class ForecastingTrainer(BaseTrainer):
    """
    Trainer for spatiotemporal graph forecasting models.
    Handles both standard and NAS (bilevel) training.
    """

    def __init__(self, model, optimizer, adj: torch.Tensor,
                 scaler, scheduler=None, device='auto',
                 experiment_name='forecasting',
                 arch_optimizer=None, **kwargs):
        super().__init__(model, optimizer, scheduler, device,
                         experiment_name, **kwargs)
        self.adj = torch.as_tensor(adj, dtype=torch.float32).to(self.device)
        self.scaler = scaler
        self.arch_optimizer = arch_optimizer   # for NAS bilevel update

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        n = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            # --- Model weights update ---
            self.optimizer.zero_grad()
            pred = self.model(x, self.adj)      # [B, T_out, N, 1]
            loss = nn.HuberLoss()(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # --- Architecture params update (NAS only) ---
            if self.arch_optimizer is not None:
                self.arch_optimizer.zero_grad()
                pred2 = self.model(x, self.adj)
                arch_loss = nn.HuberLoss()(pred2, y)
                arch_loss.backward()
                self.arch_optimizer.step()

            total_loss += loss.item() * x.shape[0]
            n += x.shape[0]

        return {'loss': total_loss / n}

    @torch.no_grad()
    def evaluate(self, loader):
        results = evaluate_forecasting(
            self.model, loader, self.scaler, self.adj, self.device)
        # Flatten for logging
        flat = {}
        for horizon, metrics in results.items():
            for k, v in metrics.items():
                flat[f'{horizon}_{k}'] = v
        return flat
