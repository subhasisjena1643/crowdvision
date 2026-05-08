"""Density estimation trainer."""

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from ..losses.density_losses import CombinedDensityLoss
from ..evaluation.density_metrics import evaluate_density


class DensityTrainer(BaseTrainer):
    """
    Trainer for density estimation models (CSRNet, AdaptiveCSRNet).
    """

    def __init__(self, model, optimizer, scheduler=None,
                 loss_fn=None, device='auto',
                 experiment_name='density', **kwargs):
        super().__init__(model, optimizer, scheduler, device,
                         experiment_name, **kwargs)
        self.loss_fn = loss_fn or CombinedDensityLoss()
        self.loss_fn = self.loss_fn.to(self.device) if hasattr(self.loss_fn, 'to') else self.loss_fn

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        for imgs, density_gt, counts_gt in loader:
            imgs       = imgs.to(self.device)
            density_gt = density_gt.to(self.device)
            counts_gt  = counts_gt.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(imgs)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]

            loss = self.loss_fn(pred, density_gt)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            bs = imgs.shape[0]
            pred_counts = pred.flatten(1).sum(1)
            mae = (pred_counts - counts_gt).abs().mean().item()

            total_loss += loss.item() * bs
            total_mae  += mae * bs
            n += bs

        return {'loss': total_loss / n, 'mae': total_mae / n}

    @torch.no_grad()
    def evaluate(self, loader):
        metrics = evaluate_density(self.model, loader, self.device)
        return metrics
