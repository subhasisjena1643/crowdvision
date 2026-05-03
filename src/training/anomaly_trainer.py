"""Anomaly detection trainer (autoencoder-based)."""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..evaluation.anomaly_metrics import evaluate_anomaly_detection


class AnomalyTrainer(BaseTrainer):
    """
    Trainer for ConvAE / ConvLSTMAE anomaly detection.
    Trains on normal examples only (reconstruction objective).
    """

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            # loader may return just frames (train), or (frames, labels) (test)
            if isinstance(batch, (list, tuple)):
                frames = batch[0].to(self.device)
            else:
                frames = batch.to(self.device)

            self.optimizer.zero_grad()
            recon = self.model(frames)
            loss = F.mse_loss(recon, frames)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * frames.shape[0]
            n += frames.shape[0]

        return {'recon_loss': total_loss / n}

    @torch.no_grad()
    def evaluate(self, loader):
        # loader here should be the TEST loader (has labels)
        metrics = evaluate_anomaly_detection(
            self.model, None, loader, self.device)
        return metrics
