"""Base trainer with checkpoint, logging, early stopping."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    """
    Abstract base trainer. Subclasses implement train_epoch and evaluate.
    """

    def __init__(self, model: nn.Module, optimizer, scheduler=None,
                 device: str = 'auto', experiment_name: str = 'exp',
                 save_dir: str = 'checkpoints', log_dir: str = 'runs'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=Path(log_dir) / experiment_name)
        self.best_metric = float('inf')
        self.history: Dict = {'train': [], 'val': []}
        self.start_epoch = 0

    # ------------------------------------------------------------------
    # To be overridden
    # ------------------------------------------------------------------

    def train_epoch(self, loader) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate(self, loader) -> Dict[str, float]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader,
              epochs: int = 100,
              patience: int = 20,
              metric_key: str = 'mae',
              lower_is_better: bool = True) -> Dict:
        """
        Full training loop with early stopping.

        Returns:
            dict with training history
        """
        best_epoch = 0
        no_improve = 0
        t0 = time.time()

        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            print(f"\nEpoch {epoch+1}/{self.start_epoch + epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train'].append(train_metrics)
            self._log_metrics(train_metrics, epoch, prefix='train')
            self._print_metrics(train_metrics, "TRAIN")

            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history['val'].append(val_metrics)
            self._log_metrics(val_metrics, epoch, prefix='val')
            self._print_metrics(val_metrics, "VAL  ")

            # LR scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get(metric_key, 0))
                    else:
                        self.scheduler.step()

            # Early stopping / checkpoint
            current = val_metrics.get(metric_key, float('inf'))
            is_best = (current < self.best_metric) if lower_is_better else (current > self.best_metric)
            if is_best:
                self.best_metric = current
                best_epoch = epoch + 1
                no_improve = 0
                self.save_checkpoint('best.pt', epoch, val_metrics)
                print(f"  ✓ New best {metric_key}: {current:.4f}")
            else:
                no_improve += 1

            self.save_checkpoint('last.pt', epoch, val_metrics)

            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        elapsed = (time.time() - t0) / 60
        print(f"\nTraining complete. Best {metric_key}={self.best_metric:.4f} at epoch {best_epoch}")
        print(f"Total time: {elapsed:.1f} min")
        self.writer.close()
        self._save_history()
        return self.history

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        torch.save(state, self.save_dir / filename)

    def load_checkpoint(self, filename: str = 'best.pt'):
        path = self.save_dir / filename
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        self.start_epoch = state['epoch'] + 1
        self.best_metric = state.get('best_metric', float('inf'))
        print(f"Loaded checkpoint from epoch {state['epoch']}, best={self.best_metric:.4f}")

    def _log_metrics(self, metrics: Dict, step: int, prefix: str):
        for k, v in metrics.items():
            self.writer.add_scalar(f'{prefix}/{k}', v, step)

    @staticmethod
    def _print_metrics(metrics: Dict, tag: str):
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(f"  [{tag}] " + "  ".join(parts))

    def _save_history(self):
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
