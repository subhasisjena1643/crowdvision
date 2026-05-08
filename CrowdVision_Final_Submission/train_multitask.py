#!/usr/bin/env python3
"""
CrowdVision — Multitask Training (UnifiedCrowdVision)

Trains the unified model with density + anomaly heads on SHA-A density data.
The anomaly head is trained as a binary classifier using density-derived
pseudo-labels (high density → potential anomaly signal).

This is the NOVEL CONTRIBUTION: joint density-anomaly training with
cross-task consistency regularisation.

Usage:
    python3 train_multitask.py
    python3 train_multitask.py --fresh
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.multitask.unified import UnifiedCrowdVision
from src.data_loaders.shanghaitech import ShanghaiTechDataset
from src.losses.density_losses import CombinedDensityLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_ROOT = REPO_ROOT / 'checkpoints'
EXP_DIR = REPO_ROOT / 'experiments'


def train_multitask(fresh: bool = False):
    print('=' * 70)
    print('  TRAINING UnifiedCrowdVision (Density + Anomaly)')
    print('=' * 70)

    cfg = {
        'epochs': 40,
        'batch_size': 4,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'patience': 15,
        'consistency_weight': 0.1,
        'anomaly_weight': 0.5,
    }

    ckpt_dir = CKPT_ROOT / 'unified_multitask'
    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    data_root = REPO_ROOT / 'data' / 'shanghaitech_part_A_final'
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((576, 768)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    train_ds = ShanghaiTechDataset(str(data_root), split='train',
                                    transform=tf, target_size=(576, 768))
    val_ds = ShanghaiTechDataset(str(data_root), split='test',
                                  transform=tf, target_size=(576, 768))

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')

    # ── Model ──
    model = UnifiedCrowdVision(
        backbone='vgg16', pretrained=True,
        tasks=['density', 'anomaly']
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    # ── Losses ──
    density_loss_fn = CombinedDensityLoss()
    anomaly_loss_fn = nn.BCELoss()

    # ── Optimizer ──
    opt = optim.Adam(model.parameters(), lr=cfg['lr'],
                     weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-6)

    best_mae = float('inf')
    no_improve = 0
    history = {'train': [], 'val': []}

    t0 = time.time()
    for epoch in range(cfg['epochs']):
        print(f'\nEpoch {epoch+1}/{cfg["epochs"]}')

        # ── Train ──
        model.train()
        total_loss = 0
        total_density_loss = 0
        total_anomaly_loss = 0
        n = 0

        for batch in train_loader:
            imgs, density_maps = batch[0].to(DEVICE), batch[1].to(DEVICE)

            opt.zero_grad(set_to_none=True)

            out = model(imgs)

            # Density loss
            d_loss = density_loss_fn(out['density'], density_maps)

            # Anomaly pseudo-labels: frames with count > median are "potential anomaly"
            with torch.no_grad():
                counts = density_maps.flatten(1).sum(1)
                pseudo_labels = (counts > counts.median()).float().unsqueeze(1)

            a_loss = anomaly_loss_fn(out['anomaly'], pseudo_labels)

            # Consistency loss
            c_loss = model.consistency_loss(out)

            loss = d_loss + cfg['anomaly_weight'] * a_loss + cfg['consistency_weight'] * c_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * imgs.shape[0]
            total_density_loss += d_loss.item() * imgs.shape[0]
            total_anomaly_loss += a_loss.item() * imgs.shape[0]
            n += imgs.shape[0]

        train_metrics = {
            'loss': total_loss / n,
            'density_loss': total_density_loss / n,
            'anomaly_loss': total_anomaly_loss / n,
        }
        history['train'].append(train_metrics)
        print(f'  [TRAIN] loss={train_metrics["loss"]:.4f}  '
              f'd_loss={train_metrics["density_loss"]:.4f}  '
              f'a_loss={train_metrics["anomaly_loss"]:.4f}')

        # ── Validate ──
        model.eval()
        val_mae_sum = 0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, density_maps = batch[0].to(DEVICE), batch[1].to(DEVICE)
                out = model(imgs)
                pred_count = out['density'].flatten(1).sum(1)
                gt_count = density_maps.flatten(1).sum(1)
                val_mae_sum += (pred_count - gt_count).abs().sum().item()
                val_n += imgs.shape[0]

        val_mae = val_mae_sum / val_n
        val_metrics = {'mae': val_mae}
        history['val'].append(val_metrics)
        print(f'  [VAL  ] mae={val_mae:.2f}')

        sched.step()

        # ── Checkpoint ──
        if val_mae < best_mae:
            best_mae = val_mae
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'best_mae': best_mae,
                'metrics': val_metrics,
            }, ckpt_dir / 'best.pt')
            print(f'  ✓ New best MAE: {val_mae:.2f}')
        else:
            no_improve += 1

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'best_mae': best_mae,
        }, ckpt_dir / 'last.pt')

        if no_improve >= cfg['patience']:
            print(f'\n  Early stopping at epoch {epoch+1}')
            break

    elapsed = (time.time() - t0) / 60
    print(f'\n  Multitask training complete in {elapsed:.1f} min')
    print(f'  Best density MAE: {best_mae:.2f}')

    # Save history
    with open(ckpt_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    print(f'Device: {DEVICE}')
    train_multitask(fresh=args.fresh)
