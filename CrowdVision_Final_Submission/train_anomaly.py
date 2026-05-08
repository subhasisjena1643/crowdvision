#!/usr/bin/env python3
"""
CrowdVision Anomaly Detection — Production Training Script

Trains ConvAE (MemAE) and FutureFrameNet on UCSD Ped2 from scratch
with all fixes applied:
  - U-Net skip connections in ConvAE
  - Phased loss (MSE warmup → composite MSE+SSIM+GDL)
  - AMP for ~2x GPU speedup
  - Gradient clipping
  - Proper LR scheduling with warmup

Usage:
    python3 train_anomaly.py                    # train both
    python3 train_anomaly.py --model convae     # train ConvAE only
    python3 train_anomaly.py --model ffnet      # train FFNet only
    python3 train_anomaly.py --fresh             # delete old checkpoints first
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders.ucsd import get_ucsd_loaders
from src.models.anomaly.conv_ae import ConvAE, ConvLSTMAE
from src.models.anomaly.future_frame import FutureFrameNet
from src.training.anomaly_trainer import AnomalyTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_ROOT = REPO_ROOT / 'checkpoints'


def train_convae(fresh: bool = False):
    """Train ConvAE with MemAE on UCSD Ped2."""
    print('=' * 70)
    print('  TRAINING ConvAE + MemAE')
    print('=' * 70)

    cfg = {
        'ped': 'ped2',
        'clip_len': 1,           # single frame for ConvAE
        'batch_size': 128,       # bigger batch = better gradient on small dataset
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 40,
        'warmup_epochs': 5,
        'num_workers': 4,
        'base_ch': 64,           # bigger model = better capacity
    }

    ckpt_dir = CKPT_ROOT / 'convae_ped2'
    if fresh and ckpt_dir.exists():
        print(f'  Deleting old checkpoints: {ckpt_dir}')
        shutil.rmtree(ckpt_dir)

    # Data
    train_loader, test_loader = get_ucsd_loaders(
        ped=cfg['ped'],
        data_root=str(REPO_ROOT / 'data'),
        clip_len=cfg['clip_len'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )
    print(f'  Train samples: {len(train_loader.dataset)}')
    print(f'  Test samples:  {len(test_loader.dataset)}')

    # Model — larger base_ch for more capacity
    model = ConvAE(
        in_channels=1, base_ch=cfg['base_ch'],
        mem_slots=500, shrink_thres=0.005,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    # Optimizer + scheduler — StepLR halves every 20 epochs
    opt = optim.Adam(model.parameters(), lr=cfg['lr'],
                     weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    # Trainer — MSE-only + sparsity + per-clip eval
    trainer = AnomalyTrainer(
        model=model, optimizer=opt, scheduler=sched, device=DEVICE,
        experiment_name='convae_ped2',
        save_dir=str(CKPT_ROOT), log_dir=str(REPO_ROOT / 'runs'),
        warmup_epochs=cfg['warmup_epochs'],
        use_amp=True,
        grad_clip=1.0,
        data_root=str(REPO_ROOT / 'data'),
        ped='ped2',
    )

    if not fresh:
        trainer.load_checkpoint('last.pt')

    t0 = time.time()
    history = trainer.train(
        train_loader, test_loader,
        epochs=cfg['epochs'],
        patience=cfg['patience'],
        metric_key='auc',
        lower_is_better=False,
    )
    elapsed = (time.time() - t0) / 60
    print(f'\n  ConvAE training complete in {elapsed:.1f} min')

    # Final eval
    final = trainer.evaluate(test_loader)
    print(f'  FINAL — AUC: {final["auc"]:.2f}% | EER: {final["eer"]:.2f}% | '
          f'AP: {final["ap"]:.2f}% | F1: {final["f1"]:.2f}%')
    return history


def train_ffnet(fresh: bool = False):
    """Train FutureFrameNet on UCSD Ped2."""
    print('\n' + '=' * 70)
    print('  TRAINING FutureFrameNet (U-Net Predictor)')
    print('=' * 70)

    cfg = {
        'ped': 'ped2',
        'clip_len': 5,           # 4 input + 1 target
        'batch_size': 32,        # clips use more memory
        'lr': 2e-4,
        'weight_decay': 1e-5,
        'epochs': 80,
        'patience': 30,
        'warmup_epochs': 8,
        'num_workers': 4,
    }

    ckpt_dir = CKPT_ROOT / 'ffnet_ped2'
    if fresh and ckpt_dir.exists():
        print(f'  Deleting old checkpoints: {ckpt_dir}')
        shutil.rmtree(ckpt_dir)

    # Data
    train_loader, test_loader = get_ucsd_loaders(
        ped=cfg['ped'],
        data_root=str(REPO_ROOT / 'data'),
        clip_len=cfg['clip_len'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )
    print(f'  Train samples: {len(train_loader.dataset)}')
    print(f'  Test samples:  {len(test_loader.dataset)}')

    # Model
    model = FutureFrameNet(
        num_input_frames=cfg['clip_len'] - 1,
        in_channels=1,
        base_ch=32,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    # Optimizer + scheduler
    opt = optim.Adam(model.parameters(), lr=cfg['lr'],
                     weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'], eta_min=1e-6)

    # Trainer — phased composite loss + per-clip eval
    trainer = AnomalyTrainer(
        model=model, optimizer=opt, scheduler=sched, device=DEVICE,
        experiment_name='ffnet_ped2',
        save_dir=str(CKPT_ROOT), log_dir=str(REPO_ROOT / 'runs'),
        warmup_epochs=cfg['warmup_epochs'],
        use_amp=True,
        grad_clip=1.0,
        data_root=str(REPO_ROOT / 'data'),
        ped='ped2',
    )

    if not fresh:
        trainer.load_checkpoint('last.pt')

    t0 = time.time()
    history = trainer.train(
        train_loader, test_loader,
        epochs=cfg['epochs'],
        patience=cfg['patience'],
        metric_key='auc',
        lower_is_better=False,
    )
    elapsed = (time.time() - t0) / 60
    print(f'\n  FutureFrameNet training complete in {elapsed:.1f} min')

    # Final eval
    final = trainer.evaluate(test_loader)
    print(f'  FINAL — AUC: {final["auc"]:.2f}% | EER: {final["eer"]:.2f}% | '
          f'AP: {final["ap"]:.2f}% | F1: {final["f1"]:.2f}%')
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrowdVision Anomaly Training')
    parser.add_argument('--model', choices=['convae', 'ffnet', 'both'],
                        default='both', help='Which model to train')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete old checkpoints and train from scratch')
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    if args.model in ('convae', 'both'):
        train_convae(fresh=args.fresh)

    if args.model in ('ffnet', 'both'):
        train_ffnet(fresh=args.fresh)

    print('\n' + '=' * 70)
    print('  ALL TRAINING COMPLETE')
    print('=' * 70)
