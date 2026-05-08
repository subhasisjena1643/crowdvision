import torch
import torch.optim as optim
from pathlib import Path

from src.data_loaders.ucsd import get_ucsd_loaders
from src.models import FutureFrameNet
from src.training import AnomalyTrainer
import sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

REPO_ROOT = Path('/home/ubuntu/crowdvision')
CKPT_ROOT = REPO_ROOT / 'checkpoints'

CFG = {
    'ped': 'ped2',
    'clip_len': 5,
    'batch_size': 32,
    'lr': 0.0002,
    'weight_decay': 1e-5,
    'epochs_ffnet': 80,
    'patience': 25,
    'num_workers': 4,
}

train_loader, test_loader = get_ucsd_loaders(
    ped=CFG['ped'],
    data_root=str(REPO_ROOT / 'data'),
    clip_len=CFG['clip_len'],
    batch_size=CFG['batch_size'],
    num_workers=CFG['num_workers']
)

print('=' * 60)
print('TRAINING FutureFrameNet')
print('=' * 60)

model_ffnet = FutureFrameNet(
    num_input_frames=CFG['clip_len'] - 1,
    in_channels=1,
    base_ch=32,
).to(DEVICE)
print(f'FutureFrameNet parameters: {sum(p.numel() for p in model_ffnet.parameters()):,}')

opt_ffn   = optim.Adam(model_ffnet.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
sched_ffn = optim.lr_scheduler.CosineAnnealingLR(opt_ffn, T_max=CFG['epochs_ffnet'], eta_min=1e-6)

trainer_ffn = AnomalyTrainer(
    model=model_ffnet, optimizer=opt_ffn, scheduler=sched_ffn, device=DEVICE,
    experiment_name=f'ffnet_{CFG["ped"]}',
    save_dir=str(CKPT_ROOT), log_dir=str(REPO_ROOT / 'runs'),
)
trainer_ffn.load_checkpoint('last.pt')

history_ffn = trainer_ffn.train(
    train_loader, test_loader,
    epochs=CFG['epochs_ffnet'], patience=CFG['patience'],
    metric_key='auc',
    lower_is_better=False,
)
