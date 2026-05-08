import torch
import torch.optim as optim
from pathlib import Path

from src.data_loaders.ucsd import get_ucsd_loaders
from src.models import ConvLSTMAE
from src.training import AnomalyTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

REPO_ROOT = Path('/home/ubuntu/crowdvision')
CKPT_ROOT = REPO_ROOT / 'checkpoints'
EXP_NAME = 'lstm_ped2'

CFG = {
    'ped': 'ped2',
    'clip_len': 10,
    'batch_size': 16,  # 16 is better for GPU memory with ConvLSTMAE
    'lr': 0.0002,
    'weight_decay': 1e-5,
    'epochs': 100,
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
print('TRAINING ConvLSTMAE (with MemAE)')
print('=' * 60)

model = ConvLSTMAE(in_channels=1, base_ch=32, t_steps=CFG['clip_len']).to(DEVICE)
print(f'ConvLSTMAE parameters: {sum(p.numel() for p in model.parameters()):,}')

opt = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-6)

trainer = AnomalyTrainer(
    model=model, optimizer=opt, scheduler=sched, device=DEVICE,
    experiment_name=EXP_NAME,
    save_dir=str(CKPT_ROOT), log_dir=str(REPO_ROOT / 'runs'),
)
# Try to resume if there's a checkpoint (but we'll delete it before running to start fresh)
trainer.load_checkpoint('last.pt')

history = trainer.train(
    train_loader, test_loader,
    epochs=CFG['epochs'], patience=CFG['patience'],
    metric_key='auc',
    lower_is_better=False,
)
