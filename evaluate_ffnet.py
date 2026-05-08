import torch
import sys
from pathlib import Path
from src.data_loaders.ucsd import get_ucsd_loaders
from src.models import FutureFrameNet
from src.evaluation.anomaly_metrics import evaluate_anomaly_detection

DEVICE = 'cuda'

print("Loading data...")
_, test_loader = get_ucsd_loaders(
    data_root='/home/ubuntu/crowdvision/data',
    ped='ped2', clip_len=5, batch_size=32, num_workers=4
)

print("Loading model...")
model = FutureFrameNet(num_input_frames=4, in_channels=1, base_ch=32).to(DEVICE)
ckpt = torch.load('/home/ubuntu/crowdvision/checkpoints/ffnet_ped2/last.pt', map_location=DEVICE)
# BaseTrainer saves state_dict under the 'model' key
model.load_state_dict(ckpt.get('model', ckpt))
model.eval()

print("Evaluating...")
res = evaluate_anomaly_detection(model, None, test_loader, DEVICE)
print(f"AUC: {res['auc']:.2f}%")
print(f"EER: {res['eer']:.2f}%")
print(f"AP:  {res['ap']:.2f}%")
print(f"F1:  {res['f1']:.2f}%")
