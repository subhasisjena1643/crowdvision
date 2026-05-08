import torch
import sys
from pathlib import Path
from src.data_loaders.ucsd import get_ucsd_loaders
from src.models import ConvLSTMAE
from src.evaluation.anomaly_metrics import evaluate_anomaly_detection

DEVICE = 'cuda'

print("Loading data...")
_, test_loader = get_ucsd_loaders(
    data_root='/home/ubuntu/crowdvision/data',
    ped='ped2', clip_len=10, batch_size=32, num_workers=4
)

print("Loading model...")
# ConvLSTMAE already has a MemoryModule built into its __init__ — no need
# to create a separate one or monkey-patch the forward method.  The model's
# native forward already routes the temporal bottleneck through the memory.
model = ConvLSTMAE(in_channels=1, base_ch=32, t_steps=10).to(DEVICE)

# BaseTrainer saves state_dict under the 'model' key
ckpt = torch.load('/home/ubuntu/crowdvision/checkpoints/lstm_ped2/best.pt',
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt.get('model', ckpt))
model.eval()

print("Evaluating...")
res = evaluate_anomaly_detection(model, None, test_loader, DEVICE)
print(f"AUC: {res['auc']:.2f}%")
print(f"EER: {res['eer']:.2f}%")
print(f"AP:  {res['ap']:.2f}%")
print(f"F1:  {res['f1']:.2f}%")
