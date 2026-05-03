# CrowdVision

Notebook-first AI/ML pipeline for event safety at large concerts and crowd-heavy public events.

This repo is only for the ML side of the project. Frontend, backend, and product integration are handled separately.

## Quick Start

Run the notebooks in order. Each notebook is designed so you only need to execute cells.

```
notebooks/
  00_setup_and_data_check.ipynb                 ← START HERE: install deps, verify data
  01_density_estimation.ipynb                   ← Crowd density estimation on ShanghaiTech and JHU
  02_forecasting.ipynb                          ← Bottleneck and congestion forecasting on METR-LA
  03_anomaly_detection.ipynb                    ← Unsafe crowd behavior / anomaly detection on UCSD
  04_crowd_flow_and_dispatch_intelligence.ipynb ← Zone flow, bottleneck risk, and dispatch intelligence
  05_multitask_training.ipynb                   ← Unified event-safety model
  06_evaluation_and_paper_results.ipynb         ← Final evaluation, figures, and paper outputs
```

## What This Repo Does

1. Estimate crowd density from camera imagery
2. Forecast congestion and bottleneck risk
3. Detect anomalous or unsafe crowd behavior
4. Fuse these signals into zone-level safety outputs for the event team

This repo does not focus on person identity or re-identification.

## Project Structure

```
crowdvision/
├── data/                          # Datasets
├── notebooks/                     # Main execution interface
├── src/
│   ├── data_loaders/              # Dataset loaders
│   │   ├── shanghaitech.py
│   │   ├── jhu_crowd.py
│   │   ├── metr_la.py
│   │   ├── ucsd.py
│   │   ├── ucf_cc50.py
│   │   └── market1501.py
│   ├── models/
│   │   ├── density/               # CSRNet, AdaptiveCSRNet
│   │   ├── forecasting/           # GCN-GRU, AdaptiveNAS-GNN
│   │   ├── anomaly/               # ConvAE, FutureFrameNet
│   │   └── multitask/             # UnifiedCrowdVision
│   ├── losses/                    # CombinedDensityLoss, SSIM, Bayesian
│   ├── training/                  # BaseTrainer + task-specific trainers
│   ├── evaluation/                # MAE/MSE/GAME, AUC/EER, MAE/RMSE/MAPE
│   └── utils/                     # Visualisation helpers
├── configs/                       # YAML hyperparameter configs
│   ├── density/
│   ├── forecasting/
│   └── anomaly/
├── checkpoints/                   # Auto-created; saved model weights
├── experiments/                   # Auto-created; results, figures, tables
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.x, and preferably a CUDA-capable GPU for final training.

## Datasets

### Available (in `data/`)

| Dataset | Task | Size | Notes |
|---------|------|------|-------|
| ShanghaiTech A/B | Density | 482+716 images | `.mat` point annotations |
| JHU-Crowd++ v2.0 | Density | 4,372 images | train/val/test splits |
| UCF-CC-50 | Density | 50 images | Very high density (94–4,543 persons) |
| METR-LA | Forecasting | 207 sensors, 34K steps | `.h5` + `adj_mx.pkl` |
| PEMS-D3/D4/D7 | Forecasting | Various | `.npz` speed data |
| UCSD Ped1/Ped2 | Anomaly | ~100 sequences each | Frame-level anomaly labels |
| Market-1501 | Optional | 1,501 identities | Not required for the semester event-safety scope |

### Recommended Additions

| Dataset | Task | Priority | Why |
|---------|------|----------|-----|
| **UCF-QNRF** | Density | 🔴 CRITICAL | Standard CVPR benchmark (4,535 images) |
| **NWPU-Crowd** | Density | 🔴 CRITICAL | Largest crowd dataset; required for SOTA comparison |
| PEMS-BAY | Forecasting | 🟡 Recommended | 2nd standard forecasting benchmark |
| ShanghaiTech Campus | Anomaly | 🟡 Recommended | Additional anomaly benchmark |

## Reproducing Results

Everything important runs from notebooks. Training is checkpointed so interrupted runs can be resumed by rerunning cells.

## GCP / JupyterLab Setup

1. Create or open a GCP VM with JupyterLab.
2. Clone this repository into the VM.
3. Put the datasets under `crowdvision/data/` using the existing folder names.
4. Open the notebooks from the repo root and run them in order.
5. No manual code editing should be required once the data is in place.

The notebooks auto-detect the repo root, so they work in JupyterLab on GCP as long as the repo is cloned normally.

Recommended VM shapes:

1. Budget/dev: 1x NVIDIA T4, 8-16 vCPU, 32-64 GB RAM, 100-200 GB SSD
2. Best balance: 1x NVIDIA L4, 16-24 vCPU, 64-96 GB RAM, 200-500 GB SSD
3. Fastest: 1x NVIDIA A100 40GB, 24+ vCPU, 96+ GB RAM, 200-500 GB SSD

If you want the cheapest workable option, use T4 for development and L4 for final training.

**GPU memory guidance:**
- A100/H100 (40+ GB): use default `target_size = (576, 768)`, `batch_size = 8`
- RTX 3090/4090 (24 GB): use `target_size = (448, 448)`, `batch_size = 4`
- RTX 3080 (10 GB): use `target_size = (288, 384)`, `batch_size = 2`
- CPU only: notebooks auto-detect and reduce epochs to 3 for a demo run

## Output Expectations

The ML outputs that matter for the team are:

1. Density heatmaps by zone
2. Forecasted congestion or bottleneck trends
3. Anomaly alerts from video clips
4. Zone-level risk summaries that other team members can consume

## Project Focus

The project is optimized for:

1. High-quality ML outputs
2. Low-cost execution
3. Minimal manual work
4. Clear handoff to backend and frontend teammates

If you open the notebooks and run them in order, that should be enough for the ML workflow.

## Citation

```bibtex
@article{crowdvision2024,
  title   = {CrowdVision: Unified Multi-Task Learning for Crowd Analysis},
  author  = {[Your Name]},
  journal = {arXiv},
  year    = {2024},
}
```
