# CrowdVision (AIML) Architecture and Workflow

This document explains the technical architecture and end-to-end processes of the CrowdVision AIML subsystem for presentation use. It focuses only on the ML pipeline (datasets, training, evaluation, and outputs).

## 1. System Purpose (AIML scope)

CrowdVision produces crowd-safety intelligence from multimodal data:
- Crowd density estimation from images
- Congestion forecasting from sensor time-series
- Anomaly detection from video clips
- Multi-task fusion for unified event-safety signals

Outputs are designed for downstream product teams, but this repository implements only the ML layer.

## 2. High-Level Architecture (Components)

Core components:
- Notebooks: orchestration and experiment execution
- Data loaders: dataset parsing and batching
- Preprocessing: resize, normalize, density map generation
- Models: task-specific and multi-task networks
- Losses: per-task and regularization terms
- Trainers: optimization, scheduling, checkpointing
- Evaluation: metrics for each task
- Visualization: plots, heatmaps, tables
- Artifacts: checkpoints, logs, figures, reports

## 3. Data Sources and Formats

### 3.1 Density Estimation
- ShanghaiTech A/B: images + point annotations (.mat)
- JHU-Crowd++: images + point annotations (.txt)
- UCF-CC-50: high-density images + annotations (.mat)

### 3.2 Forecasting
- METR-LA: sensor time-series (.h5) + adjacency (.pkl)
- PEMS D3/D4/D7: time-series (.npz) + distance (.csv)

### 3.3 Anomaly Detection
- UCSD Ped1/Ped2: frame sequences (.tif/.bmp/.png) + anomaly labels

### 3.4 Optional ReID
- Market-1501: person ID images (optional, not required for event safety)

## 4. End-to-End Workflow (Notebook Pipeline)

The system is executed through notebooks in order:

1) 00_setup_and_data_check
- Installs dependencies
- Verifies GPU availability
- Validates datasets and prints statistics

2) 01_density_estimation
- Trains CSRNet baseline
- Trains AdaptiveCSRNet (attention + multi-scale)
- Evaluates MAE/MSE/PSNR/SSIM/GAME

3) 02_forecasting
- Trains GCN-GRU baseline
- Runs NAS search (AdaptiveNAS-GNN)
- Retrains with fixed architecture
- Evaluates MAE/RMSE/MAPE at horizons

4) 03_anomaly_detection
- Trains ConvAE (reconstruction-based)
- Trains FutureFrameNet (prediction-based)
- Evaluates AUC/EER/AP/F1

5) 04_crowd_flow_and_dispatch_intelligence
- Loads trained models
- Produces zone-level density + risk summaries
- Demonstrates dispatch prioritization logic

6) 05_multitask_training
- Trains UnifiedCrowdVision (density + anomaly + optional reid)
- Applies cross-task consistency regularization

7) 06_evaluation_and_paper_results
- Aggregates final metrics
- Builds paper-ready tables and figures

## 5. Task-Specific Model Details

### 5.1 Density Estimation
- CSRNet: VGG16 frontend + dilated backend
- AdaptiveCSRNet (novel): CBAM attention + ASPP backend + perspective head

### 5.2 Forecasting
- GCN-GRU: graph convolution + GRU temporal model
- AdaptiveNAS-GNN (novel): DARTS-style search over graph ops and temporal dilation

### 5.3 Anomaly Detection
- ConvAE: single-frame reconstruction
- ConvLSTMAE: clip-level autoencoder (optional)
- FutureFrameNet: future frame prediction

### 5.4 Unified Multi-task Model
- Shared backbone (VGG/ResNet) + FPN
- Heads: density, anomaly, reid, forecasting features
- Consistency loss ties anomaly score to density signal

## 6. Training Flow (Generic)

1) Load dataset loaders
2) Forward pass through model
3) Compute task loss
4) Backprop + optimizer step
5) Scheduler update
6) Save checkpoints (best + last)
7) Evaluate metrics on validation/test

## 7. Metrics and Evaluation

### Density
- MAE, MSE (count error)
- PSNR, SSIM (map quality)
- GAME (spatial grid error)

### Forecasting
- MAE, RMSE, MAPE
- Evaluated at 15, 30, 60 minutes

### Anomaly
- AUC-ROC, EER, AP, F1

## 8. Artifacts and Outputs

- Checkpoints: model states saved per experiment
- Experiments: figures, tables, plots
- Runs: TensorBoard logs

Primary ML outputs for downstream teams:
- Density heatmaps by zone
- Forecasted congestion trends
- Anomaly alerts (scores)
- Zone risk summaries

## 9. Key Design Notes

- Notebook-first workflow (no CLI required)
- Modular codebase for task isolation
- All experiments reproducible via configs + checkpoints
- Multi-task training includes explicit consistency regularization

## 10. Suggested Presentation Flow

1) Problem statement (crowd safety)
2) Data sources
3) Model zoo overview
4) Training/evaluation pipeline
5) Unified model contribution
6) Key outputs and deliverables

---

For diagrams, see the Mermaid charts in the presentation slides or supporting docs.
