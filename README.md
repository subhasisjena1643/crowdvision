# CrowdVision

AI/ML-powered situational awareness platform for event safety at large-scale public gatherings.

## Overview

CrowdVision provides real-time crowd monitoring and safety analysis through:

- **Crowd Density Estimation**: CNN-based models for real-time person detection and density mapping
- **Spatiotemporal Forecasting**: GCN-GRU models for 15-20 minute bottleneck prediction
- **Multimodal Anomaly Detection**: Multi-stream CNNs for fire, smoke, surge, and weapon detection
- **Person Re-Identification**: Deep metric learning for lost person search across camera feeds
- **Natural Language Intelligence**: RAG system using vLLM/LangChain and OpenAI for conversational queries

## Project Structure

```
crowdvision/
├── src/                    # Source code
│   ├── detection/         # Person detection (YOLOv8)
│   ├── density/           # Crowd density estimation (CSRNet)
│   ├── forecasting/       # Spatiotemporal forecasting (GCN-GRU)
│   ├── anomaly/           # Anomaly detection
│   ├── reid/              # Person re-identification
│   ├── tracking/          # Multi-camera tracking (DeepSORT)
│   ├── sentiment/         # Crowd sentiment analysis
│   ├── rag/               # RAG system (LangChain)
│   ├── allocation/        # Resource allocation
│   └── utils/             # Utilities and data models
├── api/                   # FastAPI inference endpoints
├── models/                # Model checkpoints (gitignored)
├── data/                  # Datasets (gitignored)
├── tests/                 # Test suite
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
├── scripts/               # Training and utility scripts
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 4. Initialize MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Development

### Running Tests

```bash
pytest tests/
```

### Starting API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Training Models

```bash
python scripts/train_density.py
python scripts/train_forecasting.py
python scripts/train_reid.py
```

## Model Checkpoints

Pre-trained model checkpoints should be placed in the `models/` directory:

- `yolov8n.pt` - Person detection
- `csrnet_density.pth` - Density estimation
- `gcn_gru.pth` - Spatiotemporal forecasting
- `anomaly_detector.pth` - Anomaly detection
- `reid_resnet50.pth` - Person re-identification

## API Endpoints

- `POST /api/v1/density/estimate` - Crowd density estimation
- `POST /api/v1/bottleneck/predict` - Bottleneck prediction
- `POST /api/v1/anomaly/detect` - Anomaly detection
- `POST /api/v1/reid/search` - Person re-identification search
- `POST /api/v1/query` - Natural language query
- `POST /api/v1/allocate` - Resource allocation

## License

Proprietary - All rights reserved

## Contact

For questions and support, contact the CrowdVision team.
