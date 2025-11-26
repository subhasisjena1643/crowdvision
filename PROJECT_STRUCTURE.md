# CrowdVision Project Structure

## Directory Layout

```
crowdvision/
│
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules (models, data, logs)
├── .env.example                   # Environment variables template
│
├── README.md                      # Project overview and documentation
├── QUICKSTART.md                  # Quick start guide
├── PROJECT_STRUCTURE.md           # This file
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup script
├── pytest.ini                     # Pytest configuration
├── Makefile                       # Build commands (Linux/Mac)
├── run.bat                        # Build commands (Windows)
│
├── config/
│   └── config.yaml               # Main configuration file
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── detection/                # Person detection (YOLOv8)
│   │   └── __init__.py
│   ├── density/                  # Crowd density estimation (CSRNet)
│   │   └── __init__.py
│   ├── forecasting/              # Spatiotemporal forecasting (GCN-GRU)
│   │   └── __init__.py
│   ├── anomaly/                  # Multimodal anomaly detection
│   │   └── __init__.py
│   ├── reid/                     # Person re-identification
│   │   └── __init__.py
│   ├── tracking/                 # Multi-camera tracking (DeepSORT)
│   │   └── __init__.py
│   ├── sentiment/                # Crowd sentiment analysis
│   │   └── __init__.py
│   ├── rag/                      # RAG system (LangChain + vLLM/OpenAI)
│   │   └── __init__.py
│   ├── allocation/               # ML-based resource allocation
│   │   └── __init__.py
│   └── utils/                    # Utilities and helpers
│       ├── __init__.py
│       ├── data_models.py        # Core data structures
│       └── config.py             # Configuration management
│
├── api/                          # FastAPI inference endpoints
│   └── __init__.py
│
├── tests/                        # Test suite
│   └── __init__.py
│
├── models/                       # Model checkpoints (gitignored)
│   └── checkpoints/
│
├── data/                         # Datasets (gitignored)
│   ├── raw/
│   ├── processed/
│   └── datasets/
│
├── notebooks/                    # Jupyter notebooks for experiments
│
└── scripts/                      # Utility scripts
    ├── setup_env.py              # Environment setup script
    └── init_mlflow.py            # MLflow initialization
```

## Key Files

### Configuration Files

- **config/config.yaml**: Main configuration for models, training, inference, and API
- **.env.example**: Template for environment variables (API keys, paths)
- **pytest.ini**: Test configuration with markers for unit/integration/property tests

### Core Modules

- **src/utils/data_models.py**: Data structures (Detection, Track, DensityMap, etc.)
- **src/utils/config.py**: Configuration loader with YAML and environment variable support

### Setup Scripts

- **scripts/setup_env.py**: Automated environment setup
- **scripts/init_mlflow.py**: MLflow experiment initialization

### Build Tools

- **Makefile**: Commands for Linux/Mac (make test, make run-api, etc.)
- **run.bat**: Commands for Windows (run.bat test, run.bat run-api, etc.)

## Module Responsibilities

### Detection Module (`src/detection/`)
- YOLOv8-based person detection
- Bounding box extraction
- Batch processing support

### Density Module (`src/density/`)
- CSRNet architecture for density estimation
- Zone-based crowd counting
- Occupancy calculation

### Forecasting Module (`src/forecasting/`)
- GCN-GRU spatiotemporal model
- 15-20 minute bottleneck prediction
- Graph-based venue modeling

### Anomaly Module (`src/anomaly/`)
- Multi-stream CNN (RGB + optical flow + audio)
- Fire/smoke detection
- Crowd surge detection
- Weapon detection

### Re-ID Module (`src/reid/`)
- ResNet-50 with triplet loss
- Feature embedding extraction
- Vector database integration (FAISS/ChromaDB)

### Tracking Module (`src/tracking/`)
- DeepSORT implementation
- Multi-camera tracking
- Trajectory management

### Sentiment Module (`src/sentiment/`)
- Visual cue analysis
- Audio classification
- Social media sentiment
- Multimodal fusion

### RAG Module (`src/rag/`)
- LangChain integration
- Vector database for analytics
- vLLM/OpenAI response generation
- Conversational memory

### Allocation Module (`src/allocation/`)
- ML-based resource allocation
- Route optimization
- Bottleneck-aware routing

### API Module (`api/`)
- FastAPI REST endpoints
- Request/response validation
- Async processing
- Health checks

## Dependencies

### Core ML/DL
- PyTorch 2.0+
- TorchVision
- PyTorch Lightning
- PyTorch Geometric (for GCN)

### Computer Vision
- OpenCV
- Ultralytics (YOLOv8)
- Albumentations (augmentation)

### API & Web
- FastAPI
- Uvicorn
- Pydantic

### RAG & LLM
- LangChain
- OpenAI
- Sentence Transformers

### Vector Database
- ChromaDB
- FAISS

### Experiment Tracking
- MLflow (default)
- Weights & Biases (alternative)

### Testing
- Pytest
- Pytest-asyncio
- Hypothesis (property-based testing)

## Next Steps

1. **Set up environment**: Run `python scripts/setup_env.py`
2. **Configure**: Edit `.env` with API keys
3. **Start MLflow**: Run `mlflow server --host 0.0.0.0 --port 5000`
4. **Initialize experiments**: Run `python scripts/init_mlflow.py`
5. **Download datasets**: ShanghaiTech, Market-1501, UCSD
6. **Train models**: Use training scripts (to be implemented)
7. **Start API**: Run `uvicorn api.main:app --reload`

## Development Workflow

1. Implement module functionality in `src/`
2. Write tests in `tests/`
3. Track experiments with MLflow
4. Expose functionality via API endpoints
5. Document in notebooks
6. Commit changes to Git

## Git Ignore

The following are excluded from version control:
- Model checkpoints (*.pth, *.pt, *.onnx)
- Datasets (data/raw/, data/processed/)
- Virtual environment (venv/)
- Experiment logs (mlruns/, wandb/)
- Vector databases (chroma_db/, *.faiss)
- Python cache (__pycache__/, *.pyc)
- IDE files (.vscode/, .idea/)
