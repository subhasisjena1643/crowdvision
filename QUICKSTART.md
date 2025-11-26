# CrowdVision Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for model training and inference)
- Git

## Installation

### 1. Clone and Navigate

```bash
cd crowdvision
```

### 2. Run Setup Script

```bash
python scripts/setup_env.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Set up environment variables

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Configure Environment

Edit the `.env` file with your configuration:

```bash
# Required: Add your OpenAI API key
OPENAI_API_KEY=your_key_here

# Optional: Adjust other settings as needed
DEVICE=cuda  # or cpu if no GPU available
```

### 5. Start MLflow Server

In a separate terminal:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then initialize MLflow experiments:

```bash
python scripts/init_mlflow.py
```

Access MLflow UI at: http://localhost:5000

## Verify Installation

Run tests to verify everything is set up correctly:

```bash
pytest tests/ -v
```

## Next Steps

### Download Pre-trained Models

Place model checkpoints in the `models/` directory:

1. **YOLOv8** (Person Detection)
   ```bash
   # Will be downloaded automatically by ultralytics
   ```

2. **CSRNet** (Density Estimation)
   - Train from scratch or download pre-trained weights
   - Place in `models/csrnet_density.pth`

3. **Re-ID Model** (Person Re-identification)
   - Train on Market-1501 dataset
   - Place in `models/reid_resnet50.pth`

### Prepare Datasets

1. **ShanghaiTech Part A** (Density Estimation)
   - Download from: [ShanghaiTech Dataset](https://github.com/desenzhou/ShanghaiTechDataset)
   - Extract to `data/datasets/shanghaitech/`

2. **Market-1501** (Person Re-ID)
   - Download from: [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
   - Extract to `data/datasets/market1501/`

3. **UCSD Ped1/Ped2** (Anomaly Detection)
   - Download from: [UCSD Anomaly Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)
   - Extract to `data/datasets/ucsd/`

### Train Models

```bash
# Train density estimation model
python scripts/train_density.py

# Train forecasting model
python scripts/train_forecasting.py

# Train Re-ID model
python scripts/train_reid.py
```

### Start API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access API documentation at: http://localhost:8000/docs

## Development Workflow

1. **Write Code**: Implement features in `src/` modules
2. **Write Tests**: Add tests in `tests/` directory
3. **Run Tests**: `pytest tests/`
4. **Track Experiments**: Use MLflow for model training
5. **Commit Changes**: `git add . && git commit -m "message"`

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config/config.yaml`:

```yaml
training:
  batch_size: 4  # Reduce from 8
```

### MLflow Connection Error

Ensure MLflow server is running:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Import Errors

Ensure virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

## Support

For issues and questions, refer to the main README.md or contact the development team.
