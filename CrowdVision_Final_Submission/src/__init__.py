"""CrowdVision: Unified Multi-Task Crowd Analysis Framework."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CONFIGS_DIR = ROOT_DIR / "configs"

CHECKPOINTS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)
