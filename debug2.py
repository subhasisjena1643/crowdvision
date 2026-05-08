import sys
import traceback
from PIL import Image
import torch

try:
    print("Loading image")
    img = Image.open('app/static/density_crowd_1.jpg').convert('RGB')
    print("Loading pipeline")
    from app.server import get_pipeline
    pipe = get_pipeline()
    print("Pipeline loaded. Running density...")
    from app.server import run_local_density
    print(run_local_density(img))
    print("Density done.")
except Exception as e:
    traceback.print_exc()
