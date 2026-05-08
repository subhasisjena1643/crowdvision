from PIL import Image
from app.server import run_local_density, run_local_anomaly
img = Image.open('app/static/density_crowd_1.jpg').convert('RGB')
print("Testing density...")
try: print(run_local_density(img))
except Exception as e: import traceback; traceback.print_exc()

print("Testing anomaly...")
try: print(run_local_anomaly(img))
except Exception as e: import traceback; traceback.print_exc()
