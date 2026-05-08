"""
UCF-CC-50 dataset loader.

50 high-density crowd images with .mat annotation files.
Used as additional data for density estimation experiments.

Data structure:
  UCF_CC_50/
    1.jpg, 2.jpg, ..., 50.jpg  (or may just have _ann.mat files)
    1_ann.mat, ..., 50_ann.mat
      -> mat['annPoints']: Nx2 array of [x, y] pixel annotations
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .shanghaitech import generate_density_map


def load_ucf_cc50_points(mat_path: str) -> np.ndarray:
    """Load UCF-CC-50 annotations from .mat file."""
    mat = sio.loadmat(mat_path)
    pts = mat.get('annPoints', np.zeros((0, 2), dtype=np.float32))
    return np.array(pts, dtype=np.float32)


class UCFCC50Dataset(Dataset):
    """
    UCF-CC-50 crowd counting dataset (50 images).

    Typically used with 5-fold cross-validation.

    Args:
        root:    path to UCF_CC_50/
        indices: list of image indices (1..50) to include; None = all
        transform: image transform
        target_size: (H, W) resize
    """

    def __init__(self, root: str,
                 indices: Optional[List[int]] = None,
                 transform=None,
                 target_size=(576, 768)):
        self.root = Path(root)
        self.transform = transform
        self.target_size = target_size

        if indices is None:
            indices = list(range(1, 51))

        # Collect img + mat pairs
        self.samples = []
        for i in indices:
            mat_path = self.root / f'{i}_ann.mat'
            # Image might be jpg or not exist (some versions have only annotations)
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                p = self.root / f'{i}{ext}'
                if p.exists():
                    img_path = p
                    break
            if mat_path.exists():
                self.samples.append({'img': img_path, 'mat': str(mat_path), 'id': i})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pts = load_ucf_cc50_points(sample['mat'])

        if sample['img'] is not None:
            img = Image.open(sample['img']).convert('RGB')
        else:
            # Create a placeholder if image is missing
            h = (self.target_size[0] if self.target_size else 576)
            w = (self.target_size[1] if self.target_size else 768)
            img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

        if self.target_size is not None:
            orig_w, orig_h = img.size
            new_h, new_w = self.target_size
            img = img.resize((new_w, new_h), Image.BILINEAR)
            if len(pts) > 0:
                pts[:, 0] *= new_w / orig_w
                pts[:, 1] *= new_h / orig_h

        w, h = img.size
        density = generate_density_map((h, w), pts)

        img_t = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img_t, torch.from_numpy(density).unsqueeze(0), torch.tensor(float(len(pts)))
