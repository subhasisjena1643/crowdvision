"""
ShanghaiTech Part A and Part B dataset loader.

Data structure:
  part_A_final/
    train_data/images/IMG_*.jpg
    train_data/ground_truth/GT_IMG_*.mat   <- mat['image_info'][0,0][0][0][0] = Nx2 [x,y]
    test_data/images/IMG_*.jpg
    test_data/ground_truth/GT_IMG_*.mat
  part_B_final/  (same structure)
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def gaussian_kernel(size: int = 15, sigma: float = 4.0) -> np.ndarray:
    """Return a 2-D Gaussian kernel."""
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def generate_density_map(img_shape: tuple, points: np.ndarray,
                          adaptive: bool = True) -> np.ndarray:
    """
    Generate a density map from point annotations.

    Args:
        img_shape: (H, W) of the image
        points:    Nx2 array of (x, y) pixel coordinates
        adaptive:  If True, adapt sigma to nearest-neighbour distance

    Returns:
        density: HxW float32 array where density.sum() ≈ len(points)
    """
    h, w = img_shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    # Clip points to valid image range
    pts = points.copy().astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    for i, (px, py) in enumerate(pts):
        if adaptive and len(pts) > 1:
            # Euclidean distance to 3 nearest neighbours → adaptive sigma
            dists = np.sqrt(((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)
                             .astype(float))
            dists[i] = np.inf  # exclude self
            k = min(3, len(pts) - 1)
            sigma = float(np.sort(dists)[:k].mean()) * 0.3
            sigma = max(sigma, 1.0)
        else:
            sigma = 4.0

        size = int(6 * sigma + 1) | 1  # ensure odd
        kernel = gaussian_kernel(size, sigma)

        # Bounding box for the kernel centred at (px, py)
        x0 = px - size // 2
        y0 = py - size // 2
        x1 = x0 + size
        y1 = y0 + size

        # Clamp to image boundaries
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x1, w), min(y1, h)

        kx0 = ix0 - x0
        ky0 = iy0 - y0
        density[iy0:iy1, ix0:ix1] += kernel[ky0:ky0 + (iy1 - iy0),
                                             kx0:kx0 + (ix1 - ix0)]
    return density


def load_mat_points(mat_path: str) -> np.ndarray:
    """Load (x, y) point annotations from a ShanghaiTech .mat file."""
    mat = sio.loadmat(mat_path)
    try:
        pts = mat['image_info'][0, 0][0][0][0]
    except (KeyError, IndexError):
        # Try alternative key layout
        pts = mat.get('annPoints', np.zeros((0, 2), dtype=np.float32))
    return np.array(pts, dtype=np.float32)


class ShanghaiTechDataset(Dataset):
    """
    Dataset for ShanghaiTech Part A or Part B crowd density estimation.

    Args:
        root:      path to data folder, e.g. 'data/part_A_final'
        split:     'train' or 'test'
        transform: torchvision transform applied to the RGB image
        target_size: (H, W) to resize images/density maps; None keeps original
        adaptive:  use adaptive Gaussian sigma
    """

    def __init__(self, root: str, split: str = 'train',
                 transform=None, target_size=None, adaptive: bool = True):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.adaptive = adaptive

        data_dir = self.root / f'{split}_data'
        img_dir = data_dir / 'images'
        gt_dir = data_dir / 'ground_truth'

        self.img_paths = sorted(img_dir.glob('IMG_*.jpg'))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found at {img_dir}")

        self.gt_paths = []
        for ip in self.img_paths:
            stem = ip.stem                    # 'IMG_1'
            gt_name = f'GT_{stem}.mat'
            self.gt_paths.append(gt_dir / gt_name)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        pts = load_mat_points(str(self.gt_paths[idx]))

        if self.target_size is not None:
            orig_w, orig_h = img.size
            new_h, new_w = self.target_size
            img = img.resize((new_w, new_h), Image.BILINEAR)
            sx = new_w / orig_w
            sy = new_h / orig_h
            if len(pts) > 0:
                pts[:, 0] *= sx
                pts[:, 1] *= sy

        w, h = img.size
        density = generate_density_map((h, w), pts, adaptive=self.adaptive)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        density_t = torch.from_numpy(density).unsqueeze(0)  # 1xHxW
        count = torch.tensor(float(len(pts)))
        return img, density_t, count

    @property
    def name(self):
        part = 'A' if 'part_A' in str(self.root) else 'B'
        return f'ShanghaiTech-{part}'


def get_shanghaitech_loaders(data_root: str, part: str = 'A',
                              batch_size: int = 8,
                              target_size=(576, 768),
                              num_workers: int = 4):
    """
    Convenience factory that returns (train_loader, test_loader).

    Args:
        data_root: path containing data/ directory
        part:      'A' or 'B'
        batch_size: samples per batch
        target_size: (H, W) to resize to; set None for variable-size
    """
    data_root = Path(data_root)
    candidates = [
        data_root / f'part_{part}_final',
        data_root / f'shanghaitech_part_{part}_final',
    ]
    root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    train_tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ShanghaiTechDataset(root, 'train', train_tf, target_size)
    test_ds = ShanghaiTechDataset(root, 'test', test_tf, target_size)

    # Collate function handles variable-size if target_size is None
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
