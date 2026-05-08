"""
UCSD Anomaly Dataset loader (Ped1 and Ped2).

Data structure:
  UCSD_Anomaly_Dataset.v1p2/
    UCSDped1/
      Train/
        Train001/ ... Train034/   <- sequences of .tif frames
      Test/
        Test001/ ... Test036/     <- test sequences
        Test00X_gt/               <- pixel-level GT masks (.bmp or .tif) for anomalous clips
    UCSDped2/  (same layout, 16 train / 12 test clips)

Anomaly score for a frame = mean reconstruction error of the autoencoder.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _load_frame(path: Path) -> Optional[np.ndarray]:
    """Load a single frame (tif / bmp / png / jpg) as uint8 HxW numpy array."""
    try:
        img = imageio.imread(str(path))
        if img.ndim == 3:
            img = img[..., 0]          # keep only first channel (grayscale)
        return img.astype(np.uint8)
    except Exception:
        return None


def _get_frame_paths(seq_dir: Path) -> List[Path]:
    """Return sorted list of image file paths in a sequence folder."""
    exts = {'.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg'}
    paths = sorted([p for p in seq_dir.iterdir()
                    if p.suffix.lower() in exts and not p.name.startswith('.')])
    return paths


class UCSDTrainDataset(Dataset):
    """
    All training frames from UCSD Ped1 or Ped2 for autoencoder training.
    Returns single frames — no labels needed (anomaly-free).
    """

    def __init__(self, root: str, ped: str = 'ped1',
                 transform=None, clip_len: int = 1):
        """
        Args:
            root:      path to UCSD_Anomaly_Dataset.v1p2/
            ped:       'ped1' or 'ped2'
            transform: torchvision transform (applied to each frame)
            clip_len:  number of consecutive frames per sample (1 = single frame mode)
        """
        self.transform = transform
        self.clip_len = clip_len

        dataset_dir = Path(root) / f'UCSDped{ped[-1]}' / 'Train'
        seq_dirs = sorted([d for d in dataset_dir.iterdir()
                           if d.is_dir() and not d.name.startswith('.')])

        self.clips: List[List[Path]] = []
        for seq_dir in seq_dirs:
            frames = _get_frame_paths(seq_dir)
            if clip_len == 1:
                self.clips.extend([[f] for f in frames])
            else:
                for i in range(len(frames) - clip_len + 1):
                    self.clips.append(frames[i: i + clip_len])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frames_raw = [_load_frame(p) for p in self.clips[idx]]
        frames_raw = [f for f in frames_raw if f is not None]

        imgs = []
        for fr in frames_raw:
            img = Image.fromarray(fr)
            img = self.transform(img) if self.transform else transforms.ToTensor()(img)
            imgs.append(img)

        if self.clip_len == 1:
            return imgs[0]                          # C x H x W
        return torch.stack(imgs, dim=0)             # T x C x H x W


class UCSDTestDataset(Dataset):
    """
    Test clips from UCSD Ped1 or Ped2.
    Returns (clip, labels) where labels is a 1-D bool tensor per frame.
    """

    def __init__(self, root: str, ped: str = 'ped1',
                 transform=None, clip_len: int = 1):
        self.transform = transform
        self.clip_len = clip_len

        dataset_dir = Path(root) / f'UCSDped{ped[-1]}' / 'Test'

        # Collect test sequences (ignore _gt dirs)
        seq_dirs = sorted([d for d in dataset_dir.iterdir()
                           if d.is_dir() and '_gt' not in d.name
                           and not d.name.startswith('.')])

        self.samples: List[Tuple[List[Path], List[Path]]] = []  # (frame_paths, gt_paths)
        for seq_dir in seq_dirs:
            frames = _get_frame_paths(seq_dir)
            gt_dir = dataset_dir / (seq_dir.name + '_gt')
            has_gt = gt_dir.exists()

            gt_paths = []
            for f in frames:
                gt_file = gt_dir / (f.stem + '.bmp')
                if not gt_file.exists():
                    gt_file = gt_dir / (f.stem + '.tif')
                if has_gt and gt_file.exists():
                    gt_paths.append(gt_file)
                else:
                    gt_paths.append(None)

            if clip_len == 1:
                for i in range(len(frames)):
                    self.samples.append(([frames[i]], [gt_paths[i]]))
            else:
                for i in range(len(frames) - clip_len + 1):
                    self.samples.append((frames[i: i + clip_len], gt_paths[i: i + clip_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, gt_paths = self.samples[idx]
        imgs = []
        for p in paths:
            fr = _load_frame(p)
            if fr is None:
                fr = np.zeros((158, 238), dtype=np.uint8)
            img = Image.fromarray(fr)
            img = self.transform(img) if self.transform else transforms.ToTensor()(img)
            imgs.append(img)

        # Return frame-level labels
        labels = []
        for gp in gt_paths:
            if gp is not None:
                gt_fr = _load_frame(gp)
                if gt_fr is not None and gt_fr.sum() > 0:
                    labels.append(1.0)
                else:
                    labels.append(0.0)
            else:
                labels.append(0.0)

        label = torch.tensor(labels, dtype=torch.float32)

        if self.clip_len == 1:
            return imgs[0], label.squeeze(0)   # scalar label for single frame
        return torch.stack(imgs, dim=0), label


def get_ucsd_loaders(data_root: str, ped: str = 'ped2',
                     img_size: Tuple[int, int] = (128, 192),
                     batch_size: int = 32, num_workers: int = 4,
                     clip_len: int = 1):
    """
    Return (train_loader, test_loader) for UCSD anomaly detection.

    Args:
        data_root: path that contains UCSD_Anomaly_Dataset.v1p2/
        ped:       'ped1' or 'ped2'
        img_size:  (H, W) to resize frames to
        clip_len:  frames per sample (1 = frame-level, >1 = clip-level)
    """
    # Try multiple possible paths (handles different download layouts)
    candidates = [
        Path(data_root) / 'UCSD_Anomaly_Dataset.v1p2',
        Path(data_root) / 'UCSD_Anomaly_Dataset' / 'UCSD_Anomaly_Dataset.v1p2',
    ]
    ucsd_root = next((c for c in candidates if c.exists()), candidates[0])

    h, w = img_size
    tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_ds = UCSDTrainDataset(str(ucsd_root), ped, tf, clip_len)
    test_ds = UCSDTestDataset(str(ucsd_root), ped, tf, clip_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader
