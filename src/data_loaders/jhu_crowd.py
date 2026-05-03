"""
JHU-Crowd++ v2.0 dataset loader.

Data structure:
  jhu_crowd_v2.0/
    train/
      images/          <- 2722 images
      gt/              <- <img_id>.txt per image  (x y sigma_x sigma_y label)
      image_labels.txt <- img_name count weather lighting density_level
    val/   (same)
    test/  (same)

GT label codes: 0=visible head, 1=occluded head, 2=uncertain
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .shanghaitech import generate_density_map   # reuse Gaussian helper


def load_jhu_gt(gt_path: str) -> np.ndarray:
    """
    Load JHU-Crowd++ annotations.

    Returns:
        pts: Nx2 float32 array of (x, y) pixel coordinates
    """
    pts = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


class JHUCrowdDataset(Dataset):
    """
    JHU-Crowd++ crowd counting & density estimation dataset.

    Args:
        root:        path to jhu_crowd_v2.0/
        split:       'train', 'val', or 'test'
        transform:   torchvision transform for the RGB image
        target_size: (H, W) resize; None → original resolution
        adaptive:    adaptive Gaussian kernels
    """

    def __init__(self, root: str, split: str = 'train',
                 transform=None,
                 target_size: Optional[Tuple[int, int]] = (576, 768),
                 adaptive: bool = True):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.adaptive = adaptive

        split_dir = self.root / split
        self.img_dir = split_dir / 'images'
        self.gt_dir = split_dir / 'gt'

        label_file = split_dir / 'image_labels.txt'
        self.records = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    count = int(parts[1])
                    self.records.append({'name': img_name, 'count': count})

        # Verify images exist; drop missing ones gracefully
        valid = []
        for rec in self.records:
            img_path = self.img_dir / rec['name']
            if not img_path.exists():
                img_path = self.img_dir / (rec['name'] + '.jpg')
            if img_path.exists():
                rec['img_path'] = str(img_path)
                # GT file uses img stem
                stem = Path(img_path).stem
                rec['gt_path'] = str(self.gt_dir / f'{stem}.txt')
                valid.append(rec)
        self.records = valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec['img_path']).convert('RGB')
        pts = load_jhu_gt(rec['gt_path']) if Path(rec['gt_path']).exists() else np.zeros((0, 2))

        if self.target_size is not None:
            orig_w, orig_h = img.size
            new_h, new_w = self.target_size
            img = img.resize((new_w, new_h), Image.BILINEAR)
            if len(pts) > 0:
                pts[:, 0] *= new_w / orig_w
                pts[:, 1] *= new_h / orig_h

        w, h = img.size
        density = generate_density_map((h, w), pts, adaptive=self.adaptive)

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        density_t = torch.from_numpy(density).unsqueeze(0)
        count = torch.tensor(float(len(pts)))
        return img_t, density_t, count


def get_jhu_loaders(data_root: str, batch_size: int = 8,
                    target_size=(576, 768), num_workers: int = 4):
    """Return (train_loader, val_loader, test_loader) for JHU-Crowd++."""
    root = Path(data_root) / 'jhu_crowd_v2.0'

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])

    loaders = {}
    for split, tf, bs, shuf in [
        ('train', train_tf, batch_size, True),
        ('val',   eval_tf,  1, False),
        ('test',  eval_tf,  1, False),
    ]:
        ds = JHUCrowdDataset(root, split, tf, target_size)
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=shuf,
            num_workers=num_workers, pin_memory=True)

    return loaders['train'], loaders['val'], loaders['test']
