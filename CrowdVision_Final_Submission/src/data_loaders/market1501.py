"""
Market-1501 person re-identification dataset loader.

Used to train the appearance feature extractor for DeepSORT tracking.

Data structure:
  Market-1501-v15.09.15/
    bounding_box_train/  <- 12,936 images from 751 identities
    bounding_box_test/   <- 19,732 images from 750 identities (gallery)
    query/               <- 3,368 images from 750 identities
    gt_bbox/
    gt_query/

Image naming: PPPP_CCSSSxFFF_DD.jpg
  PPPP = person ID (0001..0750 or -1 for distractors)
  CC   = camera ID (c1..c6)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _parse_market_filename(filename: str) -> Tuple[int, int]:
    """Return (person_id, camera_id) from a Market-1501 filename."""
    stem = Path(filename).stem
    pid = int(stem[:4])
    cam = int(stem[6]) - 1   # c1..c6 → 0..5
    return pid, cam


class Market1501Dataset(Dataset):
    """
    Market-1501 for supervised person re-identification.

    Each item: (image_tensor, person_id, camera_id)
    """

    def __init__(self, root: str, split: str = 'train',
                 transform=None, relabel: bool = True):
        """
        Args:
            root:      path to Market-1501-v15.09.15/
            split:     'train' | 'query' | 'gallery'
            transform: torchvision transform
            relabel:   remap PIDs to 0..N-1 for CE loss (train only)
        """
        self.transform = transform
        dir_map = {
            'train':   'bounding_box_train',
            'gallery': 'bounding_box_test',
            'query':   'query',
        }
        img_dir = Path(root) / dir_map[split]
        imgs = sorted(img_dir.glob('*.jpg'))

        self.samples: List[Tuple[str, int, int]] = []
        pids_seen: Dict[int, int] = {}
        for p in imgs:
            pid, cam = _parse_market_filename(p.name)
            if pid < 0:        # distractor / junk (-1)
                continue
            if relabel and split == 'train':
                if pid not in pids_seen:
                    pids_seen[pid] = len(pids_seen)
                pid = pids_seen[pid]
            self.samples.append((str(p), pid, cam))

        self.num_classes = len(pids_seen) if relabel and split == 'train' else len(
            {s[1] for s in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid, cam = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(pid), torch.tensor(cam)


def get_market1501_loaders(data_root: str, batch_size: int = 64,
                            num_workers: int = 4):
    """Return (train_loader, query_loader, gallery_loader, num_classes)."""
    root = Path(data_root) / 'Market-1501-v15.09.15' / 'Market-1501-v15.09.15'

    train_tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = Market1501Dataset(root, 'train', train_tf)
    query_ds = Market1501Dataset(root, 'query', eval_tf, relabel=False)
    gallery_ds = Market1501Dataset(root, 'gallery', eval_tf, relabel=False)

    train_loader  = DataLoader(train_ds,   batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
    query_loader  = DataLoader(query_ds,   batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_loader, query_loader, gallery_loader, train_ds.num_classes
