"""
CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly
Congested Scenes (CVPR 2018).

Architecture:
  - Frontend:  VGG-16 conv1_1 → pool3  (10 conv layers, 3 max-pools)
  - Backend:   6 dilated conv layers with rates [2,2,2,4,4,4]
  - Output:    1/8 resolution density map (bilinear up-scaled to full res)

Reference: https://arxiv.org/abs/1802.10062
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet(nn.Module):
    """
    CSRNet crowd density estimation model.

    Args:
        load_weights: if True, initialise frontend with ImageNet VGG-16 weights
    """

    def __init__(self, load_weights: bool = True):
        super().__init__()

        # Frontend: VGG-16 features up to pool3 (layers 0..23 inclusive)
        vgg = models.vgg16(weights='IMAGENET1K_V1' if load_weights else None)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[:23])   # → 1/8 spatial

        # Backend: dilated conv layers
        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
        )

        # Output head: 1 density channel
        self.output_layer = nn.Conv2d(64, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalised RGB image
        Returns:
            density: [B, 1, H, W] — upsampled to input resolution
        """
        h, w = x.shape[-2:]
        feat = self.frontend(x)        # [B, 256, H/8, W/8]
        feat = self.backend(feat)      # [B, 64, H/8, W/8]
        density = self.output_layer(feat)   # [B, 1, H/8, W/8]
        density = F.interpolate(density, size=(h, w), mode='bilinear',
                                align_corners=False)
        return density

    def count(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted crowd count (sum of density map)."""
        density = self.forward(x)
        return density.flatten(1).sum(1)


class CSRNetLite(nn.Module):
    """
    Lightweight CSRNet variant (no pre-trained VGG, fewer parameters).
    Useful for rapid prototyping and ablation studies.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feat = self.encoder(x)
        density = self.decoder(feat)
        return F.interpolate(density, size=(h, w), mode='bilinear',
                             align_corners=False)

    def count(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).flatten(1).sum(1)
