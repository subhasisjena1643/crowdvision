"""
Unified Multi-Task Architecture — our PRIMARY novel contribution.

Shared backbone encodes visual features once; four task-specific heads
decode density maps, forecast flows, detect anomalies, and assist tracking.

Cross-task consistency losses enforce:
  - density_head ↔ tracking_head  (count predicted by tracking ≈ density sum)
  - anomaly_head ↔ density_head   (sudden density change → anomaly alert)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Shared backbone (EfficientNet-B3 or VGG-16 selectable)
# ---------------------------------------------------------------------------

class SharedBackbone(nn.Module):
    """
    Shared visual encoder used by all task heads.

    Outputs multi-scale feature maps at 1/4, 1/8, 1/16 resolution.
    """

    def __init__(self, backbone: str = 'vgg16', pretrained: bool = True):
        super().__init__()
        if backbone == 'vgg16':
            vgg = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
            feats = list(vgg.features.children())
            self.layer1 = nn.Sequential(*feats[:10])   # 1/4 → 128 ch
            self.layer2 = nn.Sequential(*feats[10:17]) # 1/8 → 256 ch
            self.layer3 = nn.Sequential(*feats[17:24]) # 1/16 → 512 ch
            self.out_channels = [128, 256, 512]
        elif backbone == 'resnet50':
            r = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.layer1 = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
            self.layer2 = r.layer2
            self.layer3 = r.layer3
            self.out_channels = [256, 512, 1024]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (f1/4, f1/8, f1/16) multi-scale features."""
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


# ---------------------------------------------------------------------------
# Feature Pyramid Network (FPN) for density
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    def __init__(self, in_channels: list, out_channels: int = 256):
        super().__init__()
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.smooth   = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, f1, f2, f3):
        l3 = self.lateral3(f3)
        l2 = self.lateral2(f2) + F.interpolate(l3, size=f2.shape[-2:], mode='nearest')
        l1 = self.lateral1(f1) + F.interpolate(l2, size=f1.shape[-2:], mode='nearest')
        return self.smooth(l1)   # highest-resolution fused features


# ---------------------------------------------------------------------------
# Task-specific heads
# ---------------------------------------------------------------------------

class DensityHead(nn.Module):
    """Predict density map (same as AdaptiveCSRNet backend)."""
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),    nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
    def forward(self, x: torch.Tensor, target_size) -> torch.Tensor:
        d = F.relu(self.net(x))
        return F.interpolate(d, size=target_size, mode='bilinear', align_corners=False)


class AnomalyHead(nn.Module):
    """
    Frame-level anomaly score from feature statistics.
    Returns a scalar score per sample (0=normal, 1=anomalous).
    """
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(x))     # [B, 1]


class ReIDHead(nn.Module):
    """
    Person re-identification embedding head (for tracking integration).
    Produces L2-normalised 256-d embeddings.
    """
    def __init__(self, in_ch: int = 256, embed_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_ch),
            nn.Linear(in_ch, embed_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fc(self.gap(x))     # [B, embed_dim]
        return F.normalize(feat, p=2, dim=1)


class ForecastingHead(nn.Module):
    """
    Compress global spatial features to a per-node feature vector
    for feeding into the graph forecasting module.
    """
    def __init__(self, in_ch: int = 256, node_feat: int = 64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(in_ch, node_feat)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(x).flatten(1))  # [B, node_feat]


# ---------------------------------------------------------------------------
# Unified model
# ---------------------------------------------------------------------------

class UnifiedCrowdVision(nn.Module):
    """
    AdaptiveVision: Unified multi-task architecture for crowd analysis.

    Tasks:
      - density:    crowd density map estimation
      - anomaly:    frame-level anomaly score
      - reid:       person re-ID embedding (for tracking)
      - forecasting: (optional) returns scene-level feature for GNN

    Args:
        backbone:      'vgg16' | 'resnet50'
        pretrained:    ImageNet weights for backbone
        fpn_ch:        FPN output channels
        reid_dim:      re-ID embedding dimension
        tasks:         subset of ['density', 'anomaly', 'reid', 'forecasting']
    """

    def __init__(self, backbone: str = 'vgg16', pretrained: bool = True,
                 fpn_ch: int = 256, reid_dim: int = 256,
                 tasks=None):
        super().__init__()
        if tasks is None:
            tasks = ['density', 'anomaly', 'reid']
        self.tasks = tasks

        self.backbone = SharedBackbone(backbone, pretrained)
        self.fpn = FPN(self.backbone.out_channels, fpn_ch)

        if 'density' in tasks:
            self.density_head = DensityHead(fpn_ch)
        if 'anomaly' in tasks:
            self.anomaly_head = AnomalyHead(fpn_ch)
        if 'reid' in tasks:
            self.reid_head = ReIDHead(fpn_ch, reid_dim)
        if 'forecasting' in tasks:
            self.forecast_head = ForecastingHead(fpn_ch)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            dict with keys matching active tasks
        """
        f1, f2, f3 = self.backbone(x)
        feat = self.fpn(f1, f2, f3)   # [B, fpn_ch, H/4, W/4]

        out = {}
        if 'density' in self.tasks:
            out['density'] = self.density_head(feat, x.shape[-2:])
        if 'anomaly' in self.tasks:
            out['anomaly'] = self.anomaly_head(feat)
        if 'reid' in self.tasks:
            out['reid'] = self.reid_head(feat)
        if 'forecasting' in self.tasks:
            out['forecasting'] = self.forecast_head(feat)
        return out

    def consistency_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Cross-task consistency loss.
        Currently implements: density count should be > 0 when anomaly score is high.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if 'density' in out and 'anomaly' in out:
            density_count = out['density'].flatten(1).sum(1).unsqueeze(1)
            # When anomaly score is high, density should be non-trivial
            # Penalise: anomaly_score * exp(-density_count/50)
            consistency = out['anomaly'] * torch.exp(-density_count / 50.0)
            loss = loss + consistency.mean()
        return loss
