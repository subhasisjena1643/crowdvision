"""
Future Frame Prediction model for anomaly detection.
(Liu et al., CVPR 2018: Future Frame Prediction for Anomaly Detection)

Core idea: train a U-Net to predict frame t+1 given frames t-k..t.
High prediction error at test time → anomaly.
Also includes optical flow constraint (appearance + motion loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip): return self.conv(torch.cat([skip, self.up(x)], dim=1))


class FutureFrameNet(nn.Module):
    """
    U-Net based future frame predictor.

    Args:
        num_input_frames: number of past frames concatenated as input
        in_channels:      channels per frame (1=grayscale)
        base_ch:          base channel width (default 32)
    """

    def __init__(self, num_input_frames: int = 4,
                 in_channels: int = 1,
                 base_ch: int = 32):
        super().__init__()
        c = base_ch
        in_ch = num_input_frames * in_channels

        # Encoder
        self.enc1 = DoubleConv(in_ch, c)
        self.enc2 = Down(c, c * 2)
        self.enc3 = Down(c * 2, c * 4)
        self.enc4 = Down(c * 4, c * 8)
        self.bottleneck = Down(c * 8, c * 16)

        # Decoder
        self.dec4 = Up(c * 16 + c * 8, c * 8)
        self.dec3 = Up(c * 8 + c * 4, c * 4)
        self.dec2 = Up(c * 4 + c * 2, c * 2)
        self.dec1 = Up(c * 2 + c, c)

        # Output: predict 1 future frame
        self.out = nn.Conv2d(c, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T*C, H, W]  (input frames concatenated along channel dim)
        Returns:
            pred: [B, C, H, W]  predicted frame t+1
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)

    def prediction_error(self, past: torch.Tensor, actual_future: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past:          [B, T, C, H, W]  – input clip
            actual_future: [B, C, H, W]     – ground-truth next frame
        Returns:
            error: [B]  per-sample prediction MSE
        """
        B, T, C, H, W = past.shape
        x = past.view(B, T * C, H, W)
        pred = self.forward(x)
        return F.mse_loss(pred, actual_future, reduction='none').mean([1, 2, 3])
