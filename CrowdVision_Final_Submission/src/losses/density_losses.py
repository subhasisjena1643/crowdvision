"""
Loss functions for crowd density estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class MSEDensityLoss(nn.Module):
    """Standard MSE loss on density maps (baseline)."""
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity loss for comparing density maps.
    Returns 1 - SSIM (minimise).
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, C1=1e-4, C2=9e-4):
        super().__init__()
        self.ws = window_size
        self.C1 = C1
        self.C2 = C2
        # Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        kern = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kern = kern / kern.sum()
        kernel_2d = kern.unsqueeze(0) * kern.unsqueeze(1)
        self.register_buffer('kernel',
                             kernel_2d.unsqueeze(0).unsqueeze(0))  # 1x1xHxW

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kernel = self.kernel.expand(C, 1, -1, -1)
        pad = self.ws // 2

        mu_x  = F.conv2d(x, kernel, padding=pad, groups=C)
        mu_y  = F.conv2d(y, kernel, padding=pad, groups=C)
        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_xx = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_xx
        sig_yy = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_yy
        sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

        ssim_num = (2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)
        ssim_den = (mu_xx + mu_yy + self.C1) * (sig_xx + sig_yy + self.C2)
        ssim_map = ssim_num / ssim_den
        return 1.0 - ssim_map.mean()


class BayesianLoss(nn.Module):
    """
    Bayesian Loss for crowd counting (Ma et al., ICCV 2019).

    Treats each annotated point as a Gaussian distribution and computes
    the log likelihood of the predicted density map under that distribution.
    """
    def __init__(self, sigma: float = 8.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        """
        Simplified version: use target density map as the expected map.
        Full Bayesian version requires point coordinates at training time.
        """
        pred_count = pred.flatten(1).sum(1)
        target_count = counts.float()
        # Count loss (L2 on counts)
        count_loss = F.mse_loss(pred_count, target_count)
        # Density map loss (normalised)
        density_loss = F.mse_loss(pred, target)
        return density_loss + 0.01 * count_loss


class CombinedDensityLoss(nn.Module):
    """
    Combines MSE + SSIM + TV regularisation.
    This is what we use in training AdaptiveCSRNet.
    """
    def __init__(self, mse_weight: float = 1.0,
                 ssim_weight: float = 0.5,
                 tv_weight: float = 1e-4):
        super().__init__()
        self.mse_w  = mse_weight
        self.ssim_w = ssim_weight
        self.tv_w   = tv_weight
        self.ssim   = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse  = F.mse_loss(pred, target)
        ssim = self.ssim(pred, target)
        # Total variation regularisation (spatial smoothness)
        tv   = (torch.abs(pred[:, :, 1:] - pred[:, :, :-1]).mean() +
                torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean())
        return self.mse_w * mse + self.ssim_w * ssim + self.tv_w * tv
