"""
Convolutional Autoencoder for video anomaly detection.

Trained only on normal frames, high reconstruction error = anomaly.

Architectures:
  1. ConvAE         – standard encoder-decoder (frame-level)
  2. ConvLSTMAE     – spatiotemporal LSTM autoencoder (clip-level)
  3. FutureFrameNet – predict future frame from past frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv_block(in_ch: int, out_ch: int, downsample: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    if downsample:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


def deconv_block(in_ch: int, out_ch: int, upsample: bool = True) -> nn.Sequential:
    layers = []
    if upsample:
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
    layers += [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 1. Frame-level convolutional autoencoder
# ---------------------------------------------------------------------------

class ConvAE(nn.Module):
    """
    Convolutional autoencoder for single-frame anomaly detection.

    Args:
        in_channels: 1 for grayscale, 3 for RGB
        base_ch:     base number of channels (default 32)
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        c = base_ch

        # Encoder: 128x192 → 16x24 (3 max-pools)
        self.encoder = nn.Sequential(
            conv_block(in_channels, c),        # H/2
            conv_block(c, c * 2),              # H/4
            conv_block(c * 2, c * 4),          # H/8
            conv_block(c * 4, c * 8, False),   # no pool
        )

        # Bottleneck code dim
        self.code_ch = c * 8

        # Decoder
        self.decoder = nn.Sequential(
            deconv_block(c * 8, c * 4),        # H/4
            deconv_block(c * 4, c * 2),        # H/2
            deconv_block(c * 2, c),            # H
            nn.Conv2d(c, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed frame, same shape as input."""
        code = self.encoder(x)
        return self.decoder(code)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample mean squared reconstruction error.
        Higher = more anomalous.
        """
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2, 3])

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for reconstruction_error, normalised to [0,1] via sigmoid."""
        err = self.reconstruction_error(x)
        return torch.sigmoid((err - err.mean()) / (err.std() + 1e-8))


# ---------------------------------------------------------------------------
# ConvLSTM cell
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch, h, w, device):
        return (torch.zeros(batch, self.hidden_ch, h, w, device=device),
                torch.zeros(batch, self.hidden_ch, h, w, device=device))


# ---------------------------------------------------------------------------
# 2. Spatiotemporal ConvLSTM autoencoder for clip-level anomaly detection
# ---------------------------------------------------------------------------

class ConvLSTMAE(nn.Module):
    """
    Spatiotemporal autoencoder for video clip anomaly detection.
    Encodes a T-frame clip into a latent representation, then decodes it.

    Args:
        in_channels: 1 (grayscale) or 3 (RGB)
        base_ch:     base channels
        t_steps:     number of input frames
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32, t_steps: int = 10):
        super().__init__()
        self.t_steps = t_steps
        c = base_ch

        # Spatial encoder (shared per frame)
        self.spatial_enc = nn.Sequential(
            conv_block(in_channels, c),
            conv_block(c, c * 2),
            conv_block(c * 2, c * 4),
        )

        # Temporal encoder
        self.temp_enc = ConvLSTMCell(c * 4, c * 4)

        # Temporal decoder (reverse)
        self.temp_dec = ConvLSTMCell(c * 4, c * 4)

        # Spatial decoder (shared per frame)
        self.spatial_dec = nn.Sequential(
            deconv_block(c * 4, c * 2),
            deconv_block(c * 2, c),
            deconv_block(c, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: [B, T, C, H, W]
        Returns:
            recon: [B, T, C, H, W]
        """
        B, T, C, H, W = clip.shape
        # Spatial encode all frames
        frames = [self.spatial_enc(clip[:, t]) for t in range(T)]   # T × [B, C4, h, w]
        _, _, h, w = frames[0].shape

        # Temporal encode
        h_enc, c_enc = self.temp_enc.init_state(B, h, w, clip.device)
        enc_states = []
        for frame in frames:
            h_enc, c_enc = self.temp_enc(frame, h_enc, c_enc)
            enc_states.append(h_enc)

        # Temporal decode (start from last encoded state)
        h_dec, c_dec = h_enc, c_enc
        decoded = []
        for t in range(T - 1, -1, -1):
            h_dec, c_dec = self.temp_dec(enc_states[t], h_dec, c_dec)
            decoded.append(h_dec)
        decoded.reverse()

        # Spatial decode
        recon = torch.stack([self.spatial_dec(d) for d in decoded], dim=1)  # [B, T, C, H, W]
        return recon

    def reconstruction_error(self, clip: torch.Tensor) -> torch.Tensor:
        """Per-clip mean squared reconstruction error."""
        recon = self.forward(clip)
        return F.mse_loss(recon, clip, reduction='none').mean([1, 2, 3, 4])
