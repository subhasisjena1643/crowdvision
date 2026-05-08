"""
Convolutional Autoencoder for video anomaly detection.

Trained only on normal frames, high reconstruction error = anomaly.

Architectures:
  1. ConvAE         – encoder-decoder with MemAE bottleneck (frame-level)
  2. ConvLSTMAE     – spatiotemporal LSTM autoencoder (clip-level)
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
# Memory Module (Gong et al., ICCV 2019)
# ---------------------------------------------------------------------------

class MemoryModule(nn.Module):
    """
    Learnable memory bank of normal-pattern prototypes.

    Given a query tensor from the encoder, this module computes attention
    weights over N stored prototypes and returns a weighted combination.
    The decoder can therefore only reconstruct from stored normal patterns.

    A sparsity loss on the attention weights (entropy-based) encourages
    each query to match only a few prototypes, sharpening the separation
    between normal and anomalous inputs.

    Args:
        num_slots:   number of memory prototypes (N)
        slot_dim:    dimensionality of each prototype (must match encoder
                     output channels)
        shrink_thres: hard-shrinkage threshold for attention weights;
                      entries below this are zeroed to enforce sparsity
    """

    def __init__(self, num_slots: int = 500, slot_dim: int = 64,
                 shrink_thres: float = 0.005):
        super().__init__()
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres

        # Memory bank: N prototypes of dimension D
        self.memory = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.05)

        # Stored after each forward for the sparsity loss
        self.last_attn: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, D, H, W] — encoder bottleneck features
        Returns:
            z_hat: [B, D, H, W] — memory-addressed features
        """
        B, D, H, W = z.shape
        input_dtype = z.dtype

        # Force float32 — cosine similarity + normalize produces NaN in fp16
        with torch.amp.autocast('cuda', enabled=False):
            z = z.float()

            # Flatten spatial dims: [B, D, H*W] -> [B*H*W, D]
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)        # [BHW, D]

            # Cosine similarity attention: [BHW, N]
            z_norm = F.normalize(z_flat, dim=1)
            m_norm = F.normalize(self.memory, dim=1)
            attn = torch.mm(z_norm, m_norm.t())                   # [BHW, N]

            # Hard shrinkage: zero out small weights for sparsity
            if self.shrink_thres > 0:
                attn = F.relu(attn - self.shrink_thres) * \
                       (attn > self.shrink_thres).float()
                # Re-normalise so rows sum to 1 (all entries >= 0 after relu)
                attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
            else:
                attn = F.softmax(attn, dim=1)

            # Store for sparsity loss
            self.last_attn = attn.detach()
            self._attn_for_loss = attn

            # Retrieve: weighted combination of memory slots
            z_hat = torch.mm(attn, self.memory)                   # [BHW, D]
            z_hat = z_hat.reshape(B, H, W, D).permute(0, 3, 1, 2) # [B, D, H, W]

        return z_hat.to(input_dtype)

    def sparsity_loss(self) -> torch.Tensor:
        """
        Entropy-based sparsity loss on the most recent attention weights.
        Encourages peaky (sparse) attention distributions.
        """
        if self._attn_for_loss is None:
            return torch.tensor(0.0)
        w = self._attn_for_loss
        entropy = -(w * torch.log(w + 1e-12)).sum(dim=1).mean()
        return entropy


# ---------------------------------------------------------------------------
# 1. Frame-level convolutional autoencoder with Memory Module
# ---------------------------------------------------------------------------

class ConvAE(nn.Module):
    """
    Convolutional autoencoder for single-frame anomaly detection.

    Architecture: Deep encoder → 1×1 squeeze → MemAE → 1×1 expand → decoder.
    NO skip connections — the memory module must be the only path for
    information, forcing anomalous inputs to be poorly reconstructed.

    Trained only on normal frames.  Anomalous inputs incur higher
    reconstruction error because their features can't be well-represented
    by the normal-pattern memory bank.

    Args:
        in_channels: 1 for grayscale, 3 for RGB
        base_ch:     base number of channels (default 32)
        mem_slots:   number of memory prototype slots
        shrink_thres: hard-shrinkage threshold for memory attention
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32,
                 mem_slots: int = 50, shrink_thres: float = 0.05):
        super().__init__()
        c = base_ch

        # Encoder: 128×192 → 8×12  (4 max-pools)
        self.encoder = nn.Sequential(
            conv_block(in_channels, c),        # → 64×96
            conv_block(c, c * 2),              # → 32×48
            conv_block(c * 2, c * 4),          # → 16×24
            conv_block(c * 4, c * 8),          # → 8×12
        )

        # Bottleneck – aggressive squeeze for tight information bottleneck
        # c*8 → 16 channels at 8×12 spatial = 1,536 total values per frame
        self.bottleneck_dim = 16
        self.bottleneck_enc = nn.Sequential(
            nn.Conv2d(c * 8, self.bottleneck_dim, 1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
        )

        # Memory module: routes latent through N normal-pattern prototypes
        self.memory = MemoryModule(
            num_slots=mem_slots, slot_dim=self.bottleneck_dim,
            shrink_thres=shrink_thres,
        )

        self.bottleneck_dec = nn.Sequential(
            nn.Conv2d(self.bottleneck_dim, c * 8, 1, bias=False),   # expand
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        self.code_ch = self.bottleneck_dim  # effective bottleneck channels

        # Decoder: 8×12 → 128×192  (4 up-samples)
        self.decoder = nn.Sequential(
            deconv_block(c * 8, c * 4),        # → 16×24
            deconv_block(c * 4, c * 2),        # → 32×48
            deconv_block(c * 2, c),            # → 64×96
            deconv_block(c, c),                # → 128×192
            nn.Conv2d(c, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed frame, same shape as input."""
        z = self.encoder(x)
        z = self.bottleneck_enc(z)
        z = self.memory(z)          # route through normal-pattern memory
        z = self.bottleneck_dec(z)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hybrid per-sample anomaly score: weighted combination of
        global mean MSE and peak (max-patch) MSE.

        The peak component amplifies spatially localised anomalies that
        would otherwise be washed out by a global average.
        """
        recon = self.forward(x)
        err = (recon - x).pow(2)                          # B×C×H×W

        mean_err = err.mean(dim=[1, 2, 3])                # global mean

        # Max-patch: pool error into 8×8 patches, take per-sample max
        patch_err = F.adaptive_avg_pool2d(err, (8, 8))    # B×C×8×8
        max_patch = patch_err.amax(dim=[1, 2, 3])         # per-sample max

        return 0.6 * mean_err + 0.4 * max_patch

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

        # Memory module at the temporal bottleneck
        self.memory = MemoryModule(
            num_slots=1000, slot_dim=c * 4,
            shrink_thres=0.005,
        )

        # Temporal decoder (reverse)
        self.temp_dec = ConvLSTMCell(c * 4, c * 4)

        # Spatial decoder (shared per frame)
        self.spatial_dec = nn.Sequential(
            deconv_block(c * 4, c * 2),
            deconv_block(c * 2, c),
            deconv_block(c, in_channels),
            nn.Tanh(),
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

        # Route the final bottleneck spatial-temporal state through memory
        h_enc = self.memory(h_enc)

        # Temporal decode (start from last encoded state)
        h_dec, c_dec = h_enc, c_enc
        decoder_input = h_enc  # Initial input to the decoder
        decoded = []
        for t in range(T - 1, -1, -1):
            h_dec, c_dec = self.temp_dec(decoder_input, h_dec, c_dec)
            decoder_input = h_dec  # Feed previous output as next input
            decoded.append(h_dec)
        decoded.reverse()

        # Spatial decode
        recon = torch.stack([self.spatial_dec(d) for d in decoded], dim=1)  # [B, T, C, H, W]
        return recon

    def reconstruction_error(self, clip: torch.Tensor) -> torch.Tensor:
        """Per-frame mean squared reconstruction error."""
        recon = self.forward(clip)
        # Average over C, H, W to get error per frame -> returns shape [B, T]
        return F.mse_loss(recon, clip, reduction='none').mean(dim=[2, 3, 4])
