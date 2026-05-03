"""
GCN-GRU: Graph Convolutional Network + Gated Recurrent Unit
for spatiotemporal traffic / crowd flow forecasting.

Architecture:
  - Graph Convolution (spectral, ChebNet k=3) to aggregate spatial context
  - GRU layers for temporal modelling
  - Linear output heads per node

Reference: Kipf & Welling (ICLR 2017) + Cho et al. (GRU)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalise_adj(adj: np.ndarray) -> torch.Tensor:
    """
    Compute symmetric-normalised adjacency for spectral graph convolution.
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    adj = adj + np.eye(adj.shape[0])
    d = np.sqrt(adj.sum(axis=1))
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    adj_norm = adj * d_inv[:, None] * d_inv[None, :]
    return torch.FloatTensor(adj_norm)


# ---------------------------------------------------------------------------
# Graph Convolution
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Single-hop spectral graph convolution."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, N, F]
            adj: [N, N] normalised adjacency (on same device)
        Returns:
            out: [B, N, out_features]
        """
        out = torch.matmul(x, self.W)          # [B, N, out_features]
        out = torch.einsum('nm,bmf->bnf', adj, out)  # graph aggregation
        if self.b is not None:
            out = out + self.b
        return out


class ChebGraphConv(nn.Module):
    """
    Chebyshev polynomial graph convolution (K-hop).
    Provides multi-hop neighbourhood aggregation without an eigendecomposition.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 3):
        super().__init__()
        self.K = K
        self.thetas = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, out_features))
            for _ in range(K)
        ])
        for t in self.thetas:
            nn.init.xavier_uniform_(t)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, N, F]
            L_tilde: [N, N] scaled Laplacian (eigenvalues in [-1, 1])
        """
        T_0 = x
        T_1 = torch.einsum('nm,bmf->bnf', L_tilde, x)
        out = T_0 @ self.thetas[0] + T_1 @ self.thetas[1]
        T_prev, T_curr = T_1, None
        for k in range(2, self.K):
            T_curr = 2.0 * torch.einsum('nm,bmf->bnf', L_tilde, T_prev) - T_0
            out = out + T_curr @ self.thetas[k]
            T_0, T_prev = T_prev, T_curr
        return out


# ---------------------------------------------------------------------------
# GCN-GRU Cell
# ---------------------------------------------------------------------------

class GCGRUCell(nn.Module):
    """
    GRU cell where each matrix multiplication is replaced by a graph convolution.
    """

    def __init__(self, input_dim: int, hidden_dim: int, adj_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Reset gate, update gate, candidate — each needs x and h
        self.gc_rz_x = GraphConv(input_dim,  2 * hidden_dim)
        self.gc_rz_h = GraphConv(hidden_dim, 2 * hidden_dim, bias=False)
        self.gc_c_x  = GraphConv(input_dim,  hidden_dim)
        self.gc_c_h  = GraphConv(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, N, F_in]
            h:   [B, N, hidden_dim]
            adj: [N, N]
        Returns:
            h_new: [B, N, hidden_dim]
        """
        rz = torch.sigmoid(self.gc_rz_x(x, adj) + self.gc_rz_h(h, adj))
        r, z = rz.chunk(2, dim=-1)
        c = torch.tanh(self.gc_c_x(x, adj) + self.gc_c_h(r * h, adj))
        return z * h + (1 - z) * c


# ---------------------------------------------------------------------------
# Full GCN-GRU model
# ---------------------------------------------------------------------------

class GCNGRU(nn.Module):
    """
    Stacked GCN-GRU model for multi-step graph sequence forecasting.

    Args:
        num_nodes:   N (e.g. 207 for METR-LA)
        in_features: F per node per timestep (e.g. 2 = speed + time_of_day)
        hidden_dim:  GRU hidden state size
        num_layers:  number of stacked GCN-GRU layers
        seq_out:     number of future steps to predict
        out_features: output features per node per step (usually 1 = speed)
    """

    def __init__(self, num_nodes: int, in_features: int = 2,
                 hidden_dim: int = 64, num_layers: int = 2,
                 seq_out: int = 12, out_features: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_out = seq_out

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_d = in_features if i == 0 else hidden_dim
            self.cells.append(GCGRUCell(in_d, hidden_dim, num_nodes))

        # Output: seq_out x out_features per node
        self.out_fc = nn.Linear(hidden_dim, seq_out * out_features)
        self.out_features = out_features

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, T_in, N, F]
            adj: [N, N] normalised adjacency matrix
        Returns:
            y:   [B, T_out, N, out_features]
        """
        B, T, N, _ = x.shape

        # Initialise hidden states
        h = [torch.zeros(B, N, self.hidden_dim, device=x.device)
             for _ in range(self.num_layers)]

        # Encode input sequence
        for t in range(T):
            inp = x[:, t]   # [B, N, F]
            for l, cell in enumerate(self.cells):
                h[l] = cell(inp, h[l], adj)
                inp = h[l]

        # Decode: use last hidden state → predict all future steps at once
        out = self.out_fc(h[-1])      # [B, N, T_out * out_features]
        out = out.view(B, N, self.seq_out, self.out_features)
        out = out.permute(0, 2, 1, 3)  # [B, T_out, N, out_features]
        return out
