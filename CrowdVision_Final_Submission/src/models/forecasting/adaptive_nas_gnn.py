"""
AdaptiveNAS-GNN: Differentiable Architecture Search for Spatiotemporal GNNs.

Our novel contribution: use DARTS-style soft architecture search to
automatically discover optimal graph neural network operations for crowd
flow and traffic forecasting.

Operations searched per layer:
  - Graph aggregation: sum | mean | max | attention
  - Dilation (temporal): 1 | 2 | 4 | 8 timesteps
  - Non-linearity: relu | elu | tanh | identity

After search, the architecture is discretised (argmax) and retrained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Candidate operations
# ---------------------------------------------------------------------------

class GraphSum(nn.Module):
    def __init__(self, d: int): super().__init__(); self.W = nn.Linear(d, d, bias=False)
    def forward(self, x, adj): return torch.einsum('nm,bmf->bnf', adj, self.W(x))

class GraphMean(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
    def forward(self, x, adj):
        agg_sum = torch.einsum('nm,bmf->bnf', adj, self.W(x))
        count = adj.sum(dim=1, keepdim=True).unsqueeze(0) + 1e-6
        return agg_sum / count

class GraphMax(nn.Module):
    def __init__(self, d: int): super().__init__(); self.W = nn.Linear(d, d, bias=False)
    def forward(self, x, adj):
        out = self.W(x).unsqueeze(1).expand(-1, x.size(1), -1, -1)  # B,N,N,F
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        out = out.masked_fill(mask, float('-inf'))
        return out.max(dim=2).values

class GraphAttention(nn.Module):
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.d_h = d // heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

    def forward(self, x, adj):
        B, N, D = x.shape
        q = self.Wq(x).view(B, N, self.heads, self.d_h).permute(0,2,1,3)
        k = self.Wk(x).view(B, N, self.heads, self.d_h).permute(0,2,1,3)
        v = self.Wv(x).view(B, N, self.heads, self.d_h).permute(0,2,1,3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)
        # Mask disconnected nodes
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0,2,1,3).contiguous().view(B, N, D)
        return out


OPS_GRAPH = {'sum': GraphSum, 'mean': GraphMean, 'max': GraphMax, 'attn': GraphAttention}
OPS_ACT   = {
    'relu':     lambda: nn.ReLU(inplace=False),
    'elu':      lambda: nn.ELU(inplace=False),
    'tanh':     lambda: nn.Tanh(),
    'identity': lambda: nn.Identity(),
}
DILATIONS = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Mixed operation (DARTS soft-selection)
# ---------------------------------------------------------------------------

class MixedGraphOp(nn.Module):
    """Weighted mixture of graph aggregation operations (DARTS cell)."""

    def __init__(self, d: int):
        super().__init__()
        self.ops = nn.ModuleDict({k: cls(d) for k, cls in OPS_GRAPH.items()})
        self.arch_params = nn.Parameter(torch.zeros(len(OPS_GRAPH)))

    def forward(self, x, adj):
        weights = F.softmax(self.arch_params, dim=0)
        return sum(w * op(x, adj) for w, (_, op) in zip(weights, self.ops.items()))

    def discretise(self) -> str:
        """Return name of chosen operation (for final architecture)."""
        return list(OPS_GRAPH.keys())[self.arch_params.argmax().item()]


# ---------------------------------------------------------------------------
# NAS GNN block
# ---------------------------------------------------------------------------

class NASGNNBlock(nn.Module):
    """One searchable spatiotemporal block: graph op + temporal conv + act."""

    def __init__(self, d: int, seq_len: int):
        super().__init__()
        self.graph_op  = MixedGraphOp(d)
        # Temporal conv with learnable dilation
        self.t_convs   = nn.ModuleList([
            nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
            for dil in DILATIONS
        ])
        self.t_arch    = nn.Parameter(torch.zeros(len(DILATIONS)))
        self.act_mods  = nn.ModuleList([m() for m in OPS_ACT.values()])
        self.act_arch  = nn.Parameter(torch.zeros(len(OPS_ACT)))
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, T, N, F]
            adj: [N, N]
        Returns: [B, T, N, F]
        """
        B, T, N, D = x.shape

        # Graph aggregation at each timestep
        g_out = torch.stack([self.graph_op(x[:, t], adj) for t in range(T)], dim=1)

        # Temporal conv (operate on node-wise time series)
        t_inp = g_out.permute(0, 2, 3, 1).reshape(B * N, D, T)  # [BN, D, T]
        t_w   = F.softmax(self.t_arch, dim=0)
        t_out = sum(w * conv(t_inp) for w, conv in zip(t_w, self.t_convs))
        t_out = t_out.reshape(B, N, D, T).permute(0, 3, 1, 2)   # [B, T, N, D]

        # Activation
        a_w  = F.softmax(self.act_arch, dim=0)
        t_out = sum(w * act(t_out) for w, act in zip(a_w, self.act_mods))

        # Residual + LayerNorm
        out = self.layer_norm(t_out + x)
        return out


# ---------------------------------------------------------------------------
# Full AdaptiveNAS model
# ---------------------------------------------------------------------------

class AdaptiveNASGNN(nn.Module):
    """
    Differentiable NAS GNN for spatiotemporal forecasting.

    Training phase: optimise with bilevel optimisation (model params + arch params).
    Eval phase:     call discretise() to fix architecture, then retrain.

    Args:
        num_nodes:    N
        in_features:  input features per node per step
        hidden_dim:   internal feature dimension
        num_blocks:   number of searchable NAS blocks
        seq_in:       input sequence length
        seq_out:      output sequence length
        out_features: output features per node per step
    """

    def __init__(self, num_nodes: int, in_features: int = 2,
                 hidden_dim: int = 64, num_blocks: int = 3,
                 seq_in: int = 12, seq_out: int = 12,
                 out_features: int = 1):
        super().__init__()
        self.seq_in    = seq_in
        self.seq_out   = seq_out
        self.num_nodes = num_nodes

        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.blocks     = nn.ModuleList([
            NASGNNBlock(hidden_dim, seq_in) for _ in range(num_blocks)
        ])
        self.out_fc = nn.Linear(hidden_dim * seq_in, seq_out * out_features)
        self.out_features = out_features

    def arch_parameters(self):
        """Yield ONLY architecture parameters (for bilevel optimisation)."""
        for block in self.blocks:
            yield block.graph_op.arch_params
            yield block.t_arch
            yield block.act_arch

    def model_parameters(self):
        """Yield all NON-architecture parameters."""
        arch_ids = {id(p) for p in self.arch_parameters()}
        for p in self.parameters():
            if id(p) not in arch_ids:
                yield p

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, T_in, N, F]
            adj: [N, N]
        Returns:
            y:   [B, T_out, N, out_features]
        """
        B, T, N, D = x.shape
        x = self.input_proj(x)            # [B, T, N, hidden_dim]

        for block in self.blocks:
            x = block(x, adj)

        # Flatten temporal + feature dims for output projection
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, T*hidden]
        out = self.out_fc(x_flat)         # [B, N, T_out * out_features]
        out = out.view(B, N, self.seq_out, self.out_features)
        return out.permute(0, 2, 1, 3)   # [B, T_out, N, out_features]

    def get_architecture(self) -> dict:
        """Return the discretised architecture choices."""
        arch = {}
        for i, block in enumerate(self.blocks):
            arch[f'block_{i}'] = {
                'graph_op': block.graph_op.discretise(),
                'dilation': DILATIONS[block.t_arch.argmax().item()],
                'activation': list(OPS_ACT.keys())[block.act_arch.argmax().item()],
            }
        return arch

    def discretize(self):
        """Freeze architecture parameters to halt the search phase.
        After this, the model uses the learned architecture but continues
        refining weights with standard training.
        """
        for block in self.blocks:
            block.graph_op.arch_params.requires_grad_(False)
            block.t_arch.requires_grad_(False)
            block.act_arch.requires_grad_(False)
