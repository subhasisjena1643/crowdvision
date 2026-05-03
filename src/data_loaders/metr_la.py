"""
METR-LA and PEMS dataset loaders for spatiotemporal traffic forecasting.

Available data:
  metr-la/Datasets/
    metr-la.h5      - 207 sensors, speed data (mph), 5-min intervals
    adj_mx.pkl      - adjacency matrix (distance-based, sensor IDs, locator)
    PEMSd3.csv/npz  - 358 sensors
    pemsd4.csv/npz  - 307 sensors
    PEMSd7.csv/npz  - 883 sensors

References:
  DCRNN (Li et al., ICLR 2018) – standard METR-LA preprocessing
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def load_adj_matrix(pkl_path: str):
    """
    Load the METR-LA adjacency matrix pickle.

    Returns:
        adj: np.ndarray shape (N, N) float32  (normalised)
        sensor_ids: list of sensor id strings
        id_to_idx:  dict {sensor_id -> row/col index}
    """
    with open(pkl_path, 'rb') as f:
        sensor_ids, id_to_idx, adj = pickle.load(f, encoding='latin1')
    return np.array(adj, dtype=np.float32), sensor_ids, id_to_idx


def build_adj_from_csv(csv_path: str, num_nodes: int,
                        normalised: bool = True) -> np.ndarray:
    """
    Build an adjacency matrix from a PEMS-style distance CSV.

    CSV columns expected: from_node, to_node, distance
    """
    df = pd.read_csv(csv_path, header=0)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    std = df.iloc[:, 2].std()
    for _, row in df.iterrows():
        i, j, d = int(row.iloc[0]), int(row.iloc[1]), float(row.iloc[2])
        if i < num_nodes and j < num_nodes:
            w = np.exp(-(d ** 2) / (std ** 2))
            adj[i, j] = w
            adj[j, i] = w
    if normalised:
        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        adj += np.eye(num_nodes)
        d = np.sqrt(adj.sum(axis=1))
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        adj = adj * d_inv[:, None] * d_inv[None, :]
    return adj


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class StandardScaler:
    """Z-score normalisation, broadcast-safe."""

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data: np.ndarray):
        self.mean = data.mean()
        self.std = data.std() + 1e-8
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ---------------------------------------------------------------------------
# METR-LA
# ---------------------------------------------------------------------------

class METRLADataset(Dataset):
    """
    METR-LA sliding-window dataset for graph spatiotemporal forecasting.

    Each sample: (input_seq, target_seq)
      input_seq:  [seq_in, N, F]  – past seq_in time steps, N sensors, F features
      target_seq: [seq_out, N, 1] – future seq_out time steps, speed only
    """

    def __init__(self, data: np.ndarray, seq_in: int = 12, seq_out: int = 12):
        """
        Args:
            data:    np.ndarray [T, N, F]  (already normalised)
            seq_in:  input sequence length  (1 step = 5 min → 12 = 1 hour)
            seq_out: output sequence length
        """
        self.data = data
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.n_samples = len(data) - seq_in - seq_out + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_in]           # [T_in, N, F]
        y = self.data[idx + self.seq_in: idx + self.seq_in + self.seq_out, :, :1]  # speed only
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_metr_la(h5_path: str,
                 seq_in: int = 12,
                 seq_out: int = 12,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 adj_path: Optional[str] = None):
    """
    Load METR-LA, normalise, split, and return DataLoaders + metadata.

    Args:
        adj_path: optional path to adjacency matrix pickle file.
                  If not provided, constructs path as {h5_path_parent}/adj_mx.pkl

    Returns:
        train_loader, val_loader, test_loader, scaler, adj, num_nodes
    """
    with h5py.File(h5_path, 'r') as f:
        df = pd.DataFrame(f['df']['block0_values'][:],
                          columns=f['df']['block0_items'][:].astype(str))
    data = df.values.astype(np.float32)  # [T, N]

    # Add time-of-day feature (normalised 0..1)
    T, N = data.shape
    tod = np.tile(np.linspace(0, 1, 288, endpoint=False).repeat(
        int(np.ceil(T / 288)))[:T, np.newaxis], (1, N))  # [T, N]
    data_feat = np.stack([data, tod], axis=-1)  # [T, N, 2]

    # Normalise speed channel only
    scaler = StandardScaler()
    scaler.fit(data_feat[:int(T * train_ratio), :, 0])
    data_feat[:, :, 0] = scaler.transform(data_feat[:, :, 0])

    # Split
    t_train = int(T * train_ratio)
    t_val   = t_train + int(T * val_ratio)

    sets = {
        'train': METRLADataset(data_feat[:t_train], seq_in, seq_out),
        'val':   METRLADataset(data_feat[t_train:t_val], seq_in, seq_out),
        'test':  METRLADataset(data_feat[t_val:], seq_in, seq_out),
    }

    loaders = {split: DataLoader(ds, batch_size=batch_size if split == 'train' else 64,
                                  shuffle=(split == 'train'),
                                  num_workers=num_workers, pin_memory=True)
               for split, ds in sets.items()}

    # Load adjacency
    if adj_path is None:
        adj_path = str(Path(h5_path).parent / 'adj_mx.pkl')
    adj, sensor_ids, _ = load_adj_matrix(adj_path)
    num_nodes = N

    return (loaders['train'], loaders['val'], loaders['test'],
            scaler, adj, num_nodes)


# ---------------------------------------------------------------------------
# PEMS datasets (d3, d4, d7)
# ---------------------------------------------------------------------------

class PEMSDataset(Dataset):
    """General PEMS dataset from .npz files."""

    def __init__(self, data: np.ndarray, seq_in: int = 12, seq_out: int = 12):
        self.data = data
        self.seq_in = seq_in
        self.seq_out = seq_out

    def __len__(self):
        return len(self.data) - self.seq_in - self.seq_out + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_in]
        y = self.data[idx + self.seq_in: idx + self.seq_in + self.seq_out, :, :1]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_pems(npz_path: str, csv_path: str,
              seq_in: int = 12, seq_out: int = 12,
              train_ratio: float = 0.7, val_ratio: float = 0.1,
              batch_size: int = 64, num_workers: int = 4):
    """
    Load a PEMS dataset from .npz.

    Returns:
        train_loader, val_loader, test_loader, scaler, adj, num_nodes
    """
    raw = np.load(npz_path)
    # Try common key names
    data = raw.get('data', raw.get('x', list(raw.values())[0]))
    if data.ndim == 2:
        data = data[:, :, np.newaxis]               # [T, N] → [T, N, 1]

    data = data.astype(np.float32)
    T, N, F = data.shape

    scaler = StandardScaler()
    t_train = int(T * train_ratio)
    scaler.fit(data[:t_train, :, 0])
    data[:, :, 0] = scaler.transform(data[:, :, 0])

    t_val = t_train + int(T * val_ratio)
    sets = {
        'train': PEMSDataset(data[:t_train], seq_in, seq_out),
        'val':   PEMSDataset(data[t_train:t_val], seq_in, seq_out),
        'test':  PEMSDataset(data[t_val:], seq_in, seq_out),
    }
    loaders = {s: DataLoader(ds, batch_size=batch_size if s == 'train' else 64,
                              shuffle=(s == 'train'),
                              num_workers=num_workers, pin_memory=True)
               for s, ds in sets.items()}

    # Build adjacency
    adj = build_adj_from_csv(csv_path, N)

    return loaders['train'], loaders['val'], loaders['test'], scaler, adj, N
