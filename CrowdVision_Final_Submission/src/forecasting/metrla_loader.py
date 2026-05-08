"""
METR-LA dataset loader for spatiotemporal forecasting.
Real traffic/crowd flow data from 207 sensors over 4 months.
"""

import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset


class METRLADataset(Dataset):
    """
    Real METR-LA traffic flow dataset for GCN-GRU training.
    Replaces synthetic data with actual sensor measurements.
    """
    
    def __init__(
        self,
        data_path: str,
        adj_mx_path: str,
        seq_len: int = 20,
        forecast_horizon: int = 40,
        normalize: bool = True,
        train: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1
    ):
        """
        Args:
            data_path: Path to metr-la.h5 file
            adj_mx_path: Path to adj_mx.pkl (adjacency matrix)
            seq_len: Input sequence length (default 20 = 10 min @ 30s intervals)
            forecast_horizon: Output sequence length (default 40 = 20 min)
            normalize: Z-score normalization
            train: Train/val/test split
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
        """
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        # Load HDF5 data
        with h5py.File(data_path, 'r') as f:
            # Data shape: (num_timesteps, num_sensors)
            self.data = f['df']['block0_values'][:]
            
        # Load adjacency matrix
        with open(adj_mx_path, 'rb') as f:
            _, _, self.adj_mx = pickle.load(f, encoding='latin1')
            
        num_sensors = self.data.shape[1]
        num_timesteps = self.data.shape[0]
        
        # Normalize
        if normalize:
            self.mean = np.mean(self.data)
            self.std = np.std(self.data)
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0
            
        # Train/val/test split
        train_size = int(num_timesteps * train_ratio)
        val_size = int(num_timesteps * val_ratio)
        
        if train:
            self.data = self.data[:train_size]
        elif train_ratio + val_ratio < 1.0:  # validation
            self.data = self.data[train_size:train_size + val_size]
        else:  # test
            self.data = self.data[train_size + val_size:]
            
        # Generate sliding windows
        self.indices = []
        total_len = self.data.shape[0]
        window_size = seq_len + forecast_horizon
        
        for i in range(0, total_len - window_size + 1):
            self.indices.append(i)
            
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input sequence (seq_len, num_sensors)
            y: Target sequence (forecast_horizon, num_sensors)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        target_end = end_idx + self.forecast_horizon
        
        x = torch.FloatTensor(self.data[start_idx:end_idx])
        y = torch.FloatTensor(self.data[end_idx:target_end])
        
        return x, y
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Return graph adjacency matrix"""
        return torch.FloatTensor(self.adj_mx)


def load_metrla_data(
    data_dir: str,
    seq_len: int = 20,
    forecast_horizon: int = 40,
    batch_size: int = 16
) -> Tuple[Dataset, Dataset, Dataset, torch.Tensor]:
    """
    Load METR-LA train/val/test datasets.
    
    Args:
        data_dir: Path to directory containing metr-la.h5 and adj_mx.pkl
        seq_len: Input sequence length
        forecast_horizon: Output prediction length
        batch_size: Batch size (for reference; DataLoaders created externally)
    
    Returns:
        train_dataset, val_dataset, test_dataset, adj_matrix
    """
    data_path = Path(data_dir) / "metr-la.h5"
    adj_path = Path(data_dir) / "adj_mx.pkl"
    
    train_dataset = METRLADataset(
        str(data_path), str(adj_path),
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        train=True
    )
    
    val_dataset = METRLADataset(
        str(data_path), str(adj_path),
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        train=False  # Will use validation split
    )
    
    test_dataset = METRLADataset(
        str(data_path), str(adj_path),
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        train=False
    )
    
    adj_matrix = train_dataset.get_adjacency_matrix()
    
    return train_dataset, val_dataset, test_dataset, adj_matrix
