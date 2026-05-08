#!/usr/bin/env python3
"""
CrowdVision — Unified Inference Pipeline

Production-ready inference that combines:
  1. Density estimation (AdaptiveCSRNet)  → crowd count + heatmap
  2. Traffic forecasting (GCN-GRU)        → future traffic predictions
  3. Anomaly detection (ConvAE + MemAE)   → anomaly score (supplementary)

Usage:
    from src.inference.pipeline import CrowdVisionPipeline
    
    pipe = CrowdVisionPipeline()
    result = pipe.analyze_frame(image_tensor)
"""

import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


class CrowdVisionPipeline:
    """
    End-to-end inference pipeline for crowd analysis.
    
    Loads trained models from checkpoints and provides a unified API
    for all three tasks: density estimation, forecasting, and anomaly detection.
    """
    
    def __init__(self, 
                 checkpoint_dir: str = 'checkpoints',
                 device: str = 'auto',
                 enable_density: bool = True,
                 enable_anomaly: bool = True,
                 enable_forecasting: bool = True):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.models = {}
        self.transforms_dict = {}
        
        if enable_density:
            self._load_density()
        if enable_anomaly:
            self._load_anomaly()
        if enable_forecasting:
            self._load_forecasting()
            
        print(f'CrowdVisionPipeline initialized on {device}')
        print(f'  Active models: {list(self.models.keys())}')
    
    def _load_density(self):
        """Load AdaptiveCSRNet for crowd density estimation."""
        from src.models.density.adaptive_csrnet import AdaptiveCSRNet
        
        ckpt_path = self.checkpoint_dir / 'adaptive_csrnet_shaA' / 'best.pt'
        if not ckpt_path.exists():
            print('  [SKIP] Density: no checkpoint')
            return
            
        model = AdaptiveCSRNet(load_weights=False).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt.get('model', ckpt))
        model.eval()
        
        self.models['density'] = model
        self.transforms_dict['density'] = transforms.Compose([
            transforms.Resize((576, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print('  ✓ Density (AdaptiveCSRNet) loaded')
    
    def _load_anomaly(self):
        """Load ConvAE + MemAE for anomaly detection."""
        from src.models.anomaly.conv_ae import ConvAE
        
        ckpt_path = self.checkpoint_dir / 'convae_ped2' / 'best.pt'
        if not ckpt_path.exists():
            print('  [SKIP] Anomaly: no checkpoint')
            return
            
        model = ConvAE(in_channels=1, base_ch=64, mem_slots=500, shrink_thres=0.05)
        model = model.to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        self.models['anomaly'] = model
        self.transforms_dict['anomaly'] = transforms.Compose([
            transforms.Resize((128, 192)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        print('  ✓ Anomaly (ConvAE + MemAE) loaded')
    
    def _load_forecasting(self):
        """Load GCN-GRU for traffic forecasting."""
        from src.models.forecasting.gcn_gru import GCNGRU, normalise_adj
        from src.data_loaders.metr_la import load_adj_matrix
        
        ckpt_path = self.checkpoint_dir / 'gcn_gru_metrla' / 'best.pt'
        adj_path = Path('data/metr-la/Datasets/adj_mx.pkl')
        
        if not ckpt_path.exists():
            print('  [SKIP] Forecasting: no checkpoint')
            return
        if not adj_path.exists():
            print('  [SKIP] Forecasting: no adjacency matrix')
            return
        
        adj, sensor_ids, _ = load_adj_matrix(str(adj_path))
        num_nodes = adj.shape[0]
        adj_norm = normalise_adj(adj).to(self.device)
        
        model = GCNGRU(num_nodes=num_nodes, in_features=2, hidden_dim=64,
                       num_layers=2, seq_out=12).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        self.models['forecasting'] = model
        self._adj_norm = adj_norm
        self._num_nodes = num_nodes
        self._sensor_ids = sensor_ids
        print(f'  ✓ Forecasting (GCN-GRU) loaded ({num_nodes} sensors)')
    
    @torch.no_grad()
    def estimate_density(self, image) -> Dict:
        """
        Estimate crowd density from an image.
        
        Args:
            image: PIL Image or torch.Tensor [3, H, W]
            
        Returns:
            dict with 'density_map', 'count', 'latency_ms'
        """
        if 'density' not in self.models:
            return {'error': 'Density model not loaded'}
        
        t0 = time.time()
        if not isinstance(image, torch.Tensor):
            image = self.transforms_dict['density'](image)
        x = image.unsqueeze(0).to(self.device)
        density_map = self.models['density'](x)
        count = density_map.sum().item()
        latency = (time.time() - t0) * 1000
        
        return {
            'density_map': density_map.squeeze().cpu().numpy(),
            'count': count,
            'latency_ms': latency,
        }
    
    @torch.no_grad()
    def detect_anomaly(self, image) -> Dict:
        """
        Compute anomaly score for a single frame.
        
        Args:
            image: PIL Image or torch.Tensor [1, H, W] (grayscale)
            
        Returns:
            dict with 'anomaly_score', 'normalized_score', 'is_anomaly', 'latency_ms'
        """
        if 'anomaly' not in self.models:
            return {'error': 'Anomaly model not loaded'}
        
        t0 = time.time()
        if not isinstance(image, torch.Tensor):
            image = self.transforms_dict['anomaly'](image)
        x = image.unsqueeze(0).to(self.device)
        score = self.models['anomaly'].reconstruction_error(x).item()
        normalized = 1.0 / (1.0 + np.exp(-(score - 0.02) * 100))
        latency = (time.time() - t0) * 1000
        
        return {
            'anomaly_score': score,
            'normalized_score': normalized,
            'is_anomaly': normalized > 0.5,
            'latency_ms': latency,
        }
    
    @torch.no_grad()
    def forecast_traffic(self, historical_data: torch.Tensor) -> Dict:
        """
        Forecast future traffic from historical sensor data.
        
        Args:
            historical_data: [seq_in, N, F] tensor (12 timesteps, 207 sensors, 2 features)
            
        Returns:
            dict with 'predictions', 'num_sensors', 'horizons', 'latency_ms'
        """
        if 'forecasting' not in self.models:
            return {'error': 'Forecasting model not loaded'}
        
        t0 = time.time()
        x = historical_data.unsqueeze(0).to(self.device)
        pred = self.models['forecasting'](x, self._adj_norm)
        latency = (time.time() - t0) * 1000
        
        return {
            'predictions': pred.squeeze(0).cpu().numpy(),
            'num_sensors': self._num_nodes,
            'horizons': {'15min': 2, '30min': 5, '60min': 11},
            'latency_ms': latency,
        }
    
    @torch.no_grad()
    def analyze_frame(self, image) -> Dict:
        """Full analysis of a single frame: density + anomaly."""
        result = {'timestamp': time.time()}
        if 'density' in self.models:
            result['density'] = self.estimate_density(image)
        if 'anomaly' in self.models:
            result['anomaly'] = self.detect_anomaly(image)
        return result
    
    def get_model_info(self) -> Dict:
        """Return information about loaded models."""
        info = {}
        for name, model in self.models.items():
            n_params = sum(p.numel() for p in model.parameters())
            info[name] = {
                'parameters': n_params,
                'device': str(next(model.parameters()).device),
                'type': model.__class__.__name__,
            }
        return info


if __name__ == '__main__':
    pipe = CrowdVisionPipeline()
    print('\nModel info:')
    for name, info in pipe.get_model_info().items():
        print(f'  {name}: {info["type"]} ({info["parameters"]:,} params)')
