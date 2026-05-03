"""
Forecasting evaluation metrics: MAE, RMSE, MAPE at multiple horizons.
Masks zero (missing) values as done in the DCRNN / Graph WaveNet papers.
"""

import numpy as np
import torch
from typing import Dict, List


def masked_mae(pred: np.ndarray, target: np.ndarray,
               null_val: float = 0.0) -> float:
    mask = (target != null_val).astype(float)
    mask /= mask.mean() + 1e-8
    loss = np.abs(pred - target) * mask
    return float(np.nanmean(loss))


def masked_mse(pred: np.ndarray, target: np.ndarray,
               null_val: float = 0.0) -> float:
    mask = (target != null_val).astype(float)
    mask /= mask.mean() + 1e-8
    loss = (pred - target) ** 2 * mask
    return float(np.sqrt(np.nanmean(loss)))


def masked_mape(pred: np.ndarray, target: np.ndarray,
                null_val: float = 0.0, eps: float = 1e-8) -> float:
    mask = (np.abs(target) > eps).astype(float)
    mask /= mask.mean() + 1e-8
    loss = np.abs((pred - target) / (np.abs(target) + eps)) * mask
    return float(np.nanmean(loss)) * 100.0  # percentage


@torch.no_grad()
def evaluate_forecasting(model, loader, scaler, adj: torch.Tensor,
                          device: str = 'cuda',
                          horizons: List[int] = [3, 6, 12]) -> Dict:
    """
    Evaluate a forecasting model at multiple prediction horizons.

    Args:
        horizons: list of time-step indices to evaluate (0-indexed); e.g. 3=15min, 6=30min, 12=60min

    Returns:
        dict: {horizon_str: {mae, rmse, mape}} + 'overall'
    """
    model.eval()
    adj = adj.to(device)
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, adj)     # [B, T_out, N, 1]
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)  # [N_samples, T_out, N, 1]
    targets = np.concatenate(all_targets, axis=0)

    # Inverse-normalise
    preds   = scaler.inverse_transform(preds[..., 0])    # [N_samples, T_out, N]
    targets = scaler.inverse_transform(targets[..., 0])

    results = {}
    for h in horizons:
        if h <= preds.shape[1]:
            p = preds[:, h-1, :]    # [N_samples, N]
            t = targets[:, h-1, :]

            min_h = h * 5   # assuming 5-min intervals
            results[f'{min_h}min'] = {
                'mae':  masked_mae(p, t),
                'rmse': masked_mse(p, t),
                'mape': masked_mape(p, t),
            }

    # Overall (all horizons)
    results['overall'] = {
        'mae':  masked_mae(preds, targets),
        'rmse': masked_mse(preds, targets),
        'mape': masked_mape(preds, targets),
    }
    return results
