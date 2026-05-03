"""
Anomaly detection evaluation metrics: AUC-ROC, EER, AP, F1.
"""

import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve,
                              average_precision_score, f1_score)
from typing import Dict, Tuple
import torch


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under ROC Curve."""
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate (where FAR = FRR)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0) * 100.0  # %


def compute_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision."""
    return float(average_precision_score(labels, scores))


@torch.no_grad()
def evaluate_anomaly_detection(model, train_loader, test_loader,
                                 device: str = 'cuda') -> Dict[str, float]:
    """
    Full anomaly detection evaluation:
    1. Compute anomaly scores on test set
    2. Calculate AUC, EER, AP

    Works for ConvAE and ConvLSTMAE.
    """
    model.eval()

    scores = []
    labels = []

    for batch in test_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            frames, lbl = batch
        else:
            frames, lbl = batch, torch.zeros(batch.shape[0])

        frames = frames.to(device)

        if hasattr(model, 'reconstruction_error'):
            err = model.reconstruction_error(frames).cpu().numpy()
        elif hasattr(model, 'anomaly_score'):
            err = model.anomaly_score(frames).cpu().numpy()
        else:
            raise ValueError("Model must implement reconstruction_error or anomaly_score")

        scores.extend(err.tolist())
        labels.extend(lbl.cpu().numpy().astype(int).tolist())

    scores = np.array(scores)
    labels = np.array(labels)

    # Normalise scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = scores

    auc = compute_auc(labels, scores_norm)
    eer = compute_eer(labels, scores_norm) if len(np.unique(labels)) > 1 else 50.0
    ap  = compute_ap(labels, scores_norm)

    # Threshold-based F1 at EER operating point
    threshold = np.percentile(scores_norm, (1 - labels.mean()) * 100)
    preds = (scores_norm >= threshold).astype(int)
    f1 = float(f1_score(labels, preds, zero_division=0))

    return {
        'auc':  auc * 100.0,   # percentage
        'eer':  eer,
        'ap':   ap * 100.0,
        'f1':   f1 * 100.0,
    }
