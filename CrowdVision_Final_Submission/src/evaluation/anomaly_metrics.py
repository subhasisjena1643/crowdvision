"""
Anomaly detection evaluation metrics: AUC-ROC, EER, AP, F1.

Key improvement: Per-clip score normalization for UCSD datasets.
This is the standard protocol — each test clip has its own baseline
reconstruction error, and anomaly detection should be relative to that.

Exports:
    compute_auc, compute_eer, compute_ap
    _collect_scores          – extract raw (scores, labels) from a model+loader
    evaluate_anomaly_detection – full metric suite
"""

import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve,
                              average_precision_score, f1_score)
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


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
def _collect_scores(model, test_loader,
                    device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect raw anomaly scores and ground-truth labels from a test loader.

    Returns:
        (scores, labels) — both 1-D numpy arrays of the same length.
        Scores are NOT normalised (raw reconstruction / prediction error).

    Works for ConvAE, ConvLSTMAE, and FutureFrameNet.
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

        is_clip = (frames.dim() == 5)
        is_prediction = hasattr(model, 'prediction_error')

        if is_prediction and is_clip:
            # Prediction model (FutureFrameNet): pass full clip
            frames_eval = frames
            B = frames.shape[0]
            # Predicts the last frame, so use the last frame's label
            lbl = lbl[:, -1] if lbl.dim() == 2 else lbl
        elif is_clip and not hasattr(model, 'temp_enc'):
            # Frame-level reconstruction model: flatten to [B*T,C,H,W]
            B, T, C, H, W = frames.shape
            frames_eval = frames.reshape(B * T, C, H, W)
            lbl = lbl.reshape(B * T) if lbl.dim() == 2 else lbl
        elif is_clip and hasattr(model, 'temp_enc'):
            # ConvLSTMAE: returns [B, T] error, so we evaluate frame-by-frame
            frames_eval = frames
            B, T = frames.shape[:2]
            lbl = lbl.reshape(B * T) if lbl.dim() == 2 else lbl
        else:
            # Single frame mode
            frames_eval = frames
            B, T = frames.shape[0], 1
            lbl = lbl.view(-1) if lbl.dim() == 2 else lbl

        if hasattr(model, 'reconstruction_error'):
            err = model.reconstruction_error(frames_eval).cpu().numpy()
        elif hasattr(model, 'anomaly_score'):
            err = model.anomaly_score(frames_eval).cpu().numpy()
        else:
            raise ValueError("Model must implement reconstruction_error or anomaly_score")

        scores.extend(err.reshape(-1).tolist())
        labels.extend(lbl.cpu().numpy().astype(int).tolist())

    return np.array(scores), np.array(labels)


@torch.no_grad()
def _collect_scores_per_clip(model, data_root: str, ped: str = 'ped2',
                              device: str = 'cuda',
                              img_size=(128, 192)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect anomaly scores with PER-CLIP normalization.
    
    This is the standard evaluation protocol for UCSD Ped2:
    1. For each test clip, compute all frame-level reconstruction errors
    2. Normalize errors within each clip to [0, 1]
    3. Concatenate all normalized scores
    
    This removes inter-clip bias and focuses on RELATIVE anomaly within each clip.
    """
    from pathlib import Path
    from src.data_loaders.ucsd import _get_frame_paths, _load_frame, UCSDTestDataset
    from torchvision import transforms
    from PIL import Image

    model.eval()

    h, w = img_size
    tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset_dir = Path(data_root) / 'UCSD_Anomaly_Dataset.v1p2' / f'UCSDped{ped[-1]}' / 'Test'

    # Get all test clip directories
    seq_dirs = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and '_gt' not in d.name
                       and not d.name.startswith('.')])

    all_scores = []
    all_labels = []
    is_prediction = hasattr(model, 'prediction_error')

    for seq_dir in seq_dirs:
        frames_paths = _get_frame_paths(seq_dir)
        gt_dir = dataset_dir / (seq_dir.name + '_gt')
        has_gt = gt_dir.exists()

        # Load all frames in this clip
        clip_frames = []
        clip_labels = []
        for fp in frames_paths:
            fr = _load_frame(fp)
            if fr is None:
                continue
            img = Image.fromarray(fr)
            img_t = tf(img)
            clip_frames.append(img_t)

            # Label
            gt_file = gt_dir / (fp.stem + '.bmp')
            if not gt_file.exists():
                gt_file = gt_dir / (fp.stem + '.tif')
            if has_gt and gt_file.exists():
                gt_fr = _load_frame(gt_file)
                clip_labels.append(1 if gt_fr is not None and gt_fr.sum() > 0 else 0)
            else:
                clip_labels.append(0)

        if len(clip_frames) == 0:
            continue

        if is_prediction:
            # FutureFrameNet: need clips of consecutive frames
            clip_len = 5  # 4 input + 1 target
            clip_scores = []
            # First clip_len-1 frames don't have predictions
            for i in range(clip_len - 1):
                clip_scores.append(0.0)  # placeholder

            for i in range(len(clip_frames) - clip_len + 1):
                batch = torch.stack(clip_frames[i:i + clip_len]).unsqueeze(0).to(device)
                err = model.reconstruction_error(batch).cpu().item()
                clip_scores.append(err)

            clip_scores = np.array(clip_scores)
            # Only keep labels for frames that have scores
            clip_labels_arr = np.array(clip_labels[:len(clip_scores)])
        else:
            # ConvAE: process all frames at once
            batch = torch.stack(clip_frames).to(device)
            # Process in smaller batches to avoid OOM
            clip_scores = []
            for start in range(0, len(batch), 64):
                end = min(start + 64, len(batch))
                mini_batch = batch[start:end]
                err = model.reconstruction_error(mini_batch).cpu().numpy()
                clip_scores.extend(err.tolist())
            clip_scores = np.array(clip_scores)
            clip_labels_arr = np.array(clip_labels[:len(clip_scores)])

        # *** PER-CLIP NORMALIZATION ***
        s_min, s_max = clip_scores.min(), clip_scores.max()
        if s_max > s_min:
            clip_scores_norm = (clip_scores - s_min) / (s_max - s_min)
        else:
            clip_scores_norm = clip_scores

        all_scores.extend(clip_scores_norm.tolist())
        all_labels.extend(clip_labels_arr.tolist())

    return np.array(all_scores), np.array(all_labels)


@torch.no_grad()
def evaluate_anomaly_detection(model, train_loader, test_loader,
                                 device: str = 'cuda',
                                 data_root: str = None,
                                 ped: str = 'ped2',
                                 use_per_clip: bool = False) -> Dict[str, float]:
    """
    Full anomaly detection evaluation:
    1. Compute anomaly scores on test set
    2. Calculate AUC, EER, AP, F1

    Args:
        use_per_clip: If True, use per-clip normalization (standard UCSD protocol).
                      Requires data_root and ped.

    Works for ConvAE, ConvLSTMAE, and FutureFrameNet.
    """
    if use_per_clip and data_root:
        scores, labels = _collect_scores_per_clip(model, data_root, ped, device)
    else:
        scores, labels = _collect_scores(model, test_loader, device)

    scores = np.array(scores)
    labels = np.array(labels)

    # Guard against NaN/Inf from unstable early training
    nan_mask = ~np.isfinite(scores)
    if nan_mask.any():
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise scores to [0, 1] (global — per-clip already normalized)
    if not use_per_clip:
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = scores
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
