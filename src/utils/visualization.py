"""Visualisation utilities for CrowdVision."""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


def _to_numpy_img(tensor_or_array):
    """Convert a CHW tensor or HWC array to uint8 HWC numpy."""
    import torch
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = np.asarray(tensor_or_array)
        if arr.dtype != np.uint8:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def show_density_map(img, density_pred, density_gt=None,
                     pred_count=None, gt_count=None,
                     title: str = '',
                     save_path: str = None,
                     figsize=(14, 5)):
    """
    Display an image alongside its predicted (and optionally GT) density map.
    """
    ncols = 3 if density_gt is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    img_np = _to_numpy_img(img)
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    import torch
    if isinstance(density_pred, torch.Tensor):
        density_pred = density_pred.squeeze().detach().cpu().numpy()
    im = axes[1].imshow(density_pred, cmap='jet')
    cnt_str = f'  (count: {density_pred.sum():.1f})'
    if pred_count is not None:
        cnt_str = f'  (count: {pred_count:.1f})'
    axes[1].set_title('Predicted Density' + cnt_str)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if density_gt is not None:
        if isinstance(density_gt, torch.Tensor):
            density_gt = density_gt.squeeze().detach().cpu().numpy()
        im2 = axes[2].imshow(density_gt, cmap='jet')
        cnt_str2 = f'  (count: {density_gt.sum():.1f})'
        if gt_count is not None:
            cnt_str2 = f'  (count: {gt_count:.1f})'
        axes[2].set_title('Ground Truth Density' + cnt_str2)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def show_anomaly_heatmap(frame, recon_frame=None, error_map=None,
                          score: float = None, save_path: str = None,
                          figsize=(14, 4)):
    """Display frame, reconstruction, and pixel-level error heatmap."""
    parts = [('Input Frame', frame)]
    if recon_frame is not None:
        parts.append(('Reconstruction', recon_frame))
    if error_map is not None:
        parts.append(('Error Heatmap', error_map))

    fig, axes = plt.subplots(1, len(parts), figsize=figsize)
    if len(parts) == 1:
        axes = [axes]

    for ax, (title, data) in zip(axes, parts):
        arr = _to_numpy_img(data)
        if 'Error' in title:
            ax.imshow(arr, cmap='hot')
        else:
            ax.imshow(arr.squeeze(), cmap='gray' if arr.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')

    if score is not None:
        fig.suptitle(f'Anomaly Score: {score:.4f}', fontsize=13,
                     color='red' if score > 0.5 else 'green')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def plot_forecasting(pred: np.ndarray, target: np.ndarray,
                     node_idx: int = 0,
                     sensor_name: str = None,
                     horizons_min: List[int] = None,
                     save_path: str = None, figsize=(12, 4)):
    """Plot predicted vs actual traffic speed for one sensor."""
    fig, ax = plt.subplots(figsize=figsize)

    T = pred.shape[0] if pred.ndim == 1 else pred.shape[0]
    x_axis = np.arange(T)

    if pred.ndim > 1:
        pred_v   = pred[:, node_idx]
        target_v = target[:, node_idx]
    else:
        pred_v, target_v = pred, target

    ax.plot(x_axis, target_v, label='Ground Truth', color='blue', linewidth=1.5)
    ax.plot(x_axis, pred_v,   label='Predicted',    color='red',  linewidth=1.5, linestyle='--')

    if horizons_min:
        for h in horizons_min:
            if h < T:
                ax.axvline(h, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Time Step (5 min each)')
    ax.set_ylabel('Speed (mph)')
    name = sensor_name or f'Node {node_idx}'
    ax.set_title(f'METR-LA Forecast — {name}')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def plot_training_curves(history: Dict, metric_keys: List[str] = None,
                          save_path: str = None, figsize=(12, 5)):
    """Plot train/val curves from a training history dict."""
    if metric_keys is None:
        # Use all keys from first epoch
        metric_keys = list(history.get('train', [{}])[0].keys())

    n = len(metric_keys)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, metric_keys):
        train_vals = [e.get(key) for e in history.get('train', []) if key in e]
        val_vals   = [e.get(key) for e in history.get('val', []) if key in e]
        if train_vals:
            ax.plot(train_vals, label='Train')
        if val_vals:
            ax.plot(val_vals, label='Val')
        ax.set_title(key.upper())
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def make_results_table(results: Dict[str, Dict[str, float]],
                        columns: List[str] = None) -> str:
    """
    Format a results dict as a Markdown table for the paper.

    Args:
        results: {method_name: {metric: value, ...}}
        columns: ordered list of metrics to display
    """
    if not results:
        return ""
    # Auto-detect columns
    if columns is None:
        all_keys = set()
        for v in results.values():
            all_keys.update(v.keys())
        columns = sorted(all_keys)

    header = "| Method | " + " | ".join(columns) + " |"
    sep    = "|--------|" + "|".join(["--------"] * len(columns)) + "|"
    rows   = [header, sep]

    for method, metrics in results.items():
        vals = [f"{metrics.get(c, '-'):.2f}" if isinstance(metrics.get(c), float)
                else str(metrics.get(c, '-'))
                for c in columns]
        rows.append(f"| {method} | " + " | ".join(vals) + " |")

    return "\n".join(rows)
