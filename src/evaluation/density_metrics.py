"""
Crowd density estimation evaluation metrics.

Primary:  MAE, MSE
Secondary: PSNR, SSIM (density map quality)
Advanced:  GAME (Grid Average Mean absolute Error) — required for CVPR papers
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Dict, List


def mae(pred_counts: np.ndarray, gt_counts: np.ndarray) -> float:
    """Mean Absolute Error on count predictions."""
    return float(np.abs(pred_counts - gt_counts).mean())


def mse(pred_counts: np.ndarray, gt_counts: np.ndarray) -> float:
    """Mean Squared Error on count predictions."""
    return float(np.sqrt(((pred_counts - gt_counts) ** 2).mean()))


def game_metric(pred_map: np.ndarray, gt_map: np.ndarray, level: int = 0) -> float:
    """
    GAME (Grid Average Mean absolute Error) at a given level.

    level=0 → same as MAE (whole image)
    level=L → image divided into 4^L patches; avg patch-level MAE.
    """
    h, w = pred_map.shape[:2]
    n_patches = 4 ** level
    side = int(np.sqrt(n_patches))   # patches per side
    ph, pw = h // side, w // side

    total_error = 0.0
    count = 0
    for i in range(side):
        for j in range(side):
            p_patch = pred_map[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            g_patch = gt_map[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            total_error += abs(p_patch.sum() - g_patch.sum())
            count += 1
    return total_error / count if count > 0 else 0.0


def psnr_density(pred_map: np.ndarray, gt_map: np.ndarray,
                 data_range: float = None) -> float:
    """PSNR between two density maps."""
    if data_range is None:
        data_range = max(gt_map.max(), pred_map.max(), 1e-6)
    pred_norm = np.clip(pred_map / data_range, 0, 1)
    gt_norm   = np.clip(gt_map  / data_range, 0, 1)
    return peak_signal_noise_ratio(gt_norm, pred_norm, data_range=1.0)


def ssim_density(pred_map: np.ndarray, gt_map: np.ndarray,
                 data_range: float = None) -> float:
    """SSIM between two density maps."""
    if data_range is None:
        data_range = max(gt_map.max(), pred_map.max(), 1e-6)
    pred_norm = np.clip(pred_map / data_range, 0, 1)
    gt_norm   = np.clip(gt_map  / data_range, 0, 1)
    return structural_similarity(gt_norm, pred_norm, data_range=1.0)


@torch.no_grad()
def evaluate_density(model, loader, device: str = 'cuda') -> Dict[str, float]:
    """
    Full evaluation of a density estimation model.

    Returns:
        dict with keys: mae, mse, psnr, ssim, game0, game1, game2, game3
    """
    model.eval()
    pred_counts, gt_counts = [], []
    psnrs, ssims = [], []
    game_scores = {0: [], 1: [], 2: [], 3: []}

    for imgs, density_gt, counts_gt in loader:
        imgs = imgs.to(device)
        density_gt = density_gt.to(device)

        pred = model(imgs)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]

        for b in range(imgs.shape[0]):
            pred_map = pred[b, 0].cpu().numpy()
            gt_map   = density_gt[b, 0].cpu().numpy()
            pred_cnt = pred_map.sum()
            gt_cnt   = gt_map.sum()

            pred_counts.append(pred_cnt)
            gt_counts.append(gt_cnt)
            psnrs.append(psnr_density(pred_map, gt_map))
            ssims.append(ssim_density(pred_map, gt_map))
            for lvl in range(4):
                game_scores[lvl].append(game_metric(pred_map, gt_map, lvl))

    pred_counts = np.array(pred_counts)
    gt_counts   = np.array(gt_counts)

    return {
        'mae':   mae(pred_counts, gt_counts),
        'mse':   mse(pred_counts, gt_counts),
        'psnr':  float(np.mean(psnrs)),
        'ssim':  float(np.mean(ssims)),
        'game0': float(np.mean(game_scores[0])),
        'game1': float(np.mean(game_scores[1])),
        'game2': float(np.mean(game_scores[2])),
        'game3': float(np.mean(game_scores[3])),
    }
