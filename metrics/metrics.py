import numpy as np
from typing import Dict, Optional
from .psnr import compute_psnr
from .ssim import compute_ssim
from .lpips import compute_lpips


def compute_metrics(
    restored_image: np.ndarray,
    ground_truth_image: np.ndarray,
) -> Dict[str, float]:
    """Compute all quality metrics between restored and ground-truth images.

    Both images should be numpy arrays in [0, 255] range with shape (H, W, C).

    Args:
        restored_image: The restored/output image.
        ground_truth_image: The ground-truth reference image.

    Returns:
        Dictionary with keys: "psnr", "ssim", "lpips".
    """
    psnr_val = compute_psnr(restored_image, ground_truth_image)
    ssim_val = compute_ssim(restored_image, ground_truth_image)
    lpips_val = compute_lpips(restored_image, ground_truth_image)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
    }