import numpy as np
import math


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image (H x W x C) in [0, 255] range.
        img2: Second image (H x W x C) in [0, 255] range.

    Returns:
        PSNR value in dB.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))