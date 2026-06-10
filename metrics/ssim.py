import numpy as np


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    Uses a simplified but accurate SSIM implementation. Falls back to
    scikit-image if available for the standard implementation.

    Args:
        img1: First image (H x W x C) in [0, 255] range.
        img2: Second image (H x W x C) in [0, 255] range.

    Returns:
        SSIM value in [0, 1].
    """
    try:
        from skimage.metrics import structural_similarity
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        # skimage SSIM expects data_range
        return float(structural_similarity(img1_f, img2_f, channel_axis=-1, data_range=255))
    except ImportError:
        return _compute_ssim_fallback(img1, img2)


def _compute_ssim_fallback(img1: np.ndarray, img2: np.ndarray, K1: float = 0.01, K2: float = 0.03) -> float:
    """Fallback SSIM implementation using sliding window.

    Args:
        img1: First image (H x W x C) in [0, 255] range.
        img2: Second image (H x W x C) in [0, 255] range.
        K1, K2: Stability constants.

    Returns:
        Mean SSIM across all channels.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # If images are 2D (grayscale), add channel dimension
    if img1.ndim == 2:
        img1 = img1[..., np.newaxis]
        img2 = img2[..., np.newaxis]

    C1 = (K1 * 255) ** 2
    C2 = (K2 * 255) ** 2

    window_size = 11
    window = _gaussian_window(window_size, 1.5)
    window = window[np.newaxis, :, np.newaxis]  # 1 x window_size x 1
    window = window * window.transpose(0, 2, 1)  # window_size x window_size

    ssim_per_channel = []
    for c in range(img1.shape[2]):
        mu1 = _convolve(img1[..., c], window)
        mu2 = _convolve(img2[..., c], window)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _convolve(img1[..., c] ** 2, window) - mu1_sq
        sigma2_sq = _convolve(img2[..., c] ** 2, window) - mu2_sq
        sigma12 = _convolve(img1[..., c] * img2[..., c], window) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_per_channel.append(np.mean(ssim_map))

    return float(np.mean(ssim_per_channel))


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """Create a 1D Gaussian window."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _convolve(img: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Simple 2D convolution using the provided window."""
    from scipy.ndimage import convolve
    return convolve(img, window, mode='reflect')