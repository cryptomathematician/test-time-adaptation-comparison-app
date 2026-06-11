"""No-reference (blind) image quality metrics.

These metrics evaluate image quality without requiring a ground-truth
reference — essential for real-world TTA deployment where GT is unavailable.
"""

import numpy as np
from typing import Dict, Optional


def compute_brisque(img_rgb: np.ndarray) -> Optional[float]:
    """Compute BRISQUE score using OpenCV contrib.

    Lower is better (0 = perfect, 100 = worst).

    Args:
        img_rgb: Image as uint8 numpy array with shape (H, W, 3) in RGB order.

    Returns:
        BRISQUE score, or None if computation fails.
    """
    try:
        import cv2
        # Convert RGB to BGR for OpenCV, then to grayscale
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Check if cv2.quality is available (requires opencv-contrib-python)
        brisque_model = cv2.quality.QualityBRISQUE_create(
            "", ""  # Will use default paths if available
        )
        score = brisque_model.compute(gray)
        return float(score) if isinstance(score, (int, float, np.floating)) else float(score[0])
    except (ImportError, AttributeError, cv2.error):
        # cv2.quality not available (need opencv-contrib-python)
        return None
    except Exception:
        return None


def compute_niqe(img_rgb: np.ndarray) -> Optional[float]:
    """Compute NIQE score using piq library.

    Lower is better (smaller values indicate better quality).

    Args:
        img_rgb: Image as uint8 numpy array with shape (H, W, 3) in RGB order.

    Returns:
        NIQE score, or None if computation fails.
    """
    try:
        import torch
        import piq

        # Convert to torch tensor: (1, 3, H, W) float32 in [0, 1]
        t = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            score = piq.niqe(t, data_range=1.0)
        return float(score.item())
    except ImportError:
        # piq not installed
        return None
    except Exception:
        return None


def compute_no_reference_metrics(img_rgb: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute all available no-reference quality metrics.

    Args:
        img_rgb: Image as uint8 numpy array with shape (H, W, 3) in RGB order.

    Returns:
        Dictionary with keys: "brisque", "niqe".
        Values are None if the corresponding metric could not be computed.
    """
    return {
        "brisque": compute_brisque(img_rgb),
        "niqe": compute_niqe(img_rgb),
    }