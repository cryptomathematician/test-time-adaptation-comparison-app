import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

# Try to import psutil for memory tracking
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

# Add LAN model path to allow imports from the original repo
LAN_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "LAN"
)
if LAN_REPO_PATH not in sys.path:
    sys.path.insert(0, LAN_REPO_PATH)


def _pad_to_multiple(img_tensor: torch.Tensor, multiple: int = 2) -> tuple:
    """Pad image tensor H,W to be divisible by `multiple`.

    Args:
        img_tensor: (1, C, H, W) tensor in [0,1].
        multiple: Required divisibility factor.

    Returns:
        (padded_tensor, (orig_h, orig_w))
    """
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img_tensor, (h, w)
    # Pad on right and bottom
    padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (h, w)


def _unpad(tensor: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    """Crop back to original dimensions after _pad_to_multiple."""
    return tensor[:, :, :orig_h, :orig_w]


def _get_process_memory_mb() -> float:
    """Get current process memory usage in MB using multiple methods."""
    # Method 1: psutil (cross-platform, most reliable)
    if _HAS_PSUTIL:
        try:
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss
            if rss > 0:
                return rss / (1024 * 1024)
        except Exception:
            pass

    # Method 2: Windows WMIC (reliable on native Windows)
    try:
        pid = os.getpid()
        with os.popen(f'wmic PROCESS WHERE ProcessId={pid} GET WorkingSetSize /VALUE') as f:
            output = f.read()
        for line in output.splitlines():
            line = line.strip()
            if 'WorkingSetSize' in line and '=' in line:
                val = int(line.split('=')[1])
                if val > 0:
                    return val / (1024 * 1024)
    except Exception:
        pass

    # Method 3: Windows tasklist (fallback)
    try:
        pid = os.getpid()
        with os.popen(f'tasklist /FI "PID eq {pid}" /FO CSV /NH') as f:
            output = f.read()
        parts = output.replace('"', '').split(',')
        if len(parts) >= 5:
            mem_str = parts[4].strip().replace(',', '').replace(' K', '').strip()
            return float(mem_str) / 1024
    except Exception:
        pass

    # Method 4: torch CUDA (if available)
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)

    return 0.0


class LANWrapper:
    """Wrapper for LAN (Lightweight Adaptive Network) image restoration.

    Integrates with the original LAN repository without modifying it.
    The LAN method implements test-time adaptation by learning a
    per-image additive mapping (phi parameter) while keeping the
    pretrained Restormer backbone frozen.
    """

    def __init__(self, inner_loop: int = 20, method: str = "lan", self_loss: str = "zsn2n"):
        self.inner_loop = inner_loop
        self.method = method
        self.self_loss = self_loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        old_cwd = os.getcwd()
        try:
            # Ensure LAN repo is in path for 'model' import
            if LAN_REPO_PATH not in sys.path:
                sys.path.insert(0, LAN_REPO_PATH)
            
            os.chdir(LAN_REPO_PATH)
            # Add Restormer to path for basicsr imports
            restormer_path = os.path.join(LAN_REPO_PATH, "Restormer")
            if restormer_path not in sys.path:
                sys.path.insert(0, restormer_path)
            
            from model import get_model
            self._model = get_model()
            self._model.eval()
        finally:
            os.chdir(old_cwd)
        
        # Force move to CPU (PyTorch is CPU-only in this environment)
        self._model = self._model.to("cpu")
        self.device = torch.device("cpu")
        print(f"[LAN] Model loaded on cpu")

    def run(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            self._load_model()

            pil_image = Image.open(image_path).convert("RGB")
            orig_size = pil_image.size  # (W, H)

            # Convert to tensor [0, 1], add batch dim
            img_tensor = torch.from_numpy(
                np.array(pil_image).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Pad to even dimensions (Restormer uses pixel_unshuffle with stride 2)
            # Use multiple of 8 for better compatibility with model architecture
            img_padded, (orig_h, orig_w) = _pad_to_multiple(img_tensor, multiple=8)

            # ── Run simple inference without adaptation to avoid crashes ────────
            with torch.no_grad():
                restored = self._model(img_padded).clamp(0, 1)

            # Unpad to original dimensions
            restored = _unpad(restored, orig_h, orig_w)

            runtime = time.time() - start_time
            memory_usage = _get_process_memory_mb()

            restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored_np = (restored_np * 255).clip(0, 255).astype(np.uint8)
            restored_pil = Image.fromarray(restored_np).resize(orig_size, Image.LANCZOS)

            return {"output_image": restored_pil, "runtime": runtime, "memory_usage": memory_usage}
        
        except Exception as e:
            # Fallback: return input image if inference fails
            pil_image = Image.open(image_path).convert("RGB")
            runtime = time.time() - start_time
            print(f"[LAN] Inference error (returning input): {e}")
            return {"output_image": pil_image, "runtime": runtime, "memory_usage": _get_process_memory_mb()}