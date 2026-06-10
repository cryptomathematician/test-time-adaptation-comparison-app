import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

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
        self._load_model()

        pil_image = Image.open(image_path).convert("RGB")
        orig_size = pil_image.size  # (W, H)

        # Convert to tensor [0, 1], add batch dim
        img_tensor = torch.from_numpy(
            np.array(pil_image).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad to even dimensions (Restormer uses pixel_unshuffle with stride 2)
        img_padded, (orig_h, orig_w) = _pad_to_multiple(img_tensor, multiple=2)

        # ── Run LAN adaptation ────────────────────────────────────────────────
        start_time = time.time()

        class LanPhi(torch.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.phi = torch.nn.parameter.Parameter(torch.zeros(shape), requires_grad=True)
            def forward(self, x):
                return x + torch.tanh(self.phi)

        lan_phi = LanPhi(img_padded.shape).to(self.device) if self.method == "lan" else torch.nn.Identity()
        params = list(lan_phi.parameters()) if self.method == "lan" else list(self._model.parameters())
        optimizer = torch.optim.Adam(params, lr=5e-4 if self.method == "lan" else 5e-6)

        # Load loss function
        old_cwd = os.getcwd()
        try:
            os.chdir(LAN_REPO_PATH)
            from adapt.zsn2n import loss_func as zsn2n_loss
            from adapt.nbr2nbr import loss_func as nbr2nbr_loss
        finally:
            os.chdir(old_cwd)

        if self.self_loss == "zsn2n":
            loss_func = zsn2n_loss
        elif self.self_loss == "nbr2nbr":
            loss_func = nbr2nbr_loss
        else:
            raise ValueError(f"Unknown self_loss: {self.self_loss}")

        for i in range(self.inner_loop):
            optimizer.zero_grad()
            adapted_input = lan_phi(img_padded)
            with torch.no_grad():
                pred = self._model(adapted_input).clamp(0, 1)
            loss = loss_func(adapted_input, self._model, i, self.inner_loop)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            adapted_input = lan_phi(img_padded)
            restored = self._model(adapted_input).clamp(0, 1)

        # Unpad to original dimensions
        restored = _unpad(restored, orig_h, orig_w)

        runtime = time.time() - start_time

        memory_usage = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_usage = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_np = (restored_np * 255).clip(0, 255).astype(np.uint8)
        restored_pil = Image.fromarray(restored_np).resize(orig_size, Image.LANCZOS)

        return {"output_image": restored_pil, "runtime": runtime, "memory_usage": memory_usage}