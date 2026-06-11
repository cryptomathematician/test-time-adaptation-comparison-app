import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from typing import Dict, Any
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms

# Try to import psutil for memory tracking
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

TTAD_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "TTAD"
)
if TTAD_REPO_PATH not in sys.path:
    sys.path.insert(0, TTAD_REPO_PATH)

# The TTAD UNet has 5 MaxPool(2) layers → requires H,W divisible by 32
UNET_MULTIPLE = 32


def _pad_to_multiple(img_tensor: torch.Tensor, multiple: int = UNET_MULTIPLE) -> tuple:
    """Pad image tensor H,W to be divisible by `multiple`.

    Args:
        img_tensor: (1, C, H, W) tensor in [0,1] range.
        multiple: Required divisibility factor.

    Returns:
        (padded_tensor, (orig_h, orig_w))
    """
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img_tensor, (h, w)
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

    # Method 2: Windows WMIC
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

    # Method 3: Windows tasklist
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

    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


class TTADWrapper:
    """Wrapper for TTAD (Test-Time Adaptation with Diffusion) image restoration."""

    def __init__(
        self,
        max_epoch: int = 20,
        lr: float = 0.0001,
        banknum: int = 10,
        loss_type: str = "L2",
    ):
        self.max_epoch = max_epoch
        self.lr = lr
        self.banknum = banknum
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        old_cwd = os.getcwd()
        try:
            # Change to TTAD directory for imports to work
            os.chdir(TTAD_REPO_PATH)
            
            # Add to path to ensure arch_unet can be imported
            if TTAD_REPO_PATH not in sys.path:
                sys.path.insert(0, TTAD_REPO_PATH)
            
            # Import the UNet class
            from arch_unet import UNet

            checkpoint_path = os.path.join(TTAD_REPO_PATH, "epoch_model_100.pth")

            self._model = UNet()
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                self._model.load_state_dict(checkpoint, strict=True)
                print(f"[TTAD] Model loaded from {checkpoint_path}")
            else:
                print(f"[TTAD] No pretrained checkpoint found at {checkpoint_path}, using randomly initialized model")
        finally:
            os.chdir(old_cwd)

        self._model = self._model.to(self.device)
        self._model.train()
        print(f"[TTAD] Model loaded on {self.device}")

    def run(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            self._load_model()

            # ── Load image ─────────────────────────────────────────────────────────
            pil_image = Image.open(image_path).convert("RGB")
            orig_size = pil_image.size  # (W, H)

            transform = transforms.Compose([transforms.ToTensor()])
            img_tensor = transform(pil_image).unsqueeze(0).to(self.device)  # (1, C, H, W)

            # Pad to dimensions divisible by 32 (required by UNet with 5x MaxPool)
            img_padded, (orig_h, orig_w) = _pad_to_multiple(img_tensor, UNET_MULTIPLE)

            _, _, H, W = img_padded.shape
            C = 3

            # ── Build synthetic pixel bank ─────────────────────────────────────────
            img_np = img_padded.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            bank_list = []
            for k in range(self.banknum):
                noise = np.random.normal(0, 0.02, img_np.shape).astype(np.float32)
                perturbed = np.clip(img_np + noise, 0, 1)
                bank_list.append(torch.from_numpy(np.transpose(perturbed, (2, 0, 1))))  # (C, H, W)

            # Stack into bank: (N, C, H, W)
            img_bank = torch.stack(bank_list, dim=0).to(self.device)

            # ── Setup optimizer and scheduler ──────────────────────────────────────
            optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
            scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

            mse_loss = nn.MSELoss()
            l1_loss = nn.L1Loss()

            # ── Adaptation loop ────────────────────────────────────────────────────
            for epoch in range(self.max_epoch):
                self._model.train()

                idx1 = torch.randint(0, self.banknum, size=(1,)).item()
                idx2 = torch.randint(0, self.banknum, size=(1,)).item()
                while idx2 == idx1:
                    idx2 = torch.randint(0, self.banknum, size=(1,)).item()

                img1 = img_bank[idx1:idx1+1]
                img2 = img_bank[idx2:idx2+1]

                pred = self._model(img1)

                if self.loss_type == "L2":
                    loss = mse_loss(pred, img2)
                else:
                    loss = l1_loss(pred, img2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # ── Final inference ────────────────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                pred = self._model(img_padded)
                # The UNet directly predicts the restored image (self-supervised denoising)
                restored_padded = torch.clamp(pred, 0, 1)

            # Crop back to original dimensions
            restored = _unpad(restored_padded, orig_h, orig_w)

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
            print(f"[TTAD] Inference error (returning input): {e}")
            return {"output_image": pil_image, "runtime": runtime, "memory_usage": _get_process_memory_mb()}