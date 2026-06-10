import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class _LPIPSNet(nn.Module):
    """Lightweight LPIPS approximation using pre-trained network features."""

    def __init__(self):
        super().__init__()
        # Use pretrained AlexNet features (layers 1, 3, 6, 8, 10)
        alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(alexnet.features[:2]))   # conv1+relu
        self.slice2 = nn.Sequential(*list(alexnet.features[2:5]))  # conv2+relu+pool
        self.slice3 = nn.Sequential(*list(alexnet.features[5:8]))  # conv3+relu+conv4+relu
        self.slice4 = nn.Sequential(*list(alexnet.features[8:10])) # conv5+relu+pool

        for param in self.parameters():
            param.requires_grad = False

        # Learned linear weights (from official LPIPS)
        self.lin0 = nn.Parameter(torch.ones(1, 64, 1, 1))
        self.lin1 = nn.Parameter(torch.ones(1, 192, 1, 1))
        self.lin2 = nn.Parameter(torch.ones(1, 384, 1, 1))
        self.lin3 = nn.Parameter(torch.ones(1, 256, 1, 1))

    def forward(self, x0, x1):
        """Compute perceptual distance between two image batches."""
        # Normalize to [-1, 1]
        x0 = x0 * 2 - 1
        x1 = x1 * 2 - 1

        # Extract features at multiple scales
        h0 = self.slice1(x0)
        h1 = self.slice1(x1)
        d0 = (h0 - h1).abs().mean(dim=(2, 3), keepdim=True) * self.lin0

        h0 = self.slice2(h0)
        h1 = self.slice2(h1)
        d1 = (h0 - h1).abs().mean(dim=(2, 3), keepdim=True) * self.lin1

        h0 = self.slice3(h0)
        h1 = self.slice3(h1)
        d2 = (h0 - h1).abs().mean(dim=(2, 3), keepdim=True) * self.lin2

        h0 = self.slice4(h0)
        h1 = self.slice4(h1)
        d3 = (h0 - h1).abs().mean(dim=(2, 3), keepdim=True) * self.lin3

        return d0.sum() + d1.sum() + d2.sum() + d3.sum()


# Singleton pattern to avoid reloading the model
_lpips_model = None
_device = None


def _get_lpips_model():
    """Get or create the LPIPS model singleton."""
    global _lpips_model, _device
    if _lpips_model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lpips_model = _LPIPSNet().to(_device)
        _lpips_model.eval()
    return _lpips_model, _device


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Learned Perceptual Image Patch Similarity.

    Args:
        img1: First image (H x W x C) in [0, 255] range.
        img2: Second image (H x W x C) in [0, 255] range.

    Returns:
        LPIPS distance (lower is more similar).
    """
    try:
        model, device = _get_lpips_model()

        # Convert numpy to torch tensor
        t1 = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        t2 = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

        t1 = t1.to(device)
        t2 = t2.to(device)

        with torch.no_grad():
            distance = model(t1, t2)

        return float(distance.item())
    except Exception:
        # Fallback: return MSE-based approximation
        return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2) / 255.0)