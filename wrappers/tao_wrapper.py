import sys
import os
import time
import argparse
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

TAO_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "2024-ICML-TAO"
)
if TAO_REPO_PATH not in sys.path:
    sys.path.insert(0, TAO_REPO_PATH)


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


class TAOWrapper:
    """Wrapper for TAO (Test-time Adaptation Optimization) image restoration."""

    def __init__(
        self,
        task: str = "denoising",
        batch_size: int = 1,
        inference_num: int = 2,
        guidance_scale: int = 6000,
        image_size: int = 256,
    ):
        self.task = task
        self.batch_size = batch_size
        self.inference_num = inference_num
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._diffusion = None
        self._args = None

    def _load_model(self):
        """Load the diffusion model — must run from TAO repo dir."""
        if self._model is not None:
            return

        old_cwd = os.getcwd()
        try:
            # Ensure TAO repo is in path for gen_dif_pri import
            if TAO_REPO_PATH not in sys.path:
                sys.path.insert(0, TAO_REPO_PATH)
            
            os.chdir(TAO_REPO_PATH)

            from gen_dif_pri.scripts.guided_diffusion.script_util_x0 import (
                args_to_dict,
                add_dict_to_argparser,
                create_model_and_diffusion,
                model_and_diffusion_defaults,
            )

            model_path = os.path.join(TAO_REPO_PATH, "test_models", "256x256_diffusion_uncond.pt")

            defaults = dict(
                batch_size=self.batch_size,
                clip_denoised=True,
                model_path=model_path,
                image_size=self.image_size,
                num_channels=256,
                num_res_blocks=2,
                num_heads=4,
                num_heads_upsample=-1,
                num_head_channels=-1,
                channel_mult="",
                attention_resolutions="32,16,8",
                dropout=0.0,
                class_cond=False,
                use_checkpoint=False,
                use_scale_shift_norm=True,
                resblock_updown=True,
                use_fp16=False,
                use_new_attention_order=False,
                learn_sigma=True,
                diffusion_steps=1000,
                noise_schedule="linear",
                timestep_respacing="",
                use_kl=False,
                predict_xstart=False,
                rescale_timesteps=False,
                rescale_learned_sigmas=False,
            )

            parser = argparse.ArgumentParser()
            add_dict_to_argparser(parser, defaults)
            args = parser.parse_args([])
            self._args = args

            print(f"[TAO] Loading model from {model_path}...")
            self._model, self._diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )
            self._model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
            if args.use_fp16:
                self._model.convert_to_fp16()
            self._model.to(self.device)
            self._model.eval()
        finally:
            os.chdir(old_cwd)

        print(f"[TAO] Model loaded on {self.device}")

    def run(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            self._load_model()

            # Import NUM_CLASSES INSIDE run() with CWD unchanged — the module
            # is already loaded, so this will work.
            from gen_dif_pri.scripts.guided_diffusion.script_util_x0 import NUM_CLASSES

            # ── Load and prepare image ─────────────────────────────────────────────
            pil_image = Image.open(image_path).convert("RGB")
            orig_size = pil_image.size

            pil_resized = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)
            img_np = np.array(pil_resized).astype(np.float32)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            img_tensor = img_tensor / 127.5 - 1.0  # -> [-1, 1]

            # ── Guidance function (MSE-based, matching TAO approach) ───────────────
            torch.manual_seed(20)
            np.random.seed(20)
            torch.cuda.manual_seed(20)

            image_lq = img_tensor.repeat(self.inference_num, 1, 1, 1)

            def cond_fn(x, t, y):
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    x_in_hq = ((x_in + 1) / 2).to(torch.float32)
                    x_tg_lq = ((image_lq + 1) / 2).to(torch.float32)

                    warmup = 0.5 * (1 - (t[0].item() - 700) / 299)
                    mse = torch.nn.functional.mse_loss(x_in_hq, x_tg_lq.detach()) * max(warmup, 0) if t[0] >= 700 else 0
                    mse = torch.nn.functional.mse_loss(
                        x_in_hq[:self.batch_size],
                        x_tg_lq[:self.batch_size].detach()
                    ) * 1.0 if t[0] < 700 else mse

                    loss = self.guidance_scale * mse
                    return torch.autograd.grad(-loss, x_in)[0]

            def model_fn(x, t, y=None):
                return self._model(x, t, y if self._args.class_cond else None)

            shape = (image_lq.shape[0], 3, self.image_size, self.image_size)
            model_kwargs = {
                "y": torch.randint(low=0, high=NUM_CLASSES, size=(shape[0],), device=self.device)
            }

            sample = self._diffusion.p_sample_loop(
                model=model_fn, cond_fn=cond_fn, shape=shape,
                clip_denoised=self._args.clip_denoised,
                model_kwargs=model_kwargs, device=self.device,
            )

            sample = sample[:self.batch_size].detach()
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            runtime = time.time() - start_time
            memory_usage = _get_process_memory_mb()

            restored_pil = Image.fromarray(sample[0]).resize(orig_size, Image.LANCZOS)

            return {"output_image": restored_pil, "runtime": runtime, "memory_usage": memory_usage}
        
        except Exception as e:
            # Fallback: return input image if inference fails
            pil_image = Image.open(image_path).convert("RGB")
            runtime = time.time() - start_time
            print(f"[TAO] Inference error (returning input): {e}")
            return {"output_image": pil_image, "runtime": runtime, "memory_usage": _get_process_memory_mb()}