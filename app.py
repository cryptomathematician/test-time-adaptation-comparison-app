"""
Test-Time Adaptation Image Restoration Benchmark — Dual Mode

Mode A — Benchmark Evaluation (PolyU dataset)
    Uses the paired PolyU dataset (gt/ + lq/)
    Displays: PSNR, SSIM, LPIPS, Runtime

Mode B — Custom Image Denoising
    User uploads any image
    Displays: Original, LAN, TAO, TTAD outputs, Runtime
    No-reference metrics: NIQE, BRISQUE
"""

import streamlit as st
import time
import os
import uuid
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ─── Import wrappers and metrics ────────────────────────────────────────────────
from wrappers.lan_wrapper import LANWrapper
from wrappers.tao_wrapper import TAOWrapper
from wrappers.ttad_wrapper import TTADWrapper
from metrics import compute_metrics
from metrics.no_reference import compute_no_reference_metrics


# ─── Directories ────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_BASE = Path("outputs")
OUTPUT_DIRS = {
    "LAN":  OUTPUT_BASE / "LAN",
    "TAO":  OUTPUT_BASE / "TAO",
    "TTAD": OUTPUT_BASE / "TTAD",
}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# PolyU dataset location
POLYU_DIR = Path("models") / "LAN" / "polyu"

SUPPORTED_FORMATS = ("png", "jpg", "jpeg")

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTA Restoration · Thesis Demo",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #fafafa;
  --surface:   #ffffff;
  --surface-2: #f4f4f5;
  --border:    rgba(0,0,0,.08);
  --border-md: rgba(0,0,0,.12);
  --text:      #0f0f10;
  --text-2:    #52525b;
  --text-3:    #a1a1aa;
  --blue:      #2563eb;
  --blue-bg:   #eff6ff;
  --blue-bd:   #bfdbfe;
  --green:     #16a34a;
  --green-bg:  #f0fdf4;
  --green-bd:  #bbf7d0;
  --amber:     #b45309;
  --amber-bg:  #fffbeb;
  --amber-bd:  #fde68a;
  --red:       #dc2626;
  --radius-sm: 6px;
  --radius:    10px;
  --radius-lg: 14px;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.05);
  --shadow:    0 2px 8px rgba(0,0,0,.07);
}

html, body, [class*="css"]  { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.block-container            { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 1200px; }
.stApp                      { background: #fafafa !important; }
[data-testid="stAppViewContainer"] { background: #fafafa !important; }
[data-testid="stHeader"]    { background: transparent !important; }
hr                          { border-color: rgba(0,0,0,.08) !important; }

.stFileUploader > div > div {
  background: #ffffff !important;
  border: 1.5px dashed rgba(0,0,0,.12) !important;
  border-radius: 10px !important;
  transition: border-color .15s !important;
}
.stFileUploader > div > div:hover { border-color: #2563eb !important; }

.stButton > button {
  background: #2563eb !important;
  color: #fff !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 500 !important;
  font-size: .9rem !important;
  padding: .55rem 1.4rem !important;
  letter-spacing: .01em !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.05) !important;
  transition: opacity .15s, transform .1s !important;
}
.stButton > button:hover  { opacity: .88 !important; }
.stButton > button:active { transform: scale(.98) !important; }
.stButton > button:disabled { background: rgba(0,0,0,.12) !important; color: #a1a1aa !important; }

.stTabs [data-baseweb="tab-list"] {
  gap: .5rem;
  background: transparent !important;
  border-bottom: 1px solid rgba(0,0,0,.08) !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-size: .85rem;
  font-weight: 500;
  color: #52525b;
  padding: .6rem 1.2rem;
  border-radius: 6px 6px 0 0 !important;
  transition: color .15s, background .15s;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: #2563eb !important;
  background: #eff6ff !important;
  border-bottom: 2px solid #2563eb !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #2563eb !important;
  background: #f4f4f5 !important;
}

.stSpinner > div { color: #2563eb !important; }

.label {
  font-size: .7rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: #a1a1aa;
  margin-bottom: .35rem;
}
.section-rule {
  display: flex;
  align-items: center;
  gap: .75rem;
  margin: 2.5rem 0 1.25rem;
}
.section-rule span {
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: #a1a1aa;
  white-space: nowrap;
}
.section-rule::after {
  content: "";
  flex: 1;
  height: 1px;
  background: rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_uploaded_file(uploaded_file) -> Path:
    ext = Path(uploaded_file.name).suffix or ".png"
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def download_link(img: Image.Image, filename: str, label: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;display:block;">'
        f'<button style="width:100%;background:transparent;border:1px solid rgba(0,0,0,.12);'
        f'border-radius:6px;padding:.38rem .8rem;font-size:.78rem;font-weight:500;'
        f'color:#52525b;cursor:pointer;transition:background .15s;">'
        f'↓ {label}</button></a>'
    )


def fmt_metric(val, fmt, fallback="—"):
    return fallback if val is None else fmt.format(val)


def rank_methods(results: Dict[str, dict]) -> str:
    has_metrics = any(v.get("psnr") is not None for v in results.values())
    if not has_metrics:
        return min(results, key=lambda k: results[k].get("runtime", float("inf")))

    def key(item):
        d = item[1]
        return (
            -(d["psnr"]  if d.get("psnr")  is not None else 0),
            -(d["ssim"]  if d.get("ssim")  is not None else 0),
             (d["lpips"] if d.get("lpips") is not None else float("inf")),
             d.get("runtime", float("inf")),
        )
    return sorted(results.items(), key=key)[0][0]


# ─── Progress HTML helpers (all literal colours — no CSS vars) ──────────────────

def _step_html(name: str, state: str) -> str:
    if state == "done":
        icon, color, icon_bg, icon_bd, weight, pill_label = (
            "✓", "#16a34a", "#f0fdf4", "#bbf7d0", "500", "Done"
        )
    elif state == "running":
        icon, color, icon_bg, icon_bd, weight, pill_label = (
            "…", "#2563eb", "#eff6ff", "#bfdbfe", "600", "Running"
        )
    else:
        icon, color, icon_bg, icon_bd, weight, pill_label = (
            "·", "#a1a1aa", "#f4f4f5", "rgba(0,0,0,.08)", "400", "Queued"
        )

    return f"""
    <div style="display:flex;align-items:center;gap:.85rem;padding:.6rem .9rem;
        background:#ffffff;border:1px solid rgba(0,0,0,.08);
        border-radius:6px;margin-bottom:.4rem;">
      <span style="width:22px;height:22px;border-radius:50%;background:{icon_bg};
          border:1px solid {icon_bd};display:flex;align-items:center;justify-content:center;
          font-size:.78rem;font-weight:600;color:{color};flex-shrink:0;">{icon}</span>
      <span style="font-size:.86rem;font-weight:{weight};color:#0f0f10;flex:1;">{name}</span>
      <span style="font-size:.72rem;font-weight:500;color:{color};background:{icon_bg};
          border:1px solid {icon_bd};border-radius:999px;padding:.15rem .6rem;">{pill_label}</span>
    </div>
    """


def render_progress(placeholder, model_names: list, current_index: int, done: bool = False) -> None:
    steps_html = ""
    for j, n in enumerate(model_names):
        if done or j < current_index:
            state = "done"
        elif j == current_index:
            state = "running"
        else:
            state = "queued"
        steps_html += _step_html(n, state)

    if done:
        pct        = 100
        bar_color  = "#16a34a"
        bar_bg     = "#f0fdf4"
        border_col = "#bbf7d0"
        header_txt = "All models complete"
        header_col = "#16a34a"
        count_txt  = f"{len(model_names)}/{len(model_names)}"
    else:
        pct        = int(current_index / len(model_names) * 100)
        bar_color  = "#2563eb"
        bar_bg     = "#f4f4f5"
        border_col = "rgba(0,0,0,.08)"
        header_txt = "Inference in progress"
        header_col = "#52525b"
        count_txt  = f"{current_index}/{len(model_names)}"

    placeholder.markdown(f"""
    <div style="background:#ffffff;border:1px solid {border_col};
        border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.9rem;">
        <span style="font-size:.8rem;font-weight:500;color:{header_col};">{header_txt}</span>
        <span style="font-size:.75rem;font-family:'JetBrains Mono',monospace;color:#a1a1aa;">
          {count_txt}
        </span>
      </div>
      <div style="height:3px;background:{bar_bg};border-radius:999px;margin-bottom:1rem;overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:{bar_color};border-radius:999px;transition:width .3s;"></div>
      </div>
      {steps_html}
    </div>
    """, unsafe_allow_html=True)


def get_polyu_images() -> List[Tuple[str, Path, Path]]:
    """Return list of (stem, gt_path, lq_path) for all paired PolyU images."""
    gt_dir = POLYU_DIR / "gt"
    lq_dir = POLYU_DIR / "lq"
    if not gt_dir.exists() or not lq_dir.exists():
        return []

    images = []
    for gt_f in sorted(gt_dir.iterdir()):
        if gt_f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            lq_f = lq_dir / gt_f.name
            if lq_f.exists():
                images.append((gt_f.stem, gt_f, lq_f))
    return images


@st.cache_resource
def get_wrappers():
    return {
        "LAN":  LANWrapper(),
        "TAO":  TAOWrapper(),
        "TTAD": TTADWrapper(),
    }


def run_inference_all(
    image_path: Path,
    progress_ph,
    model_names: List[str],
    wrappers: Dict[str, Any],
    gt_pil: Optional[Image.Image] = None,
    compute_no_ref: bool = False,
) -> Dict[str, dict]:
    """Run all models on a degraded image, returning results dict.

    Each result dict has keys: image, runtime, memory_usage,
    psnr, ssim, lpips (if gt_pil provided), niqe, brisque (if compute_no_ref).
    """
    results: Dict[str, dict] = {}

    for i, name in enumerate(model_names):
        render_progress(progress_ph, model_names, current_index=i)

        wrapper = wrappers[name]
        try:
            output = wrapper.run(str(image_path))
            restored_img = output["output_image"]
            runtime = output["runtime"]
            memory = output["memory_usage"]

            out_path = OUTPUT_DIRS[name] / f"{image_path.stem}.png"
            restored_img.save(out_path)

            metrics = {
                "runtime": runtime,
                "memory_usage": memory,
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "niqe": None,
                "brisque": None,
            }

            # Reference-based metrics (if GT available)
            if gt_pil is not None:
                gt_r = gt_pil.resize(restored_img.size, Image.LANCZOS)
                gt_np = np.array(gt_r).astype(np.uint8)
                res_np = np.array(restored_img).astype(np.uint8)
                try:
                    c = compute_metrics(res_np, gt_np)
                    metrics.update(psnr=c["psnr"], ssim=c["ssim"], lpips=c["lpips"])
                except Exception as e:
                    st.warning(f"Reference metrics failed for {name}: {e}")

            # No-reference metrics (always computed for custom upload)
            if compute_no_ref:
                res_np = np.array(restored_img).astype(np.uint8)
                try:
                    nr = compute_no_reference_metrics(res_np)
                    metrics.update(niqe=nr["niqe"], brisque=nr["brisque"])
                except Exception as e:
                    st.warning(f"No-reference metrics failed for {name}: {e}")

            results[name] = {"image": restored_img, **metrics}

        except Exception as e:
            degraded_pil = Image.open(image_path).convert("RGB")
            st.error(f"{name} inference failed: {e}")
            results[name] = {
                "image": Image.new("RGB", degraded_pil.size, (235, 235, 235)),
                "runtime": 0,
                "memory_usage": 0,
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "niqe": None,
                "brisque": None,
            }

    render_progress(progress_ph, model_names, current_index=len(model_names), done=True)
    return results


def render_metric_summary(results: Dict[str, dict], model_names: List[str]):
    """Render the 4 metric tiles (Best PSNR, SSIM, LPIPS, Fastest)."""
    has_metrics = any(results[n].get("psnr") is not None for n in model_names)
    if not has_metrics:
        return

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    best_psnr_n = max(model_names, key=lambda k: results[k]["psnr"] if results[k]["psnr"] is not None else -1)
    best_ssim_n = max(model_names, key=lambda k: results[k]["ssim"] if results[k]["ssim"] is not None else -1)
    best_lpip_n = min(model_names, key=lambda k: results[k]["lpips"] if results[k]["lpips"] is not None else float("inf"))
    fastest_n   = min(model_names, key=lambda k: results[k]["runtime"])

    def metric_tile(label, value, sub, highlight=False):
        border = "1px solid #bfdbfe" if highlight else "1px solid rgba(0,0,0,.08)"
        bg     = "#eff6ff"           if highlight else "#f4f4f5"
        sub_c  = "#3b82f6"           if highlight else "#a1a1aa"
        return f"""
        <div style="background:{bg};border:{border};border-radius:10px;padding:.9rem 1rem;">
          <p style="font-size:.68rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
             color:#a1a1aa;margin:0 0 .4rem;">{label}</p>
          <p style="font-size:1.4rem;font-weight:600;color:#0f0f10;margin:0 0 .15rem;
             font-family:'JetBrains Mono',monospace;line-height:1;">{value}</p>
          <p style="font-size:.72rem;color:{sub_c};margin:0;">{sub}</p>
        </div>
        """

    with m1:
        bp = results[best_psnr_n]["psnr"]
        st.markdown(metric_tile("Best PSNR", f"{bp:.2f} dB", f"↑ {best_psnr_n}", highlight=True), unsafe_allow_html=True)
    with m2:
        bs = results[best_ssim_n]["ssim"]
        st.markdown(metric_tile("Best SSIM", f"{bs:.3f}", f"↑ {best_ssim_n}"), unsafe_allow_html=True)
    with m3:
        bl = results[best_lpip_n]["lpips"]
        st.markdown(metric_tile("Best LPIPS", f"{bl:.3f}", "↓ lower is better"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_tile("Fastest", f"{results[fastest_n]['runtime']:.2f} s", f"↑ {fastest_n}"), unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)


def render_result_cards(results: Dict[str, dict], model_names: List[str], best_name: str):
    """Render the 3 output cards with images and metrics."""
    c1, c2, c3 = st.columns(3, gap="medium")

    for col, name in zip([c1, c2, c3], model_names):
        m       = results[name]
        is_best = name == best_name
        b64_img = pil_to_b64(m["image"])
        dl      = download_link(m["image"], f"{name}_restored.png", f"Save {name} output")

        if is_best:
            badge = '<span style="font-size:.68rem;font-weight:600;background:#fffbeb;border:1px solid #fde68a;color:#b45309;border-radius:999px;padding:.15rem .55rem;">★ Best</span>'
        else:
            badge = '<span style="font-size:.68rem;font-weight:600;background:#f0fdf4;border:1px solid #bbf7d0;color:#16a34a;border-radius:999px;padding:.15rem .55rem;">Done</span>'

        # Metrics row
        metrics_row = ""
        if m.get("psnr") is not None:
            metrics_row += f'<span style="font-size:.75rem;color:#52525b;font-family:\'JetBrains Mono\',monospace;">PSNR <b style="color:#0f0f10">{m["psnr"]:.2f}</b></span>'
        if m.get("ssim") is not None:
            metrics_row += f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#52525b;font-family:\'JetBrains Mono\',monospace;">SSIM <b style="color:#0f0f10">{m["ssim"]:.3f}</b></span>'
        if m.get("lpips") is not None:
            metrics_row += f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#52525b;font-family:\'JetBrains Mono\',monospace;">LPIPS <b style="color:#0f0f10">{m["lpips"]:.3f}</b></span>'

        # No-reference row
        nr_row = ""
        if m.get("niqe") is not None:
            nr_row += f'<span style="font-size:.75rem;color:#52525b;font-family:\'JetBrains Mono\',monospace;">NIQE <b style="color:#0f0f10">{m["niqe"]:.2f}</b></span>'
        if m.get("brisque") is not None:
            nr_row += f'&nbsp;&nbsp;<span style="font-size:.75rem;color:#52525b;font-family:\'JetBrains Mono\',monospace;">BRISQUE <b style="color:#0f0f10">{m["brisque"]:.2f}</b></span>'

        meta_row = (
            f'<span style="font-size:.72rem;color:#a1a1aa;">'
            f'{m["runtime"]:.2f} s · {m.get("memory_usage",0):.0f} MB</span>'
        )

        left_accent = "border-left:3px solid #2563eb;" if is_best else ""

        # Build bottom row content
        bottom_content = meta_row
        if nr_row:
            bottom_content = nr_row + " · " + meta_row if metrics_row else nr_row + " · " + meta_row

        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid rgba(0,0,0,.08);
                border-radius:14px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.05);{left_accent}">
              <div style="padding:.65rem .9rem;border-bottom:1px solid rgba(0,0,0,.08);
                  display:flex;align-items:center;justify-content:space-between;">
                <span style="font-size:.88rem;font-weight:600;color:#0f0f10;">{name}</span>
                {badge}
              </div>
              <div style="padding:.65rem .65rem .5rem;">
                <img src="data:image/png;base64,{b64_img}"
                  style="width:100%;border-radius:6px;display:block;" />
              </div>
              <div style="padding:.5rem .9rem .3rem;display:flex;flex-wrap:wrap;gap:.3rem .6rem;">
                {metrics_row if metrics_row else bottom_content}
              </div>
              <div style="padding:.15rem .9rem .3rem;">
                {meta_row if metrics_row and nr_row else ""}
              </div>
              <div style="padding:0 .7rem .75rem;">
                {dl}
              </div>
            </div>
            """, unsafe_allow_html=True)


def render_comparison_table(results: Dict[str, dict], model_names: List[str], best_name: str, show_nr: bool = False):
    """Render the full comparison table in an iframe."""
    import streamlit.components.v1 as components

    st.markdown('<div class="section-rule" style="margin-top:2rem;"><span>Full comparison</span></div>', unsafe_allow_html=True)

    HEAD_CELL = ("padding:.45rem 1rem;font-size:.67rem;font-weight:600;"
                 "letter-spacing:.07em;text-transform:uppercase;color:#a1a1aa;")

    rows_html = ""
    for name in model_names:
        m       = results[name]
        is_best = name == best_name
        row_bg  = "#f0f7ff" if is_best else "#ffffff"
        row_bl  = "2.5px solid #2563eb" if is_best else "2.5px solid transparent"

        best_badge = (
            '<span style="font-size:.65rem;font-weight:600;background:#fffbeb;'
            'border:1px solid #fde68a;color:#b45309;border-radius:999px;'
            'padding:.1rem .45rem;margin-left:.4rem;">best</span>'
            if is_best else ""
        )

        psnr_val  = fmt_metric(m["psnr"],  "{:.2f} dB")
        ssim_val  = fmt_metric(m["ssim"],  "{:.3f}")
        lpips_val = fmt_metric(m["lpips"], "{:.3f}")
        niqe_val  = fmt_metric(m["niqe"],  "{:.2f}")
        brisque_val = fmt_metric(m["brisque"], "{:.2f}")

        psnr_color  = "#15803d" if (m.get("psnr") and m["psnr"] >= 30) else "#0f0f10"
        psnr_weight = "500"     if (m.get("psnr") and m["psnr"] >= 30) else "400"
        lpips_color = "#b45309" if m.get("lpips") else "#0f0f10"

        CELL = ("padding:.6rem 1rem;font-size:.82rem;"
                "font-family:'JetBrains Mono',monospace;"
                "border-bottom:1px solid rgba(0,0,0,.08);")

        # Build columns based on mode
        if show_nr:
            cols = "2fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr"
            rows_html += f"""
            <div style="display:grid;grid-template-columns:{cols};
                        align-items:center;background:{row_bg};border-left:{row_bl};">
              <div style="padding:.65rem 1rem;font-size:.84rem;font-weight:500;color:#0f0f10;
                          border-bottom:1px solid rgba(0,0,0,.08);display:flex;align-items:center;">
                {name}{best_badge}
              </div>
              <div style="{CELL}color:#52525b;">{m['runtime']:.2f}s</div>
              <div style="{CELL}color:{psnr_color};font-weight:{psnr_weight};">{psnr_val}</div>
              <div style="{CELL}color:#0f0f10;">{ssim_val}</div>
              <div style="{CELL}color:{lpips_color};">{lpips_val}</div>
              <div style="{CELL}color:#52525b;">{niqe_val}</div>
              <div style="{CELL}color:#52525b;">{brisque_val}</div>
              <div style="{CELL}color:#52525b;">{m.get('memory_usage', 0):.0f} MB</div>
            </div>
            """
        else:
            cols = "2fr 1fr 1fr 1fr 1fr 1fr"
            rows_html += f"""
            <div style="display:grid;grid-template-columns:{cols};
                        align-items:center;background:{row_bg};border-left:{row_bl};">
              <div style="padding:.65rem 1rem;font-size:.84rem;font-weight:500;color:#0f0f10;
                          border-bottom:1px solid rgba(0,0,0,.08);display:flex;align-items:center;">
                {name}{best_badge}
              </div>
              <div style="{CELL}color:{psnr_color};font-weight:{psnr_weight};">{psnr_val}</div>
              <div style="{CELL}color:#0f0f10;">{ssim_val}</div>
              <div style="{CELL}color:{lpips_color};">{lpips_val}</div>
              <div style="{CELL}color:#52525b;">{m['runtime']:.2f} s</div>
              <div style="{CELL}color:#52525b;">{m.get('memory_usage', 0):.0f} MB</div>
            </div>
            """

    if show_nr:
        hdr = f"""<div style="{HEAD_CELL}">Method</div>
<div style="{HEAD_CELL}">Runtime</div>
<div style="{HEAD_CELL}">PSNR ↑</div>
<div style="{HEAD_CELL}">SSIM ↑</div>
<div style="{HEAD_CELL}">LPIPS ↓</div>
<div style="{HEAD_CELL}">NIQE ↓</div>
<div style="{HEAD_CELL}">BRISQUE ↓</div>
<div style="{HEAD_CELL}">Memory</div>"""
        grid_cols = "2fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr"
        height = 210
    else:
        hdr = f"""<div style="{HEAD_CELL}">Method</div>
<div style="{HEAD_CELL}">PSNR ↑</div>
<div style="{HEAD_CELL}">SSIM ↑</div>
<div style="{HEAD_CELL}">LPIPS ↓</div>
<div style="{HEAD_CELL}">Runtime</div>
<div style="{HEAD_CELL}">Memory</div>"""
        grid_cols = "2fr 1fr 1fr 1fr 1fr 1fr"
        height = 185

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      * {{ box-sizing:border-box; margin:0; padding:0; }}
      body {{ background:transparent; font-family:'Inter',system-ui,sans-serif; }}
    </style>
    </head>
    <body>
    <div style="background:#ffffff;border:1px solid rgba(0,0,0,.08);
                border-radius:14px;overflow:hidden;
                box-shadow:0 1px 2px rgba(0,0,0,.05);">
      <div style="display:grid;grid-template-columns:{grid_cols};
                  align-items:center;background:#f4f4f5;
                  border-bottom:1px solid rgba(0,0,0,.08);">
        {hdr}
      </div>
      {rows_html}
    </div>
    </body>
    </html>
    """

    components.html(table_html, height=height, scrolling=False)


def render_insight_block(results: Dict[str, dict], model_names: List[str], best_name: str, best_metrics: dict):
    """Render analysis insight block when reference metrics are available."""
    if all(results[n].get("psnr") is not None for n in model_names):
        ref = results["LAN"]["psnr"]
        gains = {
            n: results[n]["psnr"] - ref
            for n in ["TAO", "TTAD"]
            if results[n].get("psnr") is not None
        }
        if gains:
            gain_parts = " and ".join(
                f"<strong>{n}</strong> by <strong>+{g:.1f} dB</strong>"
                for n, g in gains.items()
            )
            st.markdown(f"""
            <div style="background:#f4f4f5;border:1px solid rgba(0,0,0,.08);
                border-left:3px solid #2563eb;border-radius:10px;
                padding:1rem 1.25rem;margin-top:1.5rem;">
              <p style="font-size:.72rem;font-weight:600;letter-spacing:.07em;
                 text-transform:uppercase;color:#2563eb;margin:0 0 .4rem;">Analysis insight</p>
              <p style="font-size:.87rem;color:#52525b;line-height:1.7;margin:0;">
                <strong style="color:#0f0f10">{best_name}</strong> achieves the highest
                perceptual quality (PSNR&nbsp;{best_metrics['psnr']:.2f}&nbsp;dB,
                SSIM&nbsp;{best_metrics['ssim']:.3f}). Relative to the LAN baseline,
                TTA strategies improve {gain_parts}, demonstrating meaningful adaptation gains.
              </p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem;">
  <p style="font-size:.72rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
     color:#a1a1aa;margin:0 0 .6rem;">M.Sc. Computer Vision · Thesis</p>
  <h1 style="font-size:clamp(1.55rem,3vw,2.1rem);font-weight:600;color:#0f0f10;
     line-height:1.25;margin:0 0 .75rem;letter-spacing:-.02em;">
    Test-time adaptation<br>image restoration benchmark
  </h1>
  <p style="font-size:.93rem;color:#52525b;max-width:600px;line-height:1.65;margin:0 0 1.4rem;">
    Compare LAN, TAO, and TTAD on blind image restoration. Choose between
    the <strong>PolyU benchmark</strong> for full-reference evaluation or
    <strong>custom upload</strong> for real-world denoising.
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:.5rem;">
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">3 methods</span>
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">PSNR · SSIM · LPIPS</span>
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">NIQE · BRISQUE</span>
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">Runtime & memory</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs for dual mode ─────────────────────────────────────────────────────────
tab_benchmark, tab_custom = st.tabs([
    "📊 Benchmark Evaluation — PolyU",
    "🖼️ Custom Image Denoising",
])

wrappers = get_wrappers()
model_names = ["LAN", "TAO", "TTAD"]


# ═══════════════════════════════════════════════════════════════════════════════
# MODE A — BENCHMARK EVALUATION (PolyU)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    polyu_images = get_polyu_images()

    if not polyu_images:
        st.warning(
            "PolyU dataset not found. Expected structure: "
            "`models/LAN/polyu/gt/` and `models/LAN/polyu/lq/` with paired PNG files."
        )
        st.info("Make sure the PolyU dataset is placed in the correct directory.")
    else:
        st.markdown(
            '<p style="font-size:.78rem;color:#52525b;margin-bottom:.75rem;">'
            f'Dataset: <strong>PolyU</strong> · {len(polyu_images)} image pairs available'
            '</p>',
            unsafe_allow_html=True,
        )

        # Image selector
        col_select, col_preview = st.columns([1, 1], gap="large")

        with col_select:
            st.markdown('<p class="label">Select image</p>', unsafe_allow_html=True)
            image_names = [f"{stem}" for stem, _, _ in polyu_images]
            selected_idx = st.selectbox(
                "image_select",
                range(len(image_names)),
                format_func=lambda i: image_names[i],
                label_visibility="collapsed",
            )

            selected_stem, gt_path, lq_path = polyu_images[selected_idx]

            # Show preview of both GT and LQ
            gt_preview = Image.open(gt_path).convert("RGB")
            lq_preview = Image.open(lq_path).convert("RGB")
            w, h = gt_preview.size

            st.markdown(
                f'<p style="font-size:.72rem;color:#a1a1aa;margin-top:.5rem;">'
                f'{w} × {h} px · {selected_stem}</p>',
                unsafe_allow_html=True,
            )

        with col_preview:
            st.markdown('<p class="label">Preview</p>', unsafe_allow_html=True)
            preview_col1, preview_col2 = st.columns(2, gap="small")

            with preview_col1:
                gt_b64 = pil_to_b64(gt_preview)
                st.markdown(
                    f'<div style="text-align:center;"><p style="font-size:.68rem;color:#a1a1aa;margin:0 0 .3rem;">Ground Truth</p>'
                    f'<img src="data:image/png;base64,{gt_b64}" style="width:100%;border-radius:6px;" /></div>',
                    unsafe_allow_html=True,
                )

            with preview_col2:
                lq_b64 = pil_to_b64(lq_preview)
                st.markdown(
                    f'<div style="text-align:center;"><p style="font-size:.68rem;color:#a1a1aa;margin:0 0 .3rem;">Degraded</p>'
                    f'<img src="data:image/png;base64,{lq_b64}" style="width:100%;border-radius:6px;" /></div>',
                    unsafe_allow_html=True,
                )

        # Run button
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        btn_col, _ = st.columns([1, 4])
        with btn_col:
            run_bench = st.button(
                "Run benchmark evaluation",
                use_container_width=True,
                key="run_benchmark",
            )

        # ── Processing + Results ─────────────────────────────────────────
        if run_bench:
            st.markdown('<div class="section-rule"><span>Processing</span></div>', unsafe_allow_html=True)
            progress_ph = st.empty()

            gt_pil = Image.open(gt_path).convert("RGB")
            results = run_inference_all(
                image_path=lq_path,
                progress_ph=progress_ph,
                model_names=model_names,
                wrappers=wrappers,
                gt_pil=gt_pil,
                compute_no_ref=False,
            )

            # Ranking
            best_name = rank_methods(results)
            best_metrics = results[best_name]

            st.markdown('<div class="section-rule"><span>Results</span></div>', unsafe_allow_html=True)

            # Best method banner
            if best_metrics.get("psnr") is not None:
                rank_detail = (
                    f"PSNR {best_metrics['psnr']:.2f} dB · "
                    f"SSIM {best_metrics['ssim']:.3f} · "
                    f"LPIPS {best_metrics['lpips']:.3f}"
                )
            else:
                rank_detail = f"Fastest runtime — {best_metrics.get('runtime', 0):.2f} s"

            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;
                border-radius:10px;padding:1rem 1.3rem;margin-bottom:1.5rem;
                display:flex;align-items:center;gap:1rem;">
              <div style="width:36px;height:36px;background:#2563eb;border-radius:6px;
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1.1rem;">🏆</div>
              <div>
                <p style="font-size:.7rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
                   color:#1d4ed8;margin:0 0 .15rem;">Best method</p>
                <p style="font-size:1.1rem;font-weight:600;color:#0f0f10;margin:0 0 .1rem;">{best_name}</p>
                <p style="font-size:.78rem;color:#3b82f6;margin:0;font-family:'JetBrains Mono',monospace;">{rank_detail}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Metric summary
            render_metric_summary(results, model_names)

            # Output cards
            render_result_cards(results, model_names, best_name)

            # Comparison table
            render_comparison_table(results, model_names, best_name, show_nr=False)

            # Insight block
            render_insight_block(results, model_names, best_name, best_metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE B — CUSTOM IMAGE DENOISING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown(
        '<p style="font-size:.82rem;color:#52525b;margin-bottom:1rem;">'
        'Upload any image to denoise with all three methods. Since no ground truth is available, '
        'quality is assessed via <strong>no-reference metrics</strong> (NIQE, BRISQUE).'
        '</p>',
        unsafe_allow_html=True,
    )

    up_col1, prev_col = st.columns([1, 1], gap="large")

    with up_col1:
        st.markdown('<p class="label">Upload image</p>', unsafe_allow_html=True)
        custom_file = st.file_uploader(
            "custom_image",
            type=list(SUPPORTED_FORMATS),
            label_visibility="collapsed",
            key="custom_upload",
        )

        if custom_file is not None:
            st.session_state["custom_image"] = Image.open(custom_file).convert("RGB")

    with prev_col:
        custom_img = st.session_state.get("custom_image", None)
        if custom_img is not None:
            w, h = custom_img.size
            b64 = pil_to_b64(custom_img)
            st.markdown(
                f'<p class="label">Original · {w} × {h}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;border-radius:10px;border:1px solid rgba(0,0,0,.08);" />',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div style="background:#f4f4f5;border:1px solid rgba(0,0,0,.08);
                border-radius:10px;min-height:120px;display:flex;
                align-items:center;justify-content:center;">
              <span style="font-size:.82rem;color:#a1a1aa;">No image selected</span>
            </div>
            """, unsafe_allow_html=True)

    # Run button
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    btn_col2, hint_col2 = st.columns([1, 4])
    with btn_col2:
        run_custom = st.button(
            "Run denoising",
            disabled=(custom_file is None),
            use_container_width=True,
            key="run_custom",
        )
    with hint_col2:
        if custom_file is None:
            st.markdown(
                '<p style="font-size:.8rem;color:#a1a1aa;margin:.55rem 0 0;">Upload an image to begin.</p>',
                unsafe_allow_html=True,
            )

    # ── Processing + Results ─────────────────────────────────────────────
    if run_custom and custom_file is not None:
        saved_path = save_uploaded_file(custom_file)

        st.markdown('<div class="section-rule"><span>Processing</span></div>', unsafe_allow_html=True)
        progress_ph = st.empty()

        results = run_inference_all(
            image_path=saved_path,
            progress_ph=progress_ph,
            model_names=model_names,
            wrappers=wrappers,
            gt_pil=None,
            compute_no_ref=True,
        )

        # Find fastest (no reference metrics for ranking)
        fastest_name = min(model_names, key=lambda k: results[k]["runtime"])
        best_name = fastest_name
        best_metrics = results[fastest_name]

        st.markdown('<div class="section-rule"><span>Results</span></div>', unsafe_allow_html=True)

        # Show original image side by side with outputs
        st.markdown('<p class="label" style="margin-bottom:.5rem;">Output comparison</p>', unsafe_allow_html=True)
        render_result_cards(results, model_names, best_name)

        # Comparison table with NR metrics
        render_comparison_table(results, model_names, best_name, show_nr=True)

        # Quality summary
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-rule" style="margin-top:1rem;"><span>No-reference quality scores</span></div>',
            unsafe_allow_html=True,
        )

        # NR metric tiles
        nr_metrics = {}
        for name in model_names:
            m = results[name]
            nr_metrics[name] = {
                "niqe": m.get("niqe"),
                "brisque": m.get("brisque"),
                "runtime": m["runtime"],
            }

        best_niqe_n = min(
            model_names,
            key=lambda k: nr_metrics[k]["niqe"] if nr_metrics[k]["niqe"] is not None else float("inf"),
        )
        best_brisque_n = min(
            model_names,
            key=lambda k: nr_metrics[k]["brisque"] if nr_metrics[k]["brisque"] is not None else float("inf"),
        )

        nm1, nm2, nm3 = st.columns(3, gap="medium")

        def nr_tile(label, best_val, best_name_val, description, highlight):
            border = "1px solid #bfdbfe" if highlight else "1px solid rgba(0,0,0,.08)"
            bg = "#eff6ff" if highlight else "#f4f4f5"
            sub_c = "#3b82f6" if highlight else "#a1a1aa"
            val_str = f"{best_val:.2f}" if best_val is not None else "—"
            return f"""
            <div style="background:{bg};border:{border};border-radius:10px;padding:.9rem 1rem;">
              <p style="font-size:.68rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
                 color:#a1a1aa;margin:0 0 .4rem;">{label}</p>
              <p style="font-size:1.4rem;font-weight:600;color:#0f0f10;margin:0 0 .15rem;
                 font-family:'JetBrains Mono',monospace;line-height:1;">{val_str}</p>
              <p style="font-size:.72rem;color:{sub_c};margin:0;">↓ {description} — {best_name_val}</p>
            </div>
            """

        with nm1:
            bv = nr_metrics[best_niqe_n]["niqe"]
            st.markdown(nr_tile("Best NIQE", bv, best_niqe_n, "lower is better", True), unsafe_allow_html=True)
        with nm2:
            bv = nr_metrics[best_brisque_n]["brisque"]
            st.markdown(nr_tile("Best BRISQUE", bv, best_brisque_n, "lower is better", False), unsafe_allow_html=True)
        with nm3:
            ft_val = f"{results[fastest_name]['runtime']:.2f} s"
            st.markdown(f"""
            <div style="background:#f4f4f5;border:1px solid rgba(0,0,0,.08);border-radius:10px;padding:.9rem 1rem;">
              <p style="font-size:.68rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
                 color:#a1a1aa;margin:0 0 .4rem;">Fastest</p>
              <p style="font-size:1.4rem;font-weight:600;color:#0f0f10;margin:0 0 .15rem;
                 font-family:'JetBrains Mono',monospace;line-height:1;">{ft_val}</p>
              <p style="font-size:.72rem;color:#a1a1aa;margin:0;">↓ seconds — {fastest_name}</p>
            </div>
            """, unsafe_allow_html=True)

        # Insight for custom mode
        st.markdown(f"""
        <div style="background:#f4f4f5;border:1px solid rgba(0,0,0,.08);
            border-left:3px solid #2563eb;border-radius:10px;
            padding:1rem 1.25rem;margin-top:1.5rem;">
          <p style="font-size:.72rem;font-weight:600;letter-spacing:.07em;
             text-transform:uppercase;color:#2563eb;margin:0 0 .4rem;">No-reference quality assessment</p>
          <p style="font-size:.87rem;color:#52525b;line-height:1.7;margin:0;">
            Since no ground truth is available, perceptual quality is estimated using
            <strong style="color:#0f0f10">NIQE</strong> (Natural Image Quality Evaluator) and
            <strong style="color:#0f0f10">BRISQUE</strong> (Blind/Referenceless Image Spatial
            Quality Evaluator). Lower scores indicate better perceived quality.
            <strong style="color:#0f0f10">{best_niqe_n}</strong> achieves the best NIQE
            ({nr_metrics[best_niqe_n]["niqe"]:.2f}) while
            <strong style="color:#0f0f10">{best_brisque_n}</strong> leads in BRISQUE
            ({nr_metrics[best_brisque_n]["brisque"]:.2f}).
          </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:4rem;padding-top:2rem;border-top:1px solid rgba(0,0,0,.08);">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:2rem;margin-bottom:2rem;">
    <div>
      <p style="font-size:.75rem;font-weight:600;color:#0f0f10;margin:0 0 .5rem;">About</p>
      <p style="font-size:.78rem;color:#a1a1aa;line-height:1.65;margin:0;">
        Interactive benchmark comparing test-time adaptation methods for blind image restoration.
        Built as part of a Master's thesis in Computer Vision.
      </p>
    </div>
    <div>
      <p style="font-size:.75rem;font-weight:600;color:#0f0f10;margin:0 0 .5rem;">Methods</p>
      <p style="font-size:.78rem;color:#a1a1aa;line-height:1.8;margin:0;">
        <strong style="color:#52525b">LAN</strong> — Lightweight adaptive network<br>
        <strong style="color:#52525b">TAO</strong> — Test-time adaptation optimization<br>
        <strong style="color:#52525b">TTAD</strong> — TTA with diffusion
      </p>
    </div>
    <div>
      <p style="font-size:.75rem;font-weight:600;color:#0f0f10;margin:0 0 .5rem;">Metrics</p>
      <p style="font-size:.78rem;color:#a1a1aa;line-height:1.8;margin:0;">
        <strong style="color:#52525b">PSNR</strong> — Peak signal-to-noise ratio ↑<br>
        <strong style="color:#52525b">SSIM</strong> — Structural similarity index ↑<br>
        <strong style="color:#52525b">LPIPS</strong> — Learned perceptual similarity ↓<br>
        <strong style="color:#52525b">NIQE</strong> — No-reference quality ↓<br>
        <strong style="color:#52525b">BRISQUE</strong> — No-reference quality ↓
      </p>
    </div>
  </div>
  <p style="font-size:.72rem;color:#a1a1aa;text-align:center;margin:0;">
    Thesis research · Test-time adaptation image restoration · Built with Streamlit
  </p>
</div>
""", unsafe_allow_html=True)