import streamlit as st
import time
import os
import uuid
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional

# ─── Import wrappers and metrics ────────────────────────────────────────────────
from wrappers.lan_wrapper import LANWrapper
from wrappers.tao_wrapper import TAOWrapper
from wrappers.ttad_wrapper import TTADWrapper
from metrics import compute_metrics


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

SUPPORTED_FORMATS = ("png", "jpg", "jpeg")

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTA Restoration · Benchmark",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
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

/* ── Reset chrome ── */
html, body, [class*="css"]  { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.block-container            { padding-top: 2.5rem !important; padding-bottom: 4rem !important; max-width: 1200px; }
.stApp                      { background: var(--bg) !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"]    { background: transparent !important; }
hr                          { border-color: var(--border) !important; }

/* ── File uploader ── */
.stFileUploader > div > div {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border-md) !important;
  border-radius: var(--radius) !important;
  transition: border-color .15s !important;
}
.stFileUploader > div > div:hover { border-color: var(--blue) !important; }

/* ── Button ── */
.stButton > button {
  background: var(--blue) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
  font-size: .9rem !important;
  padding: .55rem 1.4rem !important;
  letter-spacing: .01em !important;
  box-shadow: var(--shadow-sm) !important;
  transition: opacity .15s, transform .1s !important;
}
.stButton > button:hover  { opacity: .88 !important; }
.stButton > button:active { transform: scale(.98) !important; }
.stButton > button:disabled { background: var(--border-md) !important; color: var(--text-3) !important; }

/* ── Spinner ── */
.stSpinner > div { color: var(--blue) !important; }

/* ── Utility classes ── */
.label {
  font-size: .7rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--text-3);
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
  color: var(--text-3);
  white-space: nowrap;
}
.section-rule::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--border);
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────────
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
        f'<button style="width:100%;background:transparent;border:1px solid var(--border-md);'
        f'border-radius:var(--radius-sm);padding:.38rem .8rem;font-size:.78rem;font-weight:500;'
        f'color:var(--text-2);cursor:pointer;transition:background .15s;">'
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


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:2.5rem;">
  <p style="font-size:.72rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
     color:#a1a1aa;margin:0 0 .6rem;">M.Sc. Computer Vision · Thesis</p>
  <h1 style="font-size:clamp(1.55rem,3vw,2.1rem);font-weight:600;color:#0f0f10;
     line-height:1.25;margin:0 0 .75rem;letter-spacing:-.02em;">
    Test-time adaptation<br>image restoration benchmark
  </h1>
  <p style="font-size:.93rem;color:#52525b;max-width:560px;line-height:1.65;margin:0 0 1.4rem;">
    Compare LAN, TAO, and TTAD on blind image restoration using PSNR, SSIM, and LPIPS.
    Upload a degraded image — optionally pair it with ground truth — then run all three methods simultaneously.
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:.5rem;">
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">3 methods</span>
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">PSNR · SSIM · LPIPS</span>
    <span style="background:#f4f4f5;border:1px solid rgba(0,0,0,.07);border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#52525b;">Runtime &amp; memory</span>
    <span style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:999px;
      padding:.25rem .75rem;font-size:.75rem;font-weight:500;color:#1d4ed8;">Real-time inference</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-rule"><span>Upload</span></div>', unsafe_allow_html=True)

up_col1, up_col2, prev_col = st.columns([1, 1, 1], gap="large")

with up_col1:
    st.markdown('<p class="label">Degraded input</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "degraded",
        type=list(SUPPORTED_FORMATS),
        label_visibility="collapsed",
        key="degraded_input",
    )

with up_col2:
    st.markdown('<p class="label">Ground truth <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:.8rem;color:#a1a1aa;">optional</span></p>', unsafe_allow_html=True)
    gt_file = st.file_uploader(
        "gt",
        type=list(SUPPORTED_FORMATS),
        label_visibility="collapsed",
        key="gt_input",
    )

with prev_col:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        w, h = image.size
        b64 = pil_to_b64(image)
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow-sm);">
          <div style="padding:.5rem .85rem;border-bottom:1px solid var(--border);
              display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:.75rem;font-weight:500;color:var(--text-2);">Preview</span>
            <span style="font-size:.7rem;color:var(--text-3);font-family:'JetBrains Mono',monospace;">
              {w}&thinsp;×&thinsp;{h}
            </span>
          </div>
          <div style="padding:.6rem;">
            <img src="data:image/png;base64,{b64}"
              style="width:100%;border-radius:var(--radius-sm);display:block;" />
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--surface-2);border:1px solid var(--border);
            border-radius:var(--radius);min-height:120px;display:flex;
            align-items:center;justify-content:center;">
          <span style="font-size:.82rem;color:var(--text-3);">No image selected</span>
        </div>
        """, unsafe_allow_html=True)

if gt_file is not None:
    st.session_state["gt_image"] = Image.open(gt_file).convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BUTTON
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

btn_col, hint_col = st.columns([1, 4])
with btn_col:
    run_clicked = st.button(
        "Run comparison",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )
with hint_col:
    if uploaded_file is None:
        st.markdown(
            '<p style="font-size:.8rem;color:var(--text-3);margin:.55rem 0 0;">Upload a degraded image to begin.</p>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING + RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if run_clicked and uploaded_file is not None:

    saved_path   = save_uploaded_file(uploaded_file)
    degraded_pil = Image.open(uploaded_file).convert("RGB")
    gt_pil       = st.session_state.get("gt_image", None)

    wrappers = {
        "LAN":  LANWrapper(),
        "TAO":  TAOWrapper(),
        "TTAD": TTADWrapper(),
    }
    results: Dict[str, dict] = {}
    model_names = ["LAN", "TAO", "TTAD"]

    # ── Processing section ───────────────────────────────────────────────────
    st.markdown('<div class="section-rule"><span>Processing</span></div>', unsafe_allow_html=True)
    progress_ph = st.empty()

    for i, name in enumerate(model_names):

        # Render honest per-step progress
        steps_html = ""
        for j, n in enumerate(model_names):
            if j < i:
                icon   = "✓"
                color  = "var(--green)"
                weight = "500"
                bg     = "var(--green-bg)"
                bd     = "var(--green-bd)"
                label  = "Done"
            elif j == i:
                icon   = "…"
                color  = "var(--blue)"
                weight = "600"
                bg     = "var(--blue-bg)"
                bd     = "var(--blue-bd)"
                label  = "Running"
            else:
                icon   = "·"
                color  = "var(--text-3)"
                weight = "400"
                bg     = "var(--surface-2)"
                bd     = "var(--border)"
                label  = "Queued"

            steps_html += f"""
            <div style="display:flex;align-items:center;gap:.85rem;padding:.6rem .9rem;
                background:var(--surface);border:1px solid var(--border);
                border-radius:var(--radius-sm);margin-bottom:.4rem;">
              <span style="width:22px;height:22px;border-radius:50%;background:{bg};
                  border:1px solid {bd};display:flex;align-items:center;justify-content:center;
                  font-size:.78rem;font-weight:600;color:{color};flex-shrink:0;">{icon}</span>
              <span style="font-size:.86rem;font-weight:{weight};color:var(--text);flex:1;">{n}</span>
              <span style="font-size:.72rem;font-weight:500;color:{color};background:{bg};
                  border:1px solid {bd};border-radius:999px;padding:.15rem .6rem;">{label}</span>
            </div>
            """

        pct = int(i / len(model_names) * 100)
        progress_ph.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius-lg);padding:1.2rem 1.4rem;margin-bottom:1rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.9rem;">
            <span style="font-size:.8rem;font-weight:500;color:var(--text-2);">
              Inference in progress
            </span>
            <span style="font-size:.75rem;font-family:'JetBrains Mono',monospace;color:var(--text-3);">
              {i}/{len(model_names)}
            </span>
          </div>
          <div style="height:3px;background:var(--surface-2);border-radius:999px;margin-bottom:1rem;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:var(--blue);border-radius:999px;transition:width .3s;"></div>
          </div>
          {steps_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Run wrapper ───────────────────────────────────────────────────
        wrapper = wrappers[name]
        try:
            output       = wrapper.run(str(saved_path))
            restored_img = output["output_image"]
            runtime      = output["runtime"]
            memory       = output["memory_usage"]

            out_path = OUTPUT_DIRS[name] / f"{saved_path.stem}.png"
            restored_img.save(out_path)

            metrics = {
                "runtime": runtime, "memory_usage": memory,
                "psnr": None, "ssim": None, "lpips": None,
            }
            if gt_pil is not None:
                gt_r    = gt_pil.resize(restored_img.size, Image.LANCZOS)
                gt_np   = np.array(gt_r).astype(np.uint8)
                res_np  = np.array(restored_img).astype(np.uint8)
                try:
                    c = compute_metrics(res_np, gt_np)
                    metrics.update(psnr=c["psnr"], ssim=c["ssim"], lpips=c["lpips"])
                except Exception as e:
                    st.warning(f"Metrics failed for {name}: {e}")

            results[name] = {"image": restored_img, **metrics}

        except Exception as e:
            st.error(f"{name} inference failed: {e}")
            results[name] = {
                "image": Image.new("RGB", degraded_pil.size, (235, 235, 235)),
                "runtime": 0, "memory_usage": 0,
                "psnr": None, "ssim": None, "lpips": None,
            }

    # ── Done state ────────────────────────────────────────────────────────
    done_steps = ""
    for name in model_names:
        done_steps += f"""
        <div style="display:flex;align-items:center;gap:.85rem;padding:.6rem .9rem;
            background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius-sm);margin-bottom:.4rem;">
          <span style="width:22px;height:22px;border-radius:50%;
              background:var(--green-bg);border:1px solid var(--green-bd);
              display:flex;align-items:center;justify-content:center;
              font-size:.78rem;font-weight:600;color:var(--green);flex-shrink:0;">✓</span>
          <span style="font-size:.86rem;font-weight:500;color:var(--text);flex:1;">{name}</span>
          <span style="font-size:.72rem;font-weight:500;color:var(--green);
              background:var(--green-bg);border:1px solid var(--green-bd);
              border-radius:999px;padding:.15rem .6rem;">Done</span>
        </div>
        """

    progress_ph.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--green-bd);
        border-radius:var(--radius-lg);padding:1.2rem 1.4rem;margin-bottom:1rem;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.9rem;">
        <span style="font-size:.8rem;font-weight:500;color:var(--green);">All models complete</span>
        <span style="font-size:.75rem;font-family:'JetBrains Mono',monospace;color:var(--text-3);">3/3</span>
      </div>
      <div style="height:3px;background:var(--green-bg);border-radius:999px;margin-bottom:1rem;overflow:hidden;">
        <div style="height:100%;width:100%;background:var(--green);border-radius:999px;"></div>
      </div>
      {done_steps}
    </div>
    """, unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════
    # RANKING
    # ═══════════════════════════════════════════════════════════════════════
    best_name    = rank_methods(results)
    best_metrics = results[best_name]

    st.markdown('<div class="section-rule"><span>Results</span></div>', unsafe_allow_html=True)

    # Best method callout
    if best_metrics.get("psnr") is not None:
        rank_detail = (
            f"PSNR {best_metrics['psnr']:.2f} dB · "
            f"SSIM {best_metrics['ssim']:.3f} · "
            f"LPIPS {best_metrics['lpips']:.3f}"
        )
    else:
        rank_detail = f"Fastest runtime — {best_metrics.get('runtime', 0):.2f} s"

    st.markdown(f"""
    <div style="background:var(--blue-bg);border:1px solid var(--blue-bd);
        border-radius:var(--radius);padding:1rem 1.3rem;margin-bottom:1.5rem;
        display:flex;align-items:center;gap:1rem;">
      <div style="width:36px;height:36px;background:#2563eb;border-radius:var(--radius-sm);
          display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:1.1rem;">🏆</div>
      <div>
        <p style="font-size:.7rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
           color:#1d4ed8;margin:0 0 .15rem;">Best method</p>
        <p style="font-size:1.1rem;font-weight:600;color:#0f0f10;margin:0 0 .1rem;">{best_name}</p>
        <p style="font-size:.78rem;color:#3b82f6;margin:0;font-family:'JetBrains Mono',monospace;">{rank_detail}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


    # ── Metric summary row ────────────────────────────────────────────────
    has_metrics = any(results[n].get("psnr") is not None for n in model_names)
    if has_metrics:
        m1, m2, m3, m4 = st.columns(4, gap="medium")

        best_psnr_n = max(model_names, key=lambda k: results[k]["psnr"] if results[k]["psnr"] is not None else -1)
        best_ssim_n = max(model_names, key=lambda k: results[k]["ssim"] if results[k]["ssim"] is not None else -1)
        best_lpip_n = min(model_names, key=lambda k: results[k]["lpips"] if results[k]["lpips"] is not None else float("inf"))
        fastest_n   = min(model_names, key=lambda k: results[k]["runtime"])

        def metric_tile(label, value, sub, highlight=False):
            border = "1px solid var(--blue-bd)" if highlight else "1px solid var(--border)"
            bg     = "var(--blue-bg)" if highlight else "var(--surface-2)"
            sub_c  = "#3b82f6" if highlight else "var(--text-3)"
            return f"""
            <div style="background:{bg};border:{border};border-radius:var(--radius);
                padding:.9rem 1rem;">
              <p style="font-size:.68rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
                 color:var(--text-3);margin:0 0 .4rem;">{label}</p>
              <p style="font-size:1.4rem;font-weight:600;color:var(--text);margin:0 0 .15rem;
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


    # ── Output cards ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    for col, name in zip([c1, c2, c3], model_names):
        m       = results[name]
        is_best = name == best_name
        b64_img = pil_to_b64(m["image"])
        dl      = download_link(m["image"], f"{name}_restored.png", f"Save {name} output")

        # Badge
        if is_best:
            badge = '<span style="font-size:.68rem;font-weight:600;background:var(--amber-bg);border:1px solid var(--amber-bd);color:var(--amber);border-radius:999px;padding:.15rem .55rem;">★ Best</span>'
        else:
            badge = '<span style="font-size:.68rem;font-weight:600;background:var(--green-bg);border:1px solid var(--green-bd);color:var(--green);border-radius:999px;padding:.15rem .55rem;">Done</span>'

        # Metrics row
        metrics_row = ""
        if m.get("psnr") is not None:
            metrics_row += f'<span style="font-size:.75rem;color:var(--text-2);font-family:\'JetBrains Mono\',monospace;">PSNR <b style="color:var(--text)">{m["psnr"]:.2f}</b></span>'
        if m.get("ssim") is not None:
            metrics_row += f'&nbsp;&nbsp;<span style="font-size:.75rem;color:var(--text-2);font-family:\'JetBrains Mono\',monospace;">SSIM <b style="color:var(--text)">{m["ssim"]:.3f}</b></span>'
        if m.get("lpips") is not None:
            metrics_row += f'&nbsp;&nbsp;<span style="font-size:.75rem;color:var(--text-2);font-family:\'JetBrains Mono\',monospace;">LPIPS <b style="color:var(--text)">{m["lpips"]:.3f}</b></span>'

        meta_row = (
            f'<span style="font-size:.72rem;color:var(--text-3);">'
            f'{m["runtime"]:.2f} s · {m.get("memory_usage",0):.0f} MB</span>'
        )

        left_accent = "border-left:3px solid var(--blue);" if is_best else ""

        with col:
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:var(--radius-lg);overflow:hidden;box-shadow:var(--shadow-sm);{left_accent}">
              <div style="padding:.65rem .9rem;border-bottom:1px solid var(--border);
                  display:flex;align-items:center;justify-content:space-between;">
                <span style="font-size:.88rem;font-weight:600;color:var(--text);">{name}</span>
                {badge}
              </div>
              <div style="padding:.65rem .65rem .5rem;">
                <img src="data:image/png;base64,{b64_img}"
                  style="width:100%;border-radius:var(--radius-sm);display:block;" />
              </div>
              <div style="padding:.5rem .9rem .3rem;display:flex;flex-wrap:wrap;gap:.3rem .6rem;">
                {metrics_row if metrics_row else meta_row}
              </div>
              <div style="padding:.15rem .7rem .7rem;">
                {meta_row if metrics_row else ""}
              </div>
              <div style="padding:0 .7rem .75rem;">
                {dl}
              </div>
            </div>
            """, unsafe_allow_html=True)


    # ── Comparison table ─────────────────────────────────────────────────
    st.markdown('<div class="section-rule" style="margin-top:2rem;"><span>Full comparison</span></div>', unsafe_allow_html=True)

    rows_html = ""
    for name in model_names:
        m       = results[name]
        is_best = name == best_name
        row_bg  = "background:#f0f7ff;" if is_best else ""
        bl      = "border-left:2.5px solid var(--blue);" if is_best else "border-left:2.5px solid transparent;"

        best_badge = (
            '&ensp;<span style="font-size:.65rem;font-weight:600;background:var(--amber-bg);'
            'border:1px solid var(--amber-bd);color:var(--amber);border-radius:999px;'
            'padding:.1rem .45rem;">best</span>'
            if is_best else ""
        )

        psnr_val  = fmt_metric(m["psnr"],  "{:.2f} dB")
        ssim_val  = fmt_metric(m["ssim"],  "{:.3f}")
        lpips_val = fmt_metric(m["lpips"], "{:.3f}")

        psnr_c  = "color:#15803d;font-weight:500;" if m.get("psnr") and m["psnr"] >= 30 else ""
        lpips_c = "color:#b45309;" if m.get("lpips") else ""

        rows_html += f"""
        <tr style="{row_bg}{bl}">
          <td style="padding:.65rem 1rem;font-size:.84rem;font-weight:500;color:var(--text);
              border-bottom:1px solid var(--border);">{name}{best_badge}</td>
          <td style="padding:.65rem 1rem;font-size:.82rem;font-family:'JetBrains Mono',monospace;
              {psnr_c}border-bottom:1px solid var(--border);">{psnr_val}</td>
          <td style="padding:.65rem 1rem;font-size:.82rem;font-family:'JetBrains Mono',monospace;
              border-bottom:1px solid var(--border);">{ssim_val}</td>
          <td style="padding:.65rem 1rem;font-size:.82rem;font-family:'JetBrains Mono',monospace;
              {lpips_c}border-bottom:1px solid var(--border);">{lpips_val}</td>
          <td style="padding:.65rem 1rem;font-size:.82rem;font-family:'JetBrains Mono',monospace;
              color:var(--text-2);border-bottom:1px solid var(--border);">{m['runtime']:.2f} s</td>
          <td style="padding:.65rem 1rem;font-size:.82rem;font-family:'JetBrains Mono',monospace;
              color:var(--text-2);border-bottom:1px solid var(--border);">{m.get('memory_usage',0):.0f} MB</td>
        </tr>
        """

    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
        border-radius:var(--radius-lg);overflow:hidden;box-shadow:var(--shadow-sm);">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:var(--surface-2);">
            <th style="padding:.5rem 1rem;text-align:left;font-size:.68rem;font-weight:600;
                letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);
                border-bottom:1px solid var(--border);">Method</th>
            <th style="padding:.5rem 1rem;text-align:left;font-size:.68rem;font-weight:600;
                letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);
                border-bottom:1px solid var(--border);">PSNR ↑</th>
            <th style="padding:.5rem 1rem;text-align:left;font-size:.68rem;font-weight:600;
                letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);
                border-bottom:1px solid var(--border);">SSIM ↑</th>
            <th style="padding:.5rem 1rem;text-align:left;font-size:.68rem;font-weight:600;
                letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);
                border-bottom:1px solid var(--border);">LPIPS ↓</th>
            <th style="padding:.5rem 1rem;text-align:left;font-size:.68rem;font-weight:600;
                letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);
                border-bottom:1px solid var(--border);">Runtime</th>
            <th style="padding:.5rem 1rem;text-align:left;font-size:.68rem;font-weight:600;
                letter-spacing:.07em;text-transform:uppercase;color:var(--text-3);
                border-bottom:1px solid var(--border);">Memory</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


    # ── Insight block ─────────────────────────────────────────────────────
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
            <div style="background:var(--surface-2);border:1px solid var(--border);
                border-left:3px solid var(--blue);border-radius:var(--radius);
                padding:1rem 1.25rem;margin-top:1.5rem;">
              <p style="font-size:.72rem;font-weight:600;letter-spacing:.07em;
                 text-transform:uppercase;color:var(--blue);margin:0 0 .4rem;">Analysis insight</p>
              <p style="font-size:.87rem;color:var(--text-2);line-height:1.7;margin:0;">
                <strong style="color:var(--text)">{best_name}</strong> achieves the highest
                perceptual quality (PSNR&nbsp;{best_metrics['psnr']:.2f}&nbsp;dB,
                SSIM&nbsp;{best_metrics['ssim']:.3f}). Relative to the LAN baseline,
                TTA strategies improve {gain_parts}, demonstrating meaningful adaptation gains.
              </p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:4rem;padding-top:2rem;border-top:1px solid var(--border);">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:2rem;margin-bottom:2rem;">
    <div>
      <p style="font-size:.75rem;font-weight:600;color:var(--text);margin:0 0 .5rem;">About</p>
      <p style="font-size:.78rem;color:var(--text-3);line-height:1.65;margin:0;">
        Interactive benchmark comparing test-time adaptation methods for blind image restoration.
        Built as part of a Master's thesis in Computer Vision.
      </p>
    </div>
    <div>
      <p style="font-size:.75rem;font-weight:600;color:var(--text);margin:0 0 .5rem;">Methods</p>
      <p style="font-size:.78rem;color:var(--text-3);line-height:1.8;margin:0;">
        <strong style="color:var(--text-2)">LAN</strong> — Lightweight adaptive network<br>
        <strong style="color:var(--text-2)">TAO</strong> — Test-time adaptation optimization<br>
        <strong style="color:var(--text-2)">TTAD</strong> — TTA with diffusion
      </p>
    </div>
    <div>
      <p style="font-size:.75rem;font-weight:600;color:var(--text);margin:0 0 .5rem;">Metrics</p>
      <p style="font-size:.78rem;color:var(--text-3);line-height:1.8;margin:0;">
        <strong style="color:var(--text-2)">PSNR</strong> — Peak signal-to-noise ratio ↑<br>
        <strong style="color:var(--text-2)">SSIM</strong> — Structural similarity index ↑<br>
        <strong style="color:var(--text-2)">LPIPS</strong> — Learned perceptual similarity ↓
      </p>
    </div>
  </div>
  <p style="font-size:.72rem;color:var(--text-3);text-align:center;margin:0;">
    Thesis research · Test-time adaptation image restoration · Built with Streamlit
  </p>
</div>
""", unsafe_allow_html=True)