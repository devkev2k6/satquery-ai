"""
frontend/app.py
===============
SatQuery AI - Interactive Earth Observation & Satellite Intelligence Dashboard.

Built with Streamlit.
Fulfills Phase 6 frontend requirements:
- Image ingestion for 1 or 2 satellite rasters (PNG, JPEG, TIFF/GeoTIFF).
- Sensor modality tagging (Optical vs SAR).
- Natural language query input and 1-Click Quick Demo Presets.
- Autonomous agent routing via agent.controller.process_query().
- Visual evidence display (inline images, visual grounding bounding box overlay,
  bi-temporal before/after inspection, optical/SAR side-by-side).
- Prominent answer display and collapsible auditable execution summary.
- One-click PDF intelligence report generation and download.
- Resilient error handling.
"""

import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import process_query
from frontend.report_generator import generate_pdf_report

# Configure Streamlit Page
st.set_page_config(
    page_title="SatQuery AI - Satellite Intelligence Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-task { background-color: #DBEAFE; color: #1E40AF; }
    .badge-success { background-color: #DCFCE7; color: #166534; }
    .badge-error { background-color: #FEE2E2; color: #991B1B; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def draw_bounding_box(image_path: str, bbox: list, label: str = "Grounding Target") -> Image.Image:
    """Draws a prominent bounding box on a satellite image for visual grounding."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        # Draw multi-pixel thickness border
        for offset in range(3):
            draw.rectangle(
                [max(0, x1 - offset), max(0, y1 - offset), min(img.width, x2 + offset), min(img.height, y2 + offset)],
                outline="#EF4444",  # Crimson red outline
            )

        # Label background
        text_bbox = [x1, max(0, y1 - 18), min(img.width, x1 + 110), y1]
        draw.rectangle(text_bbox, fill="#EF4444")
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill="#FFFFFF")

    return img


def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    """Saves a Streamlit UploadedFile to local temporary storage."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# =============================================================================
# SIDEBAR: 1-CLICK BENCHMARK DEMO PRESETS
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/satellite-in-orbit.png", width=64)
    st.title("SatQuery AI")
    st.caption("Autonomous Multimodal Satellite QA & Cross-Modal Fusion")
    st.markdown("---")

    st.subheader("⚡ 1-Click Demo Presets")
    st.markdown("Select an official problem statement benchmark query:")

    preset_options = [
        "-- Choose a preset or custom query --",
        "1. Captioning: Land-cover and major objects",
        "2. Grounding: Highlight the water body",
        "3. Change Detection: Bi-temporal change between two dates",
        "4. Fusion: Joint Optical + SAR analysis",
        "5. Trend VQA: Has built-up area changed?",
    ]

    selected_preset = st.selectbox("Preset Query Selection", preset_options, index=0)

    st.markdown("---")
    st.markdown("### 🛰️ System Architecture")
    st.markdown("""
    - **M1 Data Layer**: Multispectral & SAR loaders
    - **M2 Fine-Tuning**: BLIP-VQA with RS-LoRA
    - **M3 Baseline**: VQA, Captioning, Grounding
    - **M4 Multimodal**: Change Detection & Radar Fusion
    - **M5 Orchestrator**: Rule-Based Agent Classifier
    - **M6 Application**: Streamlit & PDF Reporting
    """)


# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.markdown('<div class="main-title">🛰️ SatQuery AI - Satellite Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Query Earth Observation imagery using natural language. The autonomous agent orchestrator dynamically validates context, invokes specialized AI vision models, and generates auditable intelligence reports.</div>', unsafe_allow_html=True)

# Default form state based on preset selection
default_query = ""
default_paths = []
default_types = []

if selected_preset == "1. Captioning: Land-cover and major objects":
    default_query = "Describe the land-cover and major objects visible in this image."
    default_paths = ["data/sample/optical/sample_optical_001.png"]
    default_types = ["optical"]

elif selected_preset == "2. Grounding: Highlight the water body":
    default_query = "Highlight the water body referred to in the query."
    default_paths = ["data/sample/optical/sample_optical_002.png"]
    default_types = ["optical"]

elif selected_preset == "3. Change Detection: Bi-temporal change between two dates":
    default_query = "What changed between these two dates, and where did the change occur?"
    default_paths = [
        "data/sample/pairs/sample_pair_001_t1.png",
        "data/sample/pairs/sample_pair_001_t2.png"
    ]
    default_types = ["optical", "optical"]

elif selected_preset == "4. Fusion: Joint Optical + SAR analysis":
    default_query = "Use the optical and SAR images together to identify built-up and water-covered regions."
    default_paths = [
        "data/sample/optical/sample_optical_003.png",
        "data/sample/sar/sample_sar_003.png"
    ]
    default_types = ["optical", "sar"]

elif selected_preset == "5. Trend VQA: Has built-up area changed?":
    default_query = "Has the built-up area increased, decreased, or remained unchanged?"
    default_paths = [
        "data/sample/pairs/sample_pair_001_t1.png",
        "data/sample/pairs/sample_pair_001_t2.png"
    ]
    default_types = ["optical", "optical"]

# Input Mode Tabs
tab_preset, tab_custom = st.tabs(["🚀 Benchmark Preset Mode", "📁 Custom Image Upload"])

active_image_paths = []
active_image_types = []

with tab_preset:
    if default_paths:
        st.info(f"**Loaded Benchmark Assets**: `{', '.join(default_paths)}`")
        col_prevs = st.columns(len(default_paths))
        for idx, (p, col) in enumerate(zip(default_paths, col_prevs)):
            with col:
                st.image(p, caption=f"Image {idx+1} ({default_types[idx].upper()})", use_container_width=True)
        active_image_paths = default_paths
        active_image_types = default_types
    else:
        st.write("👈 Select a 1-Click Demo Preset from the sidebar to preload benchmark images and queries.")

with tab_custom:
    uploaded_files = st.file_uploader(
        "Upload Satellite Imagery (1 or 2 images: .png, .jpg, .tif, .tiff)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        upload_dir = PROJECT_ROOT / "outputs" / "uploads"
        custom_paths = []
        custom_types = []

        if len(uploaded_files) > 2:
            st.warning("⚠️ SatQuery AI currently processes up to 2 images concurrently (Single Scene, Bi-Temporal Pair, or Optical+SAR Pair). Using the first 2 files.")
            uploaded_files = uploaded_files[:2]

        cols = st.columns(len(uploaded_files))
        for idx, (uf, col) in enumerate(zip(uploaded_files, cols)):
            saved_p = save_uploaded_file(uf, upload_dir)
            custom_paths.append(str(saved_p))
            with col:
                st.image(str(saved_p), caption=f"Uploaded {uf.name}", use_container_width=True)
                mod = st.selectbox(
                    f"Modality for Image {idx+1}",
                    options=["optical", "sar"],
                    index=1 if "sar" in uf.name.lower() or "s1" in uf.name.lower() else 0,
                    key=f"mod_{idx}"
                )
                custom_types.append(mod)

        if not default_paths or len(uploaded_files) > 0:
            active_image_paths = custom_paths
            active_image_types = custom_types

# Query Input Area
st.markdown("### 💬 Natural-Language Query")
user_query = st.text_area(
    "Type a question or command regarding the imagery:",
    value=default_query,
    placeholder="e.g. 'Describe the land-cover in this image' or 'What changed between these two dates?'",
    height=80,
)

col_submit, col_clear = st.columns([1, 6])
with col_submit:
    submit_clicked = st.button("🚀 Analyze Imagery", type="primary", use_container_width=True)

# =============================================================================
# EXECUTION & RESULTS PRESENTATION
# =============================================================================
if submit_clicked:
    if not active_image_paths:
        st.error("❌ Please upload at least one satellite image or select a benchmark preset from the sidebar.")
    elif not user_query.strip():
        st.error("❌ Please enter a natural-language query or description prompt.")
    else:
        with st.spinner("🧠 SatQuery Agent is classifying task and executing backend tools..."):
            result_payload = process_query(
                query=user_query,
                image_paths=active_image_paths,
                image_types=active_image_types
            )

        success = result_payload.get("success", False)
        exec_summary = result_payload.get("execution_summary", {})
        tool_result = result_payload.get("result", {})
        task = exec_summary.get("task", "unknown")
        tool_used = exec_summary.get("tool_used", "none")

        st.markdown("---")
        st.subheader("🎯 Analysis Results")

        if not success:
            err_msg = tool_result.get("error", "An error occurred during query execution.")
            st.error(f"**Execution Rejection / Error**: {err_msg}")
        else:
            # Result Banner
            conf = tool_result.get("confidence")
            conf_badge = f" | Confidence: **{conf:.2f}**" if conf is not None else ""
            st.markdown(
                f'<span class="metric-badge badge-task">Task: {task.upper()}</span> '
                f'<span class="metric-badge badge-success">Tool: {tool_used}</span> '
                f'{conf_badge}',
                unsafe_allow_html=True
            )
            st.markdown("")

            # Main text output
            main_text = ""
            if "caption" in tool_result:
                main_text = tool_result["caption"]
            elif "description" in tool_result:
                main_text = tool_result["description"]
            elif "answer" in tool_result:
                main_text = tool_result["answer"]
            else:
                main_text = str(tool_result)

            st.success(f"**Findings**: {main_text}")

            # Visual Evidence Section
            st.markdown("#### 🖼️ Visual Evidence")
            if task == "grounding" and "bbox" in tool_result:
                bbox = tool_result["bbox"]
                st.caption(f"**Bounding Box Coordinates**: `[x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]`")
                annotated_img = draw_bounding_box(active_image_paths[0], bbox)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(active_image_paths[0], caption="Raw Observation", use_container_width=True)
                with c2:
                    st.image(annotated_img, caption="Text-Guided Grounding Overlay", use_container_width=True)

            elif len(active_image_paths) == 2:
                c1, c2 = st.columns(2)
                with c1:
                    cap1 = "Prior Observation (T1)" if task == "change_detection" else f"Optical Imagery ({active_image_paths[0]})"
                    st.image(active_image_paths[0], caption=cap1, use_container_width=True)
                with c2:
                    cap2 = "Subsequent Observation (T2)" if task == "change_detection" else f"SAR Microwave Backscatter ({active_image_paths[1]})"
                    st.image(active_image_paths[1], caption=cap2, use_container_width=True)
            else:
                st.image(active_image_paths[0], caption=f"Analyzed Imagery ({active_image_paths[0]})", width=400)

        # Collapsible Execution Summary
        with st.expander("🛠️ Auditable Agent Execution Summary & Decision Trace", expanded=False):
            st.json(exec_summary)

        # PDF Report Generation & Download
        st.markdown("#### 📑 Intelligence Report")
        try:
            pdf_path = generate_pdf_report(user_query, result_payload)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download PDF Intelligence Report",
                data=pdf_bytes,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                type="secondary"
            )
        except Exception as e:
            st.warning(f"Could not compile PDF report: {e}")


# =============================================================================
# WSGI / SERVERLESS ENTRYPOINT (Vercel Python Runtime Support)
# =============================================================================
def app(environ, start_response):
    """WSGI entrypoint exposed for Vercel Python serverless deployment."""
    path = environ.get("PATH_INFO", "/")

    if path in ("/api/health", "/health"):
        status = "200 OK"
        headers = [("Content-Type", "application/json")]
        start_response(status, headers)
        return [json.dumps({"status": "healthy", "service": "SatQuery AI"}).encode("utf-8")]

    status = "200 OK"
    headers = [("Content-Type", "text/html; charset=utf-8")]
    start_response(status, headers)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatQuery AI - Satellite Intelligence Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .card {
            background: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2.5rem;
            max-width: 650px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            text-align: center;
        }
        .icon { font-size: 3rem; margin-bottom: 1rem; }
        h1 { font-size: 2rem; margin: 0 0 0.5rem 0; color: #38bdf8; }
        p { color: #94a3b8; line-height: 1.6; margin-bottom: 1.5rem; font-size: 1.05rem; }
        .features {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 1.5rem 0;
            text-align: left;
        }
        .feature-item {
            background: rgba(15, 23, 42, 0.6);
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            color: #cbd5e1;
            border-left: 3px solid #38bdf8;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #064e3b;
            color: #34d399;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon">🛰️</div>
        <h1>SatQuery AI</h1>
        <div class="status-badge">● System Online & Ready</div>
        <p>Interactive Earth Observation & Autonomous Satellite Intelligence Platform.</p>
        <div class="features">
            <div class="feature-item">📡 Optical & SAR Ingestion</div>
            <div class="feature-item">🤖 Autonomous Agent Routing</div>
            <div class="feature-item">🔍 Visual Grounding & VQA</div>
            <div class="feature-item">📑 Automated PDF Intelligence</div>
        </div>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 1.5rem;">
            Run <code>streamlit run frontend/app.py</code> locally to launch the interactive UI.
        </p>
    </div>
</body>
</html>"""
    return [html.encode("utf-8")]

