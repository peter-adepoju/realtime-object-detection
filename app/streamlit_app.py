"""
app/streamlit_app.py — Interactive Real-Time Object Detection Dashboard

Launched via:
    streamlit run app/streamlit_app.py

This app provides:
  1. Video upload and frame-by-frame detection with YOLOv8
  2. Per-class object count overlay and time-series
  3. Model variant selector (nano / small / medium)
  4. Confidence threshold slider
  5. Pre-computed benchmark metrics panel

Dependencies: streamlit, ultralytics, opencv-python, plotly, pandas, numpy
"""

import sys
import json
import time
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detection_system.loader import load_model, get_coco_classes
from detection_system.inference import detect_frame
from detection_system.counting import count_objects
from detection_system.visualize import draw_detections, bgr_to_rgb

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stProgress .st-bo { background-color: #4878CF; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 weights…")
def get_model(variant: str):
    """Load and cache a YOLOv8 model. Re-runs only when variant changes."""
    models_dir = PROJECT_ROOT / "models"
    return load_model(variant, models_dir=models_dir)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    # Model variant
    variant = st.selectbox(
        "Model Variant",
        options=["nano", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower. 'small' is recommended for CPU.",
    )

    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections below this confidence score are discarded.",
    )

    # Class filter
    coco_classes = get_coco_classes()
    selected_classes = st.multiselect(
        "Filter Classes (leave empty = all 80 COCO classes)",
        options=coco_classes,
        default=[],
        help="Select specific object classes to detect.",
    )
    class_ids = (
        [coco_classes.index(c) for c in selected_classes]
        if selected_classes else None
    )

    st.markdown("---")
    st.markdown("### 📂 Video Input")
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file to run detection on. Max recommended: 60 seconds.",
    )

   use_sample = st.checkbox("Use sample video (if available)", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**YOLOv8 Object Detection**  \n"
        "Author: P. O. Adepoju  \n"
        "Model: COCO-pretrained YOLOv8  \n"
        "80 object classes"
    )


# ── Main title ─────────────────────────────────────────────────────────────────
st.title("🎯 Real-Time Object Detection and Counting")
st.markdown(
    "Detect and count objects in video using **YOLOv8** pretrained on COCO (80 classes). "
    "Upload a video or use a sample clip to see detections, per-class counts, and "
    "engineering performance metrics."
)

# ── Load model ─────────────────────────────────────────────────────────────────
model = get_model(variant)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_detection, tab_counts, tab_benchmark = st.tabs([
    "🔍 Detection", "📊 Object Counts", "⚡ Benchmark Metrics"
])


# ── Helper: Resolve video path ────────────────────────────────────────────────
def get_video_bytes_path():
    """Return (bytes_data, temp_path) for the video to process."""
    if uploaded_file is not None:
        return uploaded_file.read(), None
    if use_sample:
        sample = PROJECT_ROOT / "data" / "raw" / "sample_traffic.mp4"
        mock = PROJECT_ROOT / "data" / "mock" / "mock_video.mp4"
        if sample.exists():
            return None, sample
        elif mock.exists():
            return None, mock
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab_detection:
    video_bytes, video_path = get_video_bytes_path()

    if video_bytes is None and video_path is None:
        st.info(
            "📁 Upload a video using the sidebar."
        )
        st.stop()

    # Write uploaded bytes to a temp file if needed
    if video_bytes is not None:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_bytes)
        tmp.close()
        video_path = Path(tmp.name)

    # ── Video metadata ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps_video
    cap.release()

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Resolution", f"{width}×{height}")
    col_info2.metric("Duration", f"{duration_s:.1f}s")
    col_info3.metric("Frames", str(total_frames))
    col_info4.metric("Video FPS", f"{fps_video:.0f}")

    st.markdown("---")

    # ── Frame slider ───────────────────────────────────────────────────────────
    frame_idx = st.slider(
        "Select frame",
        min_value=0,
        max_value=max(total_frames - 1, 0),
        value=min(50, total_frames // 2),
        step=1,
    )

    # ── Run detection on selected frame ───────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error(f"Could not read frame {frame_idx} from the video.")
    else:
        t0 = time.perf_counter()
        detections, latency_ms = detect_frame(
            model, frame,
            confidence=confidence,
            classes=class_ids,
        )
        t1 = time.perf_counter()

        annotated = draw_detections(frame.copy(), detections)
        annotated_rgb = bgr_to_rgb(annotated)

        # Display
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.subheader("Original Frame")
            st.image(bgr_to_rgb(frame), use_container_width=True)
        with col_annot:
            st.subheader(f"Detections — {len(detections)} objects")
            st.image(annotated_rgb, use_container_width=True)

        # Metrics row
        fps_inf = 1000.0 / latency_ms if latency_ms > 0 else 0
        counts = count_objects(detections)

        st.markdown("---")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Detections", len(detections))
        col_m2.metric("Latency", f"{latency_ms:.1f} ms")
        col_m3.metric("Equiv. FPS", f"{fps_inf:.1f}")
        col_m4.metric("Model", f"YOLOv8-{variant}")

        # Detection table
        if detections:
            st.markdown("### Detection Details")
            det_df = pd.DataFrame([
                {
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']:.3f}",
                    "x1": int(d["bbox_xyxy"][0]),
                    "y1": int(d["bbox_xyxy"][1]),
                    "x2": int(d["bbox_xyxy"][2]),
                    "y2": int(d["bbox_xyxy"][3]),
                }
                for d in detections
            ])
            st.dataframe(det_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Object Counts
# ═══════════════════════════════════════════════════════════════════════════════
with tab_counts:
    count_csv = PROJECT_ROOT / "data" / "processed" / "count_timeseries.csv"
    count_sum_csv = PROJECT_ROOT / "data" / "processed" / "count_summary.csv"

    if count_csv.exists():
        count_df = pd.read_csv(count_csv, index_col=0)
        class_cols = [c for c in count_df.columns if c != "time_s"]

        if not class_cols:
            st.warning("No objects were detected in the processed video.")
        else:
            top5 = count_df[class_cols].sum().sort_values(ascending=False).head(5).index.tolist()
            time_axis = count_df.get("time_s", count_df.index)

            # Count time-series
            st.subheader("Object Count Over Time")
            fig_ts = go.Figure()
            for cls in top5:
                if cls in count_df.columns:
                    fig_ts.add_trace(go.Scatter(
                        x=time_axis, y=count_df[cls],
                        mode="lines", name=cls,
                        line=dict(width=2),
                    ))
            fig_ts.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Objects per frame",
                legend_title="Class",
                height=350,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Summary table
            if count_sum_csv.exists():
                st.subheader("Detection Summary Statistics")
                summary_df = pd.read_csv(count_sum_csv, index_col=0)
                st.dataframe(summary_df.style.format(precision=1), use_container_width=True)

            # Total counts bar chart
            st.subheader("Total Detection Count by Class")
            totals = count_df[class_cols].sum().sort_values(ascending=False).head(10)
            fig_bar = px.bar(
                x=totals.index, y=totals.values,
                labels={"x": "Class", "y": "Total detections"},
                color=totals.index,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_bar.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(
            "📊 Count data not yet generated.  \n"
            "Run **Notebook 04** (video inference) then **Notebook 05** (counting) to generate it."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Benchmark Metrics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    bench_csv = PROJECT_ROOT / "data" / "processed" / "benchmark_results.csv"

    if bench_csv.exists():
        bench_df = pd.read_csv(bench_csv)

        st.subheader("Engineering Benchmark Results")
        st.markdown(
            "Benchmarked on identical frames (640×480 px, CPU inference). "
            "First 10 frames are warm-up and excluded from timing."
        )

        # Summary metrics
        cols = st.columns(len(bench_df))
        for col, (_, row) in zip(cols, bench_df.iterrows()):
            col.metric(row["model_name"], f"{row['mean_fps']:.1f} FPS",
                       f"{row['mean_latency_ms']:.0f} ms/frame")

        st.markdown("---")

        # Full table
        display_cols = [
            "model_name", "model_size_mb", "mean_fps",
            "mean_latency_ms", "p95_latency_ms", "peak_memory_mb"
        ]
        display_cols = [c for c in display_cols if c in bench_df.columns]
        st.dataframe(
            bench_df[display_cols].round(1).rename(columns={
                "model_name": "Model",
                "model_size_mb": "Size (MB)",
                "mean_fps": "Mean FPS",
                "mean_latency_ms": "Mean Lat (ms)",
                "p95_latency_ms": "p95 Lat (ms)",
                "peak_memory_mb": "Peak RAM Δ (MB)",
            }),
            use_container_width=True
        )

        # FPS vs Size scatter
        st.subheader("Speed vs. Model Size")
        fig_scatter = px.scatter(
            bench_df,
            x="model_size_mb",
            y="mean_fps",
            text="model_name",
            size=[50] * len(bench_df),
            color="model_name",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={"model_size_mb": "Model Size (MB)", "mean_fps": "Mean FPS"},
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Latency distribution
        st.subheader("Latency Distribution")
        lat_df = bench_df[["model_name", "mean_latency_ms", "p95_latency_ms", "p99_latency_ms"]].melt(
            id_vars="model_name",
            var_name="Percentile",
            value_name="Latency (ms)",
        )
        lat_df["Percentile"] = lat_df["Percentile"].map({
            "mean_latency_ms": "Mean",
            "p95_latency_ms": "p95",
            "p99_latency_ms": "p99",
        })
        fig_lat = px.bar(
            lat_df, x="model_name", y="Latency (ms)",
            color="Percentile", barmode="group",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={"model_name": "Model"},
        )
        fig_lat.update_layout(height=380)
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info(
            "⚡ Benchmark data not yet generated.  \n"
            "Run **Notebook 06** (benchmarking) to generate it."
        )
