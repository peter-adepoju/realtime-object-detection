"""
scripts/run_all.py — End-to-end pipeline runner.

This script reproduces the full notebook workflow non-interactively.
It is the scripted equivalent of running Notebooks 01–08 in order.

Usage
-----
    # From the project root:
    python scripts/run_all.py

    # With options:
    python scripts/run_all.py --variant small --max_frames 200

The notebooks remain the primary learning path. This script is for
reproducibility and automation (e.g., CI pipelines, rerunning after
model updates).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full object detection pipeline."
    )
    parser.add_argument(
        "--variant", type=str, default="small",
        choices=["nano", "small", "medium", "large", "xlarge"],
        help="YOLOv8 model variant to use (default: small).",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video file. Defaults to data/raw/sample_traffic.mp4 "
             "or data/mock/mock_video.mp4.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Max frames to process (default: all). Set to 100 for a quick test.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--skip_benchmark", action="store_true",
        help="Skip the multi-model benchmarking step (faster).",
    )
    return parser.parse_args()


def header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    args = parse_args()
    t_total_start = time.perf_counter()

    # ── Import pipeline modules ────────────────────────────────────────────────
    from detection_system.loader import load_model
    from detection_system.inference import detect_video
    from detection_system.counting import count_objects_over_time, summarise_counts
    from detection_system.benchmark import benchmark_model, compare_benchmarks
    from detection_system.utils import load_config, ensure_dirs, frames_from_video

    # ── Setup ──────────────────────────────────────────────────────────────────
    header("Step 0: Setup")
    ensure_dirs()
    cfg = load_config("config.yaml")

    MODELS_DIR = PROJECT_ROOT / "models"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve video path ─────────────────────────────────────────────────────
    if args.video:
        video_path = Path(args.video)
    else:
        real = PROJECT_ROOT / "data" / "raw" / "sample_traffic.mp4"
        mock = PROJECT_ROOT / "data" / "mock" / "mock_video.mp4"
        video_path = real if real.exists() else mock

    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        print("Run Notebook 02 to download a sample video, or pass --video <path>.")
        sys.exit(1)

    print(f"Video : {video_path}")
    print(f"Model : YOLOv8-{args.variant}")
    print(f"Conf  : {args.confidence}")
    print(f"Frames: {args.max_frames or 'all'}")

    # ── Step 1: Load model ─────────────────────────────────────────────────────
    header("Step 1: Load Model")
    model = load_model(args.variant, models_dir=MODELS_DIR)

    # ── Step 2: Full video inference ───────────────────────────────────────────
    header("Step 2: Video Inference")
    output_video = PROCESSED_DIR / "annotated_output.mp4"
    output_json = PROCESSED_DIR / "detections.json"

    results = detect_video(
        model=model,
        video_path=video_path,
        output_path=output_video,
        confidence=args.confidence,
        max_frames=args.max_frames,
        verbose=True,
    )

    # Save detection JSON
    json_payload = {
        "video_file": video_path.name,
        "model_variant": args.variant,
        "confidence_threshold": args.confidence,
        "frame_count": results["frame_count"],
        "fps_video": results["fps_video"],
        "frame_detections": results["all_detections"],
        "latencies_ms": results["latencies_ms"],
    }
    with open(output_json, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Detections saved to: {output_json}")

    # ── Step 3: Object counting ────────────────────────────────────────────────
    header("Step 3: Object Counting")
    count_df = count_objects_over_time(results["all_detections"])
    count_df["time_s"] = count_df.index / results["fps_video"]
    count_df.to_csv(PROCESSED_DIR / "count_timeseries.csv")
    print(f"Count time-series saved: {count_df.shape}")

    summary = summarise_counts(count_df.drop(columns=["time_s"], errors="ignore"))
    summary.to_csv(PROCESSED_DIR / "count_summary.csv")
    print("\nTop detected classes:")
    print(summary.head(10).to_string())

    # ── Step 4: Benchmarking ───────────────────────────────────────────────────
    if not args.skip_benchmark:
        header("Step 4: Multi-Model Benchmarking")

        import cv2
        bench_frames_raw = frames_from_video(video_path, max_frames=120)
        bench_frames = [cv2.resize(f, (640, 480)) for f in bench_frames_raw]

        variants_to_bench = cfg["benchmark"]["variants_to_compare"]
        weight_map = {"nano": "yolov8n.pt", "small": "yolov8s.pt", "medium": "yolov8m.pt"}

        bench_results = []
        for variant in variants_to_bench:
            print(f"\n--- Benchmarking {variant} ---")
            bench_model = load_model(variant, models_dir=MODELS_DIR)
            result = benchmark_model(
                model=bench_model,
                frames=bench_frames,
                model_name=f"YOLOv8-{variant}",
                weight_path=MODELS_DIR / weight_map.get(variant, ""),
                confidence=args.confidence,
                n_warmup=cfg["benchmark"]["n_warmup_frames"],
                verbose=True,
            )
            bench_results.append(result)
            del bench_model
            import gc; gc.collect()

        comparison_df = compare_benchmarks(bench_results)
        comparison_df.to_csv(PROCESSED_DIR / "benchmark_results.csv", index=False)
        print("\nBenchmark comparison:")
        print(comparison_df[["model_name", "mean_fps", "mean_latency_ms",
                              "p95_latency_ms", "model_size_mb"]].to_string(index=False))
    else:
        print("\nStep 4: Benchmarking skipped (--skip_benchmark).")

    # ── Done ───────────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_total_start
    header("Pipeline Complete")
    print(f"Total runtime: {t_total:.1f} seconds")
    print(f"\nOutputs in: {PROCESSED_DIR}")
    print("  ✓ detections.json")
    print("  ✓ annotated_output.mp4")
    print("  ✓ count_timeseries.csv")
    print("  ✓ count_summary.csv")
    if not args.skip_benchmark:
        print("  ✓ benchmark_results.csv")
    print("\nTo launch the Streamlit dashboard:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
