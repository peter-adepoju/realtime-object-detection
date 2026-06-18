# Notebooks — Execution Guide

Run notebooks in the order shown. Each notebook builds on outputs from the previous one.

## Execution Order

| # | Notebook | Purpose | Key Outputs |
|---|----------|---------|-------------|
| 01 | `01_environment_and_model_setup.ipynb` | Verify environment; download YOLOv8 weights | `models/*.pt` |
| 02 | `02_dataset_and_coco_exploration.ipynb` | Download sample video; explore COCO classes | `data/raw/sample_traffic.mp4`, `data/mock/mock_video.mp4` |
| 03 | `03_single_frame_detection_and_visualisation.ipynb` | Single-frame detection; threshold tuning | `reports/figures/03_*.png` |
| 04 | `04_video_inference_pipeline.ipynb` | Full video inference + annotated output | `data/processed/detections.json`, `annotated_output.mp4` |
| 05 | `05_object_counting_logic.ipynb` | Count time-series and summary stats | `data/processed/count_timeseries.csv` |
| 06 | `06_model_benchmarking_latency_fps_memory.ipynb` | Engineering benchmarks (nano/small/medium) | `data/processed/benchmark_results.csv` |
| 07 | `07_model_size_comparison_nano_small_medium.ipynb` | Trade-off analysis; final comparison table | `reports/figures/07_*.png`, `reports/tables/07_*.csv` |
| 08 | `08_results_analysis_and_publication_figures.ipynb` | Publication-quality final figures and tables | `reports/figures/fig*.png`, `paper/figures/` |
| 09 | `09_streamlit_app_walkthrough.ipynb` | Dashboard documentation and launch guide | None (documentation) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter lab notebooks/

# Run notebooks 01 → 09 in order
```

## Minimum Viable Run (quick test)

If you want to test the pipeline quickly without processing the full video, open `04_video_inference_pipeline.ipynb` and set `max_frames=50` in the `detect_video()` call before running.

## Notes

- All figures are saved automatically to `reports/figures/` — you do not need to save them manually.
- Mock data (in `data/mock/`) is used only for tests. All real analysis uses the video in `data/raw/`.
- If a notebook fails to find an expected input file, it will print a clear error message telling you which earlier notebook to run first.
