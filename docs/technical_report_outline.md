# Technical Report — Real-Time Object Detection and Counting from Video Streams

**Author:** P. O. Adepoju  
**Affiliation:** AI/ML Engineering Portfolio  
**Date:** 2025  
**Status:** Template — fill in with results from notebooks

---

## Abstract

*(Fill in after running all notebooks and collecting results.)*

This report describes the design, implementation, and evaluation of a real-time object detection and counting system built using YOLOv8 pretrained on the COCO dataset. The system processes video streams frame-by-frame, annotates detections with bounding boxes and class labels, and counts objects of interest per frame. We compare three model variants — nano, small, and medium — across key engineering metrics: inference latency, sustained throughput (FPS), on-disk model size, and peak memory usage. Results suggest a clear monotonic relationship between model size and latency, with the nano variant achieving approximately `X` FPS and the medium variant achieving approximately `Y` FPS on CPU at 640×480 resolution. We discuss practical deployment trade-offs and limitations including the absence of temporal tracking, reliance on COCO class coverage, and hardware-specific variability.

---

## 1. Introduction

### 1.1 Motivation

Computer vision–based object detection is a foundational capability in modern AI/ML engineering. Applications span:

- **Traffic monitoring** — counting vehicles per lane, detecting violations
- **Warehouse automation** — tracking inventory, detecting anomalies
- **Retail analytics** — measuring customer dwell time, product interaction
- **Safety monitoring** — detecting crowding, fall detection, PPE compliance
- **Event analysis** — crowd counting, occupancy estimation

### 1.2 Project Goals

This project has two goals:
1. Build a production-quality, end-to-end video object detection and counting pipeline.
2. Systematically benchmark YOLOv8 model variants to characterise the speed-accuracy-memory trade-off.

### 1.3 Scope and Limitations

- Uses pretrained COCO weights — no domain-specific fine-tuning.
- Frame-level counting only — no temporal tracking across frames.
- CPU-only benchmarks in this report — GPU results would differ substantially.

---

## 2. Methods

### 2.1 Model

YOLOv8 (Ultralytics, 2023) is a single-stage object detector that predicts bounding boxes and class probabilities in a single forward pass. We use the COCO-pretrained variants:

| Variant | Parameters | File Size | Architecture |
|---------|-----------|-----------|-------------|
| YOLOv8n | ~3.2M | ~6 MB | Nano backbone |
| YOLOv8s | ~11.2M | ~22 MB | Small backbone |
| YOLOv8m | ~25.9M | ~50 MB | Medium backbone |

### 2.2 Dataset

COCO 80-class pretrained weights. No additional training data was used. The pretrained model detects persons, vehicles, animals, furniture, food, electronics, and household items.

### 2.3 Detection Pipeline

```
Video File
    → Frame Extraction (cv2.VideoCapture)
    → YOLOv8 Inference (model.predict)
    → Bounding Box Parsing (xyxy format)
    → Confidence Thresholding (default: 0.25)
    → Per-Frame Count Aggregation
    → Annotated Video Writer (cv2.VideoWriter)
    → Detection JSON Export
```

### 2.4 Object Counting

Frame-level counting: for each frame, we count the number of detected instances per class. This produces a (n_frames × n_classes) count matrix. Limitations: without temporal tracking, re-entering objects are double-counted.

### 2.5 Benchmarking Protocol

- **Warm-up:** First 10 frames discarded (JIT compilation overhead).
- **Measurement:** 100 frames per model.
- **Frame size:** 640×480 px (all models evaluated at identical resolution).
- **Device:** CPU only.
- **Timing:** `time.perf_counter()` — wall-clock time per forward pass.
- **Memory:** `psutil.Process.memory_info().rss` — peak RSS increase during inference.

---

## 3. Results

*(Fill in from Notebook 06 and Notebook 07 outputs.)*

### 3.1 Inference Speed

**Table 1: Benchmark results (CPU, 640×480 px)**

| Model | FPS (mean) | Latency-mean (ms) | p95 Lat (ms) | Size (MB) | RAM Δ (MB) |
|-------|-----------|------------------|-------------|----------|-----------|
| YOLOv8n | TBD | TBD | TBD | ~6 | TBD |
| YOLOv8s | TBD | TBD | TBD | ~22 | TBD |
| YOLOv8m | TBD | TBD | TBD | ~50 | TBD |

*Replace TBD values with results from `reports/tables/06_benchmark_comparison.csv`.*

### 3.2 Object Counting Results

*(Fill in from Notebook 05 outputs.)*

Top detected classes in the sample video: ...

**Figure 3:** Count time-series for top 5 classes.

### 3.3 Speed-Accuracy Trade-off

*(Discuss FPS vs. model scale relationship.)*

---

## 4. Discussion

### 4.1 Deployment Recommendations

Based on the benchmarking results:

- **Edge / CPU real-time (>15 FPS):** Use YOLOv8n (nano). Suitable for resource-constrained devices, webcam-based applications, and latency-sensitive pipelines.
- **Server / near real-time (8–15 FPS):** Use YOLOv8s (small). Good balance of speed and detection sensitivity.
- **Batch or offline processing:** Use YOLOv8m (medium). Higher detection sensitivity at the cost of throughput.
- **GPU deployment:** All variants will achieve substantially higher FPS. The relative ordering (nano > small > medium in speed) is preserved.

### 4.2 Limitations

1. **No temporal tracking.** Frame-level counting counts objects independently per frame. Objects that exit and re-enter the frame will be double-counted in cumulative totals.

2. **COCO class coverage.** The model can only detect the 80 COCO object categories. Domain-specific objects (e.g., custom products, specific vehicle types) require fine-tuning on labelled domain data.

3. **Hardware variability.** CPU benchmarks depend on chip generation, clock speed, thermal throttling, and background processes. Results should not be directly compared across different machines.

4. **No ground-truth accuracy evaluation.** mAP evaluation on COCO val2017 was not performed. The detection count proxy used in Notebook 07 is not a substitute for proper accuracy measurement.

### 4.3 Future Extensions

1. **Temporal tracking (SORT / ByteTrack):** Maintain object identities across frames for true unique-object counting.
2. **Fine-tuning on domain data:** Annotate a domain-specific dataset and fine-tune YOLOv8 for improved precision on target objects.
3. **GPU deployment and ONNX export:** Export to ONNX/TensorRT for GPU inference; benchmark GPU throughput.
4. **Edge deployment:** Test on Raspberry Pi / Jetson Nano; optimise with quantisation (INT8).
5. **Line-crossing counting:** Implement virtual line/ROI counting for traffic flow measurement.

---

## 5. Reproducibility

All code is available at: `[GitHub link]`  
Environment: Python 3.10, ultralytics≥8.0, torch≥2.0, opencv-python≥4.7  
Setup: `pip install -r requirements.txt`  
Full pipeline: `python scripts/run_all.py`  
Notebooks: Run in order, 01–09.

---

## 6. Ethical Considerations

This system can be used for surveillance and monitoring of individuals without their knowledge or consent. Deployment must:
- Comply with applicable privacy laws (GDPR, NDPR for Nigeria, etc.)
- Obtain appropriate consent where required
- Not be used for biometric identification or racial/ethnic profiling
- Include human oversight for high-stakes decisions (safety systems)

---

## References

1. Jocher, G., et al. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
2. Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV 2014.
3. Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016.
4. Bewley, A., et al. (2016). *Simple Online and Realtime Tracking.* ICIP 2016.
