"""
tests/test_pipeline.py — Unit tests for the detection pipeline.

All tests use MOCK DATA (synthetic numpy arrays and tiny fake videos).
No real YOLOv8 weights are downloaded during testing.
Tests run without GPU and without internet access.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=detection_system
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detection_system.counting import (
    count_objects,
    count_objects_over_time,
    summarise_counts,
    get_dominant_classes,
)
from detection_system.visualize import draw_detections, bgr_to_rgb, resize_frame
from detection_system.utils import get_project_root


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — reusable mock data
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def blank_frame():
    """480×640 black BGR frame — no objects."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def coloured_frame():
    """480×640 coloured BGR frame — has distinct visual content."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.rectangle(frame, (400, 200), (600, 400), (255, 0, 0), -1)
    return frame


@pytest.fixture
def mock_detection():
    """A single well-formed detection dict."""
    return {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.85,
        "bbox_xyxy": [100.0, 150.0, 250.0, 400.0],
        "bbox_xywh": [175.0, 275.0, 150.0, 250.0],
    }


@pytest.fixture
def mock_detections_multi():
    """Three detections of two different classes."""
    return [
        {
            "class_id": 0, "class_name": "person", "confidence": 0.91,
            "bbox_xyxy": [50.0, 100.0, 200.0, 350.0],
            "bbox_xywh": [125.0, 225.0, 150.0, 250.0],
        },
        {
            "class_id": 2, "class_name": "car", "confidence": 0.78,
            "bbox_xyxy": [300.0, 200.0, 500.0, 380.0],
            "bbox_xywh": [400.0, 290.0, 200.0, 180.0],
        },
        {
            "class_id": 0, "class_name": "person", "confidence": 0.65,
            "bbox_xyxy": [400.0, 150.0, 550.0, 420.0],
            "bbox_xywh": [475.0, 285.0, 150.0, 270.0],
        },
    ]


@pytest.fixture
def mock_video_path(tmp_path):
    """Create a tiny 10-frame mock MP4 video in a temp directory."""
    video_path = tmp_path / "mock_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 25, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — counting module
# ═══════════════════════════════════════════════════════════════════════════════

class TestCountObjects:
    def test_empty_detections_returns_empty_dict(self):
        result = count_objects([])
        assert result == {}

    def test_single_detection(self, mock_detection):
        result = count_objects([mock_detection])
        assert result == {"person": 1}

    def test_multiple_same_class(self, mock_detection):
        dets = [mock_detection, mock_detection, mock_detection]
        result = count_objects(dets)
        assert result == {"person": 3}

    def test_multiple_classes(self, mock_detections_multi):
        result = count_objects(mock_detections_multi)
        assert result["person"] == 2
        assert result["car"] == 1

    def test_returns_dict_type(self, mock_detections_multi):
        result = count_objects(mock_detections_multi)
        assert isinstance(result, dict)
        for key in result:
            assert isinstance(key, str)
            assert isinstance(result[key], int)


class TestCountObjectsOverTime:
    def test_shape_is_frames_by_classes(self, mock_detections_multi):
        all_dets = [mock_detections_multi, mock_detections_multi, []]
        df = count_objects_over_time(all_dets)
        assert df.shape[0] == 3  # 3 frames
        assert "person" in df.columns
        assert "car" in df.columns

    def test_empty_frame_has_zeros(self, mock_detections_multi):
        all_dets = [mock_detections_multi, []]
        df = count_objects_over_time(all_dets)
        # Second frame (index 1) should have all zeros
        assert df.iloc[1]["person"] == 0
        assert df.iloc[1]["car"] == 0

    def test_counts_match_expected_values(self, mock_detections_multi):
        all_dets = [mock_detections_multi]
        df = count_objects_over_time(all_dets)
        assert df.iloc[0]["person"] == 2
        assert df.iloc[0]["car"] == 1

    def test_index_name_is_frame(self, mock_detections_multi):
        df = count_objects_over_time([mock_detections_multi])
        assert df.index.name == "frame"

    def test_all_values_are_non_negative(self, mock_detections_multi):
        df = count_objects_over_time([mock_detections_multi, [], mock_detections_multi])
        assert (df >= 0).all().all()


class TestSummariseCounts:
    def test_returns_dataframe(self, mock_detections_multi):
        count_df = count_objects_over_time([mock_detections_multi] * 5)
        summary = summarise_counts(count_df)
        assert isinstance(summary, pd.DataFrame)

    def test_expected_columns_present(self, mock_detections_multi):
        count_df = count_objects_over_time([mock_detections_multi] * 3)
        summary = summarise_counts(count_df)
        for col in ["total_appearances", "mean_per_frame", "max_in_single_frame",
                    "frames_present", "pct_frames_present"]:
            assert col in summary.columns, f"Missing column: {col}"

    def test_sorted_by_total_descending(self, mock_detections_multi):
        count_df = count_objects_over_time([mock_detections_multi] * 3)
        summary = summarise_counts(count_df)
        totals = summary["total_appearances"].tolist()
        assert totals == sorted(totals, reverse=True)

    def test_excludes_zero_count_classes(self):
        # One class with zero appearances in all frames
        dets = [{"class_id": 0, "class_name": "person", "confidence": 0.9,
                 "bbox_xyxy": [0, 0, 10, 10], "bbox_xywh": [5, 5, 10, 10]}]
        count_df = count_objects_over_time([dets], all_class_names=["person", "giraffe"])
        summary = summarise_counts(count_df)
        assert "giraffe" not in summary.index


class TestGetDominantClasses:
    def test_returns_correct_number(self, mock_detections_multi):
        count_df = count_objects_over_time([mock_detections_multi] * 3)
        top1 = get_dominant_classes(count_df, top_n=1)
        assert len(top1) == 1

    def test_most_common_is_first(self, mock_detections_multi):
        count_df = count_objects_over_time([mock_detections_multi] * 3)
        top2 = get_dominant_classes(count_df, top_n=2)
        assert top2[0] == "person"  # person appears 2× per frame, car 1×


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — visualize module
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrawDetections:
    def test_returns_ndarray(self, blank_frame, mock_detection):
        result = draw_detections(blank_frame.copy(), [mock_detection])
        assert isinstance(result, np.ndarray)

    def test_output_shape_unchanged(self, blank_frame, mock_detection):
        original_shape = blank_frame.shape
        result = draw_detections(blank_frame.copy(), [mock_detection])
        assert result.shape == original_shape

    def test_empty_detections_returns_unchanged_frame(self, blank_frame):
        result = draw_detections(blank_frame.copy(), [])
        np.testing.assert_array_equal(result, blank_frame)

    def test_multiple_detections(self, coloured_frame, mock_detections_multi):
        # Should not raise any exception
        result = draw_detections(coloured_frame.copy(), mock_detections_multi)
        assert result is not None
        assert result.shape == coloured_frame.shape


class TestBgrToRgb:
    def test_channel_order_swapped(self):
        # BGR frame where R=0, G=0, B=255 (blue in BGR)
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel in BGR
        rgb = bgr_to_rgb(bgr)
        # After conversion, the blue value should be in channel 2 (RGB red)
        assert rgb[0, 0, 2] == 255
        assert rgb[0, 0, 0] == 0

    def test_shape_preserved(self, blank_frame):
        rgb = bgr_to_rgb(blank_frame)
        assert rgb.shape == blank_frame.shape


class TestResizeFrame:
    def test_wide_frame_gets_resized(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = resize_frame(frame, max_width=640)
        assert result.shape[1] <= 640

    def test_narrow_frame_unchanged(self):
        frame = np.zeros((480, 320, 3), dtype=np.uint8)
        result = resize_frame(frame, max_width=640)
        assert result.shape == frame.shape

    def test_aspect_ratio_preserved(self):
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        result = resize_frame(frame, max_width=400)
        original_ratio = 800 / 600
        new_ratio = result.shape[1] / result.shape[0]
        assert abs(original_ratio - new_ratio) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — utils module
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetProjectRoot:
    def test_returns_path_object(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_requirements_txt_exists_at_root(self):
        root = get_project_root()
        assert (root / "requirements.txt").exists()

    def test_src_dir_exists_at_root(self):
        root = get_project_root()
        assert (root / "src").exists()


class TestSaveFigureAndTable:
    def test_save_figure_creates_files(self, tmp_path):
        """Test that save_figure writes PNG and PDF files."""
        import matplotlib.pyplot as plt
        from detection_system.utils import save_figure, get_project_root

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        saved = save_figure(
            fig, "test_fig",
            subdirs=[str(tmp_path)],
            formats=["png"],
            close_after=True,
        )
        assert len(saved) == 1
        assert saved[0].exists()
        assert saved[0].suffix == ".png"

    def test_save_table_creates_csv(self, tmp_path):
        """Test that save_table writes a CSV file."""
        from detection_system.utils import save_table

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        saved = save_table(df, "test_table", subdirs=[str(tmp_path)], latex=False)
        assert len(saved) >= 1
        assert any(p.suffix == ".csv" for p in saved)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — video utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestFramesFromVideo:
    def test_extracts_correct_number_of_frames(self, mock_video_path):
        from detection_system.utils import frames_from_video
        frames = frames_from_video(mock_video_path, max_frames=5)
        assert len(frames) <= 5
        assert len(frames) > 0

    def test_frames_are_numpy_arrays(self, mock_video_path):
        from detection_system.utils import frames_from_video
        frames = frames_from_video(mock_video_path, max_frames=3)
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3  # H, W, C

    def test_missing_video_raises(self, tmp_path):
        from detection_system.utils import frames_from_video
        with pytest.raises(RuntimeError):
            frames_from_video(tmp_path / "nonexistent.mp4", max_frames=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests — data schema / JSON format
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectionSchema:
    """Verify that detection dicts have the expected schema."""

    REQUIRED_KEYS = {"class_id", "class_name", "confidence", "bbox_xyxy", "bbox_xywh"}

    def test_single_detection_has_required_keys(self, mock_detection):
        assert self.REQUIRED_KEYS.issubset(mock_detection.keys())

    def test_class_id_is_int(self, mock_detection):
        assert isinstance(mock_detection["class_id"], int)

    def test_class_name_is_str(self, mock_detection):
        assert isinstance(mock_detection["class_name"], str)

    def test_confidence_in_range(self, mock_detection):
        assert 0.0 <= mock_detection["confidence"] <= 1.0

    def test_bbox_xyxy_has_four_values(self, mock_detection):
        assert len(mock_detection["bbox_xyxy"]) == 4

    def test_bbox_x2_greater_than_x1(self, mock_detection):
        x1, y1, x2, y2 = mock_detection["bbox_xyxy"]
        assert x2 > x1
        assert y2 > y1

    def test_all_mock_detections_valid(self, mock_detections_multi):
        for det in mock_detections_multi:
            assert self.REQUIRED_KEYS.issubset(det.keys())
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox_xyxy"]) == 4
