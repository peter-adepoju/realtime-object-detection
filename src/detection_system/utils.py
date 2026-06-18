"""
utils.py — Shared helpers: figure saving, path management, config loading.

This module handles:
- Saving matplotlib figures to reports/figures/ (PNG + PDF)
- Saving pandas DataFrames to reports/tables/ (CSV + LaTeX)
- Finding the project root directory reliably
- Loading YAML config files

Why a separate module?
  Every notebook saves figures and tables. Centralising the save logic
  ensures consistent filenames, DPI settings, and directory creation
  across all notebooks without repetition.

Inputs:  matplotlib Figure objects, pandas DataFrames, config file paths
Outputs: Saved files in reports/figures/ and reports/tables/
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def get_project_root() -> Path:
    """
    Walk up from this file's location to find the project root.

    The project root is identified by the presence of requirements.txt.
    This makes path resolution work regardless of the current working
    directory when a notebook or script is run.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Raises
    ------
    RuntimeError
        If requirements.txt is not found in any parent directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "requirements.txt").exists():
            return parent
    raise RuntimeError(
        "Could not find project root (requirements.txt not found). "
        "Make sure you are running from inside the project directory."
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
    subdirs: Optional[list[str]] = None,
    dpi: int = 300,
    formats: Optional[list[str]] = None,
    close_after: bool = True,
) -> list[Path]:
    """
    Save a matplotlib figure to reports/figures/ and paper/figures/.

    Saves high-resolution PNG by default, plus PDF for vector graphics.
    Both directories are created automatically if they do not exist.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        Base filename WITHOUT extension, e.g. "06_fps_comparison".
        The extension is added automatically for each format.
    subdirs : list of str, optional
        Additional subdirectory paths to also save to.
        Defaults to ["reports/figures", "paper/figures"].
    dpi : int
        Resolution for raster formats (PNG). Default 300 for publication.
    formats : list of str, optional
        List of file extensions to save. Default: ["png", "pdf"].
    close_after : bool
        If True, calls plt.close(fig) after saving to free memory.

    Returns
    -------
    list of Path
        Paths of all saved files.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> saved = save_figure(fig, "01_example_plot")
    >>> print(saved)
    """
    root = get_project_root()

    if subdirs is None:
        subdirs = ["reports/figures", "paper/figures"]

    if formats is None:
        formats = ["png", "pdf"]

    saved_paths = []

    for subdir in subdirs:
        out_dir = root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            out_path = out_dir / f"{filename}.{fmt}"
            fig.savefig(
                out_path,
                dpi=dpi if fmt == "png" else None,  # PDF is always vector
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            saved_paths.append(out_path)
            try:
                display_path = out_path.relative_to(root)
            except ValueError:
                display_path = out_path
            print(f"  Saved: {display_path}")

    if close_after:
        plt.close(fig)

    return saved_paths


def save_table(
    df: pd.DataFrame,
    filename: str,
    subdirs: Optional[list[str]] = None,
    index: bool = True,
    latex: bool = True,
) -> list[Path]:
    """
    Save a pandas DataFrame to reports/tables/ as CSV (and optionally LaTeX).

    Parameters
    ----------
    df : pd.DataFrame
        The table to save.
    filename : str
        Base filename WITHOUT extension, e.g. "benchmark_comparison".
    subdirs : list of str, optional
        Directories to save to. Defaults to ["reports/tables", "paper/tables"].
    index : bool
        Whether to include the DataFrame index in the CSV. Default True.
    latex : bool
        If True, also saves a .tex file (LaTeX table format).

    Returns
    -------
    list of Path
        Paths of all saved files.

    Example
    -------
    >>> save_table(summary_df, "06_model_comparison")
    """
    root = get_project_root()

    if subdirs is None:
        subdirs = ["reports/tables", "paper/tables"]

    saved_paths = []

    for subdir in subdirs:
        out_dir = root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = out_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=index)
        saved_paths.append(csv_path)
        try:
            display_path = csv_path.relative_to(root)
        except ValueError:
            display_path = csv_path
        print(f"  Saved: {display_path}")

        # LaTeX
        if latex:
            tex_path = out_dir / f"{filename}.tex"
            try:
                df.to_latex(tex_path, index=index)
                saved_paths.append(tex_path)
                try:
                    display_tex = tex_path.relative_to(root)
                except ValueError:
                    display_tex = tex_path
                print(f"  Saved: {display_tex}")
            except Exception as e:
                print(f"  Warning: Could not save LaTeX table: {e}")

    return saved_paths


def load_config(config_name: str = "config.yaml") -> dict:
    """
    Load a YAML config file from the configs/ directory.

    Parameters
    ----------
    config_name : str
        Filename of the config file, e.g. "config.yaml" or "benchmark.yaml".

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist in configs/.

    Example
    -------
    >>> cfg = load_config("config.yaml")
    >>> print(cfg["confidence_threshold"])
    """
    root = get_project_root()
    config_path = root / "configs" / config_name

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: {list((root / 'configs').glob('*.yaml'))}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dirs() -> None:
    """
    Create all required project output directories if they do not exist.

    Called at the top of run_all.py and scripts to guarantee that
    all save targets exist before any notebook output is written.
    """
    root = get_project_root()
    dirs = [
        "reports/figures",
        "reports/tables",
        "paper/figures",
        "paper/tables",
        "data/interim",
        "data/processed",
        "models",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    print("All output directories verified.")


def frames_from_video(video_path: str | Path, max_frames: int = 200) -> list:
    """
    Extract frames from a video file into a list of numpy arrays.

    Used by benchmark.py to get a fixed set of frames for timing.
    Uniformly samples up to max_frames frames across the full video.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    max_frames : int
        Maximum number of frames to extract.

    Returns
    -------
    list of np.ndarray
        List of BGR frames.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    frames = []
    idx = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step

    cap.release()
    print(f"Extracted {len(frames)} frames from {Path(video_path).name}")
    return frames
