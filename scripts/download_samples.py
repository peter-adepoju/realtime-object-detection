"""
scripts/download_samples.py — Download sample videos for the project.

Usage
-----
    python scripts/download_samples.py

Downloads a short public-domain traffic video to data/raw/.
Falls back to creating a mock video if all downloads fail.
"""

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MOCK_DIR = PROJECT_ROOT / "data" / "mock"

RAW_DIR.mkdir(parents=True, exist_ok=True)
MOCK_DIR.mkdir(parents=True, exist_ok=True)

# Public domain video sources (try in order)
SOURCES = [
    {
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "filename": "sample_traffic.mp4",
        "description": "Google sample video (~2 MB)",
    },
    {
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4",
        "filename": "sample_traffic.mp4",
        "description": "Google sample video — street scene (~5 MB)",
    },
]


def download_video(url: str, dest: Path, description: str) -> bool:
    """Try to download a video. Returns True on success."""
    print(f"Attempting: {description}")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"  ✓ Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def create_mock_video():
    """Generate a synthetic mock video in data/mock/."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("cv2 not available — cannot create mock video.")
        return

    mock_path = MOCK_DIR / "mock_video.mp4"
    if mock_path.exists():
        print(f"Mock video already exists: {mock_path}")
        return

    WIDTH, HEIGHT, FPS, N_FRAMES = 640, 480, 25, 100
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mock_path), fourcc, FPS, (WIDTH, HEIGHT))
    np.random.seed(42)

    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        x = int(50 + (i / N_FRAMES) * (WIDTH - 100))
        cv2.rectangle(frame, (x, 150), (x + 80, 250), (86, 180, 233), -1)
        cv2.putText(frame, f"[MOCK DATA] Frame {i}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    print(f"✓ Mock video created: {mock_path}")


def main():
    print("=" * 55)
    print("  Downloading sample videos")
    print("=" * 55)

    dest = RAW_DIR / "sample_traffic.mp4"

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"Sample video already exists ({size_mb:.1f} MB): {dest}")
        print("Delete it to re-download.")
        create_mock_video()
        return

    for source in SOURCES:
        success = download_video(source["url"], dest, source["description"])
        if success:
            break
    else:
        print("\nAll download attempts failed.")
        print("Creating mock video as fallback...")
        create_mock_video()
        return

    create_mock_video()

    print("\nDownload complete. Available videos:")
    for path in list(RAW_DIR.glob("*.mp4")) + list(MOCK_DIR.glob("*.mp4")):
        size_mb = path.stat().st_size / (1024 ** 2)
        label = "[MOCK]" if "mock" in path.name else "[REAL]"
        print(f"  {label} {path.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
