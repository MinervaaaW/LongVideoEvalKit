from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from longvideo_eval.schema import VideoMetadata

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
TIMESTAMP_SUFFIX_RE = re.compile(r"_(\d{8}_\d{6})$")


def _timestamp_key(path: Path) -> tuple[int, float]:
    match = TIMESTAMP_SUFFIX_RE.search(path.stem)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            return (1, dt.timestamp())
        except ValueError:
            pass
    return (0, float(path.stat().st_mtime))


def _select_records(
    records: list[tuple[str, str, Path]],
    video_selection: str,
) -> list[tuple[str, str, Path]]:
    if video_selection == "all":
        return sorted(records, key=lambda item: (item[0], item[1], str(item[2])))
    if video_selection != "latest":
        raise ValueError(f"Unsupported video_selection: {video_selection}")

    selected: dict[tuple[str, str], Path] = {}
    for model, prompt_id, path in records:
        key = (model, prompt_id)
        prev = selected.get(key)
        if prev is None or _timestamp_key(path) > _timestamp_key(prev):
            selected[key] = path
    return sorted(
        ((model, prompt_id, path) for (model, prompt_id), path in selected.items()),
        key=lambda item: (item[0], item[1], str(item[2])),
    )


def list_video_records(
    video_root: str | Path,
    layout: str = "standard",
    model_name: str | None = None,
    video_selection: str = "all",
) -> list[tuple[str, str, Path]]:
    """Return (model_name, prompt_id, path) tuples for supported dataset layouts."""
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"Video root not found: {root}")

    records: list[tuple[str, str, Path]] = []
    if layout == "standard":
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            model = model_dir.name
            for path in sorted(model_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    records.append((model, path.stem, path))
        return _select_records(records, video_selection)

    if layout == "prompt_dirs":
        model = model_name or root.name
        for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            videos = [
                path
                for path in sorted(prompt_dir.iterdir())
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ]
            for path in videos:
                records.append((model, prompt_dir.name, path))
        return _select_records(records, video_selection)

    raise ValueError(f"Unsupported dataset layout: {layout}")


def read_video_sampled(
    path: str | Path,
    sample_fps: float = 2.0,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, VideoMetadata]:
    """Read a video and return sampled RGB frames, timestamps, and metadata.

    Returns:
        frames: uint8 array of shape [N, H, W, 3] in RGB order.
        timestamps: float array of shape [N], seconds.
        metadata: original video metadata.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = sample_fps if sample_fps > 0 else 1.0

    duration = total / fps if total > 0 else 0.0
    step = max(int(round(fps / sample_fps)), 1) if sample_fps > 0 else 1

    frames: List[np.ndarray] = []
    times: List[float] = []
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            times.append(frame_idx / fps)
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")

    metadata = VideoMetadata(
        duration_sec=float(duration),
        video_fps=float(fps),
        num_frames=int(total),
        width=width,
        height=height,
    )
    return np.stack(frames, axis=0), np.array(times, dtype=np.float32), metadata


def segment_indices(
    timestamps: Sequence[float],
    segment_seconds: float,
    max_segments: int,
    max_frames_per_segment: int,
) -> list[np.ndarray]:
    """Return arrays of frame indices for equally spaced time segments."""
    ts = np.asarray(timestamps, dtype=np.float32)
    if len(ts) == 0:
        return []
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")

    start = float(ts[0])
    end = float(ts[-1]) + 1e-6
    segments: list[np.ndarray] = []
    cur = start
    while cur <= end and len(segments) < max_segments:
        mask = np.where((ts >= cur) & (ts < cur + segment_seconds))[0]
        if len(mask) > 0:
            if len(mask) > max_frames_per_segment:
                pick = np.linspace(0, len(mask) - 1, max_frames_per_segment).round().astype(int)
                mask = mask[pick]
            segments.append(mask.astype(int))
        cur += segment_seconds
    return segments
