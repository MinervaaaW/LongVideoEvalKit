from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import cv2


@dataclass(frozen=True)
class VideoClipSpec:
    name: str
    start_sec: float
    end_sec: float
    tag: str = ""


def extract_named_time_clips(
    src_path: str | Path,
    output_dir: str | Path,
    clip_specs: Sequence[VideoClipSpec],
) -> Dict[str, Path]:
    src_path = Path(src_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [spec for spec in clip_specs if spec.end_sec > spec.start_sec]
    specs = sorted(specs, key=lambda spec: (spec.start_sec, spec.end_sec, spec.name))
    if not specs:
        return {}

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for clipping: {src_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 16.0
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video resolution for clipping: {src_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_duration = 1.0 / fps
    current_idx = 0
    frame_idx = 0
    writer = None
    writer_name: str | None = None
    created: Dict[str, Path] = {}
    written_counts: Dict[str, int] = {spec.name: 0 for spec in specs}

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = frame_idx / fps

            while current_idx < len(specs) and t >= specs[current_idx].end_sec:
                if writer is not None:
                    writer.release()
                    writer = None
                    writer_name = None
                current_idx += 1

            if current_idx >= len(specs):
                break

            spec = specs[current_idx]
            if t + frame_duration <= spec.start_sec:
                frame_idx += 1
                continue

            if spec.start_sec <= t < spec.end_sec:
                if writer is None or writer_name != spec.name:
                    if writer is not None:
                        writer.release()
                    dst_path = output_dir / f"{spec.name}.mp4"
                    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to create clip writer: {dst_path}")
                    writer_name = spec.name
                    created[spec.name] = dst_path
                writer.write(bgr)
                written_counts[spec.name] += 1
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {
        name: path
        for name, path in created.items()
        if written_counts.get(name, 0) > 0
    }
