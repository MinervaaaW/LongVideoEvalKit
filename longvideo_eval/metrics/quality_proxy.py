from __future__ import annotations

from typing import Dict, Sequence

import cv2
import numpy as np


def _frame_quality_proxy(frame_rgb: np.ndarray) -> float:
    """A lightweight no-reference proxy for smoke tests.

    Combines sharpness and exposure stability. This is not MUSIQ and should not be
    reported as VBench Imaging Quality.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Saturating transform keeps the scale bounded and robust across resolutions.
    sharp_score = sharpness / (sharpness + 100.0)

    mean_luma = float(gray.mean()) / 255.0
    exposure_score = 1.0 - min(abs(mean_luma - 0.5) / 0.5, 1.0)
    return float(0.7 * sharp_score + 0.3 * exposure_score)


def compute_quality_proxy(frames: np.ndarray, segments: Sequence[np.ndarray]) -> Dict[str, float]:
    frame_scores = np.array([_frame_quality_proxy(fr) for fr in frames], dtype=np.float32)
    segment_scores = []
    for idx in segments:
        if len(idx) == 0:
            continue
        segment_scores.append(float(frame_scores[idx].mean()))

    if not segment_scores:
        return {}
    seg = np.array(segment_scores, dtype=np.float32)
    delta_abs = float(abs(seg[0] - seg[-1])) if len(seg) >= 2 else 0.0
    delta_drop = float(seg[0] - seg[-1]) if len(seg) >= 2 else 0.0
    return {
        "quality_proxy.mean": float(seg.mean()),
        "quality_proxy.head": float(seg[0]),
        "quality_proxy.tail": float(seg[-1]),
        "quality_proxy.segment_std": float(seg.std(ddof=0)),
        "quality_proxy.delta_abs": delta_abs,
        "quality_proxy.delta_drop": delta_drop,
    }
