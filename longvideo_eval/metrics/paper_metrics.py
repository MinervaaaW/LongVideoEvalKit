from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from longvideo_eval.io.video_clipper import VideoClipSpec
from longvideo_eval.models.features import cosine_matrix, l2_normalize


def time_range_indices(
    timestamps: Sequence[float],
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float32)
    if len(ts) == 0 or end_sec <= start_sec:
        return np.zeros((0,), dtype=int)
    return np.where((ts >= start_sec) & (ts < end_sec))[0].astype(int)


def fixed_duration_clip_indices(
    timestamps: Sequence[float],
    clip_seconds: float,
    max_clips: int,
) -> list[np.ndarray]:
    ts = np.asarray(timestamps, dtype=np.float32)
    if len(ts) == 0 or clip_seconds <= 0 or max_clips <= 0:
        return []
    start = float(ts[0])
    end = float(ts[-1]) + 1e-6
    clips: list[np.ndarray] = []
    cur = start
    while cur < end and len(clips) < max_clips:
        idx = time_range_indices(ts, cur, cur + clip_seconds)
        if len(idx) > 0:
            clips.append(idx)
        cur += clip_seconds
    return clips


def clip_mean_features(
    frame_features: np.ndarray,
    clip_indices: Sequence[np.ndarray],
) -> np.ndarray:
    means: List[np.ndarray] = []
    for idx in clip_indices:
        if len(idx) == 0:
            continue
        means.append(frame_features[idx].mean(axis=0))
    if not means:
        return np.zeros((0, frame_features.shape[-1]), dtype=np.float32)
    return l2_normalize(np.stack(means, axis=0))


def first_last_clip_features(
    frame_features: np.ndarray,
    timestamps: Sequence[float],
    clip_seconds: float,
) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float32)
    if len(ts) == 0 or clip_seconds <= 0:
        return np.zeros((0, frame_features.shape[-1]), dtype=np.float32)
    start = float(ts[0])
    end = float(ts[-1]) + 1e-6
    head = time_range_indices(ts, start, start + clip_seconds)
    tail = time_range_indices(ts, max(start, end - clip_seconds), end)
    return clip_mean_features(frame_features, [head, tail])


def compute_clip_first_last_drift(
    frame_features: np.ndarray,
    timestamps: Sequence[float],
    clip_seconds: float,
    field_name: str = "paper.drift_clip_first_last",
) -> Dict[str, float]:
    clip_feats = first_last_clip_features(frame_features, timestamps, clip_seconds)
    if len(clip_feats) < 2:
        return {}
    sim = float(cosine_matrix(clip_feats[1:2], clip_feats[0:1]).reshape(-1)[0])
    return {field_name: float(1.0 - sim)}


def compute_clip_global_repetition(
    frame_features: np.ndarray,
    timestamps: Sequence[float],
    clip_seconds: float,
    max_clips: int,
    prefix: str = "paper.repetition_clip_global",
) -> Dict[str, float]:
    clip_indices = fixed_duration_clip_indices(timestamps, clip_seconds, max_clips)
    clip_feats = clip_mean_features(frame_features, clip_indices)
    n = len(clip_feats)
    if n < 2:
        return {}
    sim = cosine_matrix(clip_feats, clip_feats)
    values: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(float(sim[i, j]))
    if not values:
        return {}
    arr = np.array(values, dtype=np.float32)
    return {
        prefix: float(arr.mean()),
        f"{prefix}.num_pairs": float(len(values)),
    }


def compute_context_forcing_window_consistency(
    frame_features: np.ndarray,
    timestamps: Sequence[float],
    window_radius_seconds: float,
    prefix: str = "paper.dinov2_cf",
) -> Dict[str, float]:
    ts = np.asarray(timestamps, dtype=np.float32)
    if len(ts) == 0 or window_radius_seconds < 0:
        return {}
    sims = cosine_matrix(frame_features, frame_features[0:1]).reshape(-1)
    scores: List[float] = []
    for center in ts:
        idx = time_range_indices(ts, float(center - window_radius_seconds), float(center + window_radius_seconds) + 1e-6)
        if len(idx) == 0:
            continue
        scores.append(float(sims[idx].mean()))
    if not scores:
        return {}
    arr = np.array(scores, dtype=np.float32)
    return {
        f"{prefix}.mean": float(arr.mean()),
        f"{prefix}.end": float(arr[-1]),
        f"{prefix}.drop": float(arr[0] - arr[-1]),
    }


def build_named_quality_clip_specs(
    base_name: str,
    duration_sec: float,
    head_tail_seconds: float,
    segment_seconds: float,
    max_segments: int,
) -> list[VideoClipSpec]:
    if duration_sec <= 0 or head_tail_seconds <= 0:
        return []
    end = float(duration_sec)
    specs: list[VideoClipSpec] = [
        VideoClipSpec(
            name=f"{base_name}__paper_head",
            start_sec=0.0,
            end_sec=min(head_tail_seconds, end),
            tag="head",
        ),
        VideoClipSpec(
            name=f"{base_name}__paper_tail",
            start_sec=max(0.0, end - head_tail_seconds),
            end_sec=end,
            tag="tail",
        ),
    ]
    if segment_seconds > 0 and max_segments > 0:
        cur = 0.0
        seg_idx = 0
        while cur < end and seg_idx < max_segments:
            seg_end = min(cur + segment_seconds, end)
            specs.append(
                VideoClipSpec(
                    name=f"{base_name}__paper_seg_{seg_idx:03d}",
                    start_sec=cur,
                    end_sec=seg_end,
                    tag=f"seg_{seg_idx:03d}",
                )
            )
            cur += segment_seconds
            seg_idx += 1
    return specs


def add_balance_to_summary_rows(
    summary_rows: Sequence[dict],
    drift_key: str = "paper.drift_clip_first_last.mean",
    repetition_key: str = "paper.repetition_clip_global.mean",
    out_key: str = "paper.balance.mean",
) -> list[dict]:
    rows = [dict(row) for row in summary_rows]
    candidates = [row for row in rows if drift_key in row and repetition_key in row]
    if len(candidates) < 2:
        return rows

    drift_vals = np.array([float(row[drift_key]) for row in candidates], dtype=np.float32)
    repetition_vals = np.array([float(row[repetition_key]) for row in candidates], dtype=np.float32)

    drift_min = float(drift_vals.min())
    drift_max = float(drift_vals.max())
    repetition_min = float(repetition_vals.min())
    repetition_max = float(repetition_vals.max())
    drift_span = drift_max - drift_min
    repetition_span = repetition_max - repetition_min

    for row in candidates:
        drift_scaled = 0.0 if drift_span <= 1e-12 else (float(row[drift_key]) - drift_min) / drift_span
        repetition_scaled = 0.0 if repetition_span <= 1e-12 else (float(row[repetition_key]) - repetition_min) / repetition_span
        row[out_key] = float(drift_scaled + repetition_scaled)
    return rows
