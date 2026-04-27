from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from longvideo_eval.models.features import FeatureExtractor, cosine_matrix, l2_normalize


def segment_mean_features(features: np.ndarray, segments: Sequence[np.ndarray]) -> np.ndarray:
    means: List[np.ndarray] = []
    for idx in segments:
        if len(idx) == 0:
            continue
        means.append(features[idx].mean(axis=0))
    if not means:
        return np.zeros((0, features.shape[-1]), dtype=np.float32)
    return l2_normalize(np.stack(means, axis=0))


def long_consistency_from_features(segment_feats: np.ndarray, prefix: str) -> Dict[str, float]:
    if len(segment_feats) == 0:
        return {}
    ref = segment_feats[0:1]
    sims = cosine_matrix(segment_feats, ref).reshape(-1)
    out = {
        f"{prefix}.mean": float(sims.mean()),
        f"{prefix}.end": float(sims[-1]),
        f"{prefix}.drop": float(sims[0] - sims[-1]),
    }
    return out


def drift_from_features(segment_feats: np.ndarray, prefix: str) -> Dict[str, float]:
    if len(segment_feats) <= 1:
        return {}
    ref = segment_feats[0:1]
    sims = cosine_matrix(segment_feats[1:], ref).reshape(-1)
    drift = 1.0 - sims
    return {
        f"{prefix}.mean": float(drift.mean()),
        f"{prefix}.end": float(drift[-1]),
        f"{prefix}.max": float(drift.max()),
    }


def repetition_from_features(
    segment_feats: np.ndarray,
    prefix: str,
    min_gap_segments: int = 5,
    threshold: float = 0.95,
) -> Dict[str, float]:
    n = len(segment_feats)
    if n <= min_gap_segments:
        return {}
    sim = cosine_matrix(segment_feats, segment_feats)
    values: List[float] = []
    hits = 0
    for i in range(n):
        for j in range(i + min_gap_segments, n):
            s = float(sim[i, j])
            values.append(s)
            if s >= threshold:
                hits += 1
    if not values:
        return {}
    arr = np.array(values, dtype=np.float32)
    return {
        f"{prefix}.ratio": float(hits / len(values)),
        f"{prefix}.mean_sim": float(arr.mean()),
        f"{prefix}.max_sim": float(arr.max()),
        f"{prefix}.num_pairs": float(len(values)),
    }


def clip_t_from_segments(
    segment_feats: np.ndarray,
    text_feat: np.ndarray,
    prefix: str = "clip_t",
) -> Dict[str, float]:
    if len(segment_feats) == 0:
        return {}
    sims = cosine_matrix(segment_feats, text_feat.reshape(1, -1)).reshape(-1)
    return {
        f"{prefix}.mean": float(sims.mean()),
        f"{prefix}.end": float(sims[-1]),
        f"{prefix}.drop": float(sims[0] - sims[-1]),
    }


def compute_metric_bundle_from_segment_features(
    segment_feats: np.ndarray,
    prefix: str,
    repetition_min_gap_segments: int,
    repetition_threshold: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(long_consistency_from_features(segment_feats, f"{prefix}_lc"))
    out.update(drift_from_features(segment_feats, f"drift_{prefix}"))
    out.update(
        repetition_from_features(
            segment_feats,
            f"repetition_{prefix}",
            min_gap_segments=repetition_min_gap_segments,
            threshold=repetition_threshold,
        )
    )
    return out


def compute_metric_bundle_from_frame_features(
    features: np.ndarray,
    segments: Sequence[np.ndarray],
    prefix: str,
    repetition_min_gap_segments: int,
    repetition_threshold: float,
) -> tuple[Dict[str, float], np.ndarray]:
    segment_feats = segment_mean_features(features, segments)
    return (
        compute_metric_bundle_from_segment_features(
            segment_feats,
            prefix=prefix,
            repetition_min_gap_segments=repetition_min_gap_segments,
            repetition_threshold=repetition_threshold,
        ),
        segment_feats,
    )


def compute_image_feature_metric_bundle(
    frames: np.ndarray,
    segments: Sequence[np.ndarray],
    extractor: FeatureExtractor,
    prefix: str,
    repetition_min_gap_segments: int,
    repetition_threshold: float,
) -> tuple[Dict[str, float], np.ndarray]:
    features = extractor.encode_images(list(frames))
    return compute_metric_bundle_from_frame_features(
        features,
        segments=segments,
        prefix=prefix,
        repetition_min_gap_segments=repetition_min_gap_segments,
        repetition_threshold=repetition_threshold,
    )
