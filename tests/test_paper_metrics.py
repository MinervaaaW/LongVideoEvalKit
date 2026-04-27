from pathlib import Path

import numpy as np

from longvideo_eval.io.video_clipper import VideoClipSpec
from longvideo_eval.metrics.paper_metrics import (
    add_balance_to_summary_rows,
    build_named_quality_clip_specs,
    compute_clip_first_last_drift,
    compute_clip_global_repetition,
    compute_context_forcing_window_consistency,
)


def test_compute_clip_first_last_drift():
    feats = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ts = np.array([0.0, 1.0, 5.0, 6.0], dtype=np.float32)

    out = compute_clip_first_last_drift(feats, ts, clip_seconds=2.0)

    assert "paper.drift_clip_first_last" in out
    assert np.isclose(out["paper.drift_clip_first_last"], 1.0)


def test_compute_clip_global_repetition():
    feats = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ts = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    out = compute_clip_global_repetition(feats, ts, clip_seconds=1.0, max_clips=8)

    assert "paper.repetition_clip_global" in out
    assert out["paper.repetition_clip_global.num_pairs"] == 6.0


def test_compute_context_forcing_window_consistency():
    feats = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ts = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    out = compute_context_forcing_window_consistency(
        feats,
        ts,
        window_radius_seconds=0.5,
        prefix="paper.dinov2_cf",
    )

    assert set(out.keys()) == {"paper.dinov2_cf.mean", "paper.dinov2_cf.end", "paper.dinov2_cf.drop"}
    assert out["paper.dinov2_cf.end"] < out["paper.dinov2_cf.mean"]


def test_build_named_quality_clip_specs():
    specs = build_named_quality_clip_specs(
        base_name="demo",
        duration_sec=6.0,
        head_tail_seconds=5.0,
        segment_seconds=2.0,
        max_segments=30,
    )

    assert specs[0] == VideoClipSpec(name="demo__paper_head", start_sec=0.0, end_sec=5.0, tag="head")
    assert specs[1] == VideoClipSpec(name="demo__paper_tail", start_sec=1.0, end_sec=6.0, tag="tail")
    assert any(spec.tag == "seg_000" for spec in specs)


def test_add_balance_to_summary_rows():
    rows = [
        {"model": "a", "paper.drift_clip_first_last.mean": 0.2, "paper.repetition_clip_global.mean": 0.4},
        {"model": "b", "paper.drift_clip_first_last.mean": 0.5, "paper.repetition_clip_global.mean": 0.6},
    ]

    out = add_balance_to_summary_rows(rows)

    assert "paper.balance.mean" in out[0]
    assert out[0]["paper.balance.mean"] < out[1]["paper.balance.mean"]
