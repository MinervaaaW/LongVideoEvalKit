from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from longvideo_eval.cli import main


def _make_video(path: Path, frame_count: int, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 32))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((32, 32, 3), color + idx, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_paired_run_smoke(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    out_dir = tmp_path / "out"

    _make_video(gt_dir / "alpha_scene.mp4", frame_count=10, color=20)
    _make_video(pred_dir / "model_alpha_scene.mp4", frame_count=14, color=20)
    _make_video(gt_dir / "beta_clip.mp4", frame_count=12, color=40)
    _make_video(pred_dir / "beta_clip.avi", frame_count=12, color=40)

    exit_code = main(
        [
            "paired-run",
            "--gt-dir",
            str(gt_dir),
            "--pred-dir",
            str(pred_dir),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--skip-fvd",
            "--skip-lpips",
        ]
    )
    assert exit_code == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    final_results = json.loads((out_dir / "final_results.json").read_text())
    pairing_report = json.loads((out_dir / "pairing_report.json").read_text())

    assert summary["video_count"] == 2
    assert final_results["config"]["matched_video_count"] == 2
    assert final_results["config"]["non_exact_match_count"] == 2
    assert pairing_report["summary"]["non_exact_match_count"] == 2
    assert pairing_report["summary"]["unmatched_gt_count"] == 0
    assert pairing_report["summary"]["unmatched_pred_count"] == 0

