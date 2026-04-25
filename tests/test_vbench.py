from __future__ import annotations

from pathlib import Path

import pytest

from longvideo_eval.config import load_config
from longvideo_eval.metrics.vbench_wrapper import (
    SUPPORTED_CUSTOM_INPUT_DIMENSIONS,
    VBenchRunner,
    merge_vbench_per_video_into_rows,
    merge_vbench_summary_into_model_summary,
    parse_vbench_outputs,
    run_and_collect_vbench,
)


def test_vbench_runner_rejects_unsupported_dimension(tmp_path: Path):
    runner = VBenchRunner(command="vbench", mode="custom_input")
    with pytest.raises(ValueError):
        runner.run(
            video_root=tmp_path / "videos",
            output_dir=tmp_path / "out",
            dimensions=["temporal_flickering"],
            dry_run=True,
        )


def test_vbench_runner_builds_custom_input_commands(tmp_path: Path):
    runner = VBenchRunner(command="vbench", mode="custom_input")
    invocations = runner.run(
        video_root=tmp_path / "videos",
        output_dir=tmp_path / "out",
        dimensions=["aesthetic_quality", "imaging_quality"],
        dry_run=True,
    )
    assert [item.dimension for item in invocations] == ["aesthetic_quality", "imaging_quality"]
    assert invocations[0].command == [
        "vbench",
        "evaluate",
        "--videos_path",
        str(tmp_path / "videos"),
        "--dimension",
        "aesthetic_quality",
        "--mode",
        "custom_input",
    ]
    assert invocations[0].cwd == tmp_path / "out" / "aesthetic_quality"


def test_parse_vbench_outputs_collects_summary_and_per_video(tmp_path: Path):
    summary_dir = tmp_path / "aesthetic_quality"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.csv").write_text("score\n0.87\n", encoding="utf-8")
    (summary_dir / "per_video.csv").write_text(
        "video_path,score\n/model_a/sample_01.mp4,0.91\n",
        encoding="utf-8",
    )

    results = parse_vbench_outputs(
        tmp_path,
        dimensions=["aesthetic_quality"],
        model_name="model_a",
    )

    assert results.summary_rows == [{"model": "model_a", "vbench.aesthetic_quality": 0.87}]
    assert results.per_video_rows == [
        {
            "model": "model_a",
            "video_path": "/model_a/sample_01.mp4",
            "video_name": "sample_01.mp4",
            "video_stem": "sample_01",
            "vbench.aesthetic_quality": 0.91,
        }
    ]


def test_merge_vbench_results_into_existing_reports():
    model_summary = [{"model": "model_a", "num_videos": 2, "clip_f.mean.mean": 0.9}]
    vbench_summary = [{"model": "model_a", "vbench.aesthetic_quality": 0.88}]
    merged_summary = merge_vbench_summary_into_model_summary(model_summary, vbench_summary)
    assert merged_summary == [
        {
            "model": "model_a",
            "num_videos": 2,
            "clip_f.mean.mean": 0.9,
            "vbench.aesthetic_quality": 0.88,
        }
    ]

    rows = [
        {
            "model": "model_a",
            "prompt_id": "000001",
            "video_path": "videos/model_a/sample_01.mp4",
        }
    ]
    vbench_per_video = [
        {
            "model": "model_a",
            "video_path": "/tmp/sample_01.mp4",
            "video_name": "sample_01.mp4",
            "video_stem": "sample_01",
            "vbench.aesthetic_quality": 0.91,
        }
    ]
    merged_rows = merge_vbench_per_video_into_rows(rows, vbench_per_video)
    assert merged_rows[0]["vbench.aesthetic_quality"] == 0.91


def test_run_and_collect_vbench_parses_fake_outputs(tmp_path: Path, monkeypatch):
    video_root = tmp_path / "videos"
    video_root.mkdir()

    def fake_run(cmd, check, text, capture_output, cwd):
        dim = cmd[cmd.index("--dimension") + 1]
        out_dir = Path(cwd)
        (out_dir / "summary.json").write_text('{"score": 0.75}', encoding="utf-8")
        (out_dir / "per_video.jsonl").write_text(
            '{"video_path": "sample_02.mp4", "score": 0.8}\n',
            encoding="utf-8",
        )

        class Result:
            stdout = f"ran {dim}"
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    invocations, results = run_and_collect_vbench(
        video_root=video_root,
        output_dir=tmp_path / "raw",
        dimensions=["imaging_quality"],
        model_name="demo_model",
        command="vbench",
        mode="custom_input",
    )

    assert len(invocations) == 1
    assert results.summary_rows == [{"model": "demo_model", "vbench.imaging_quality": 0.75}]
    assert results.per_video_rows[0]["vbench.imaging_quality"] == 0.8


def test_supported_custom_dimensions_constant_is_non_empty():
    assert "aesthetic_quality" in SUPPORTED_CUSTOM_INPUT_DIMENSIONS


def test_load_config_reads_vbench_section(tmp_path: Path):
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  video_root: ./videos",
                "  prompt_file: ./prompts.jsonl",
                "vbench:",
                "  enabled: true",
                "  dimensions: [aesthetic_quality, imaging_quality]",
                "  command: custom-vbench",
                "  mode: custom_input",
                "  raw_output_subdir: custom_raw",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.vbench.enabled is True
    assert cfg.vbench.dimensions == ["aesthetic_quality", "imaging_quality"]
    assert cfg.vbench.command == "custom-vbench"
    assert cfg.vbench.raw_output_subdir == "custom_raw"
