from __future__ import annotations

from pathlib import Path

import pytest

from longvideo_eval.config import load_config
from longvideo_eval.pipeline import _stage_vbench_inputs
from longvideo_eval.metrics.vbench_wrapper import (
    SUPPORTED_CUSTOM_INPUT_DIMENSIONS,
    VBenchExecutionError,
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


def test_vbench_runner_builds_custom_input_commands(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(VBenchRunner, "_preflight", lambda self: None)
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
        str((tmp_path / "videos").resolve()),
        "--dimension",
        "aesthetic_quality",
        "--mode",
        "custom_input",
        "--output_path",
        str((tmp_path / "out" / "aesthetic_quality").resolve()),
        "--load_ckpt_from_local",
        "True",
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


def test_parse_vbench_outputs_collects_summary_from_dimension_keyed_json(tmp_path: Path):
    summary_dir = tmp_path / "dynamic_degree"
    summary_dir.mkdir(parents=True)
    (summary_dir / "results.json").write_text(
        (
            '{"dynamic_degree": [1.0, [{"video_path": "/model_a/sample_01.mp4", "video_results": true}, '
            '{"video_path": "/model_a/sample_02.mp4", "video_results": true}]]}'
        ),
        encoding="utf-8",
    )

    results = parse_vbench_outputs(
        tmp_path,
        dimensions=["dynamic_degree"],
        model_name="model_a",
    )

    assert results.summary_rows == [{"model": "model_a", "vbench.dynamic_degree": 1.0}]
    assert results.per_video_rows == []


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
    monkeypatch.setattr(VBenchRunner, "_preflight", lambda self: None)

    def fake_run(cmd, check, text, capture_output, cwd, env=None):
        dim = cmd[cmd.index("--dimension") + 1]
        assert cmd[cmd.index("--output_path") + 1] == str(Path(cwd))
        out_dir = Path(cwd)
        (out_dir / "summary.json").write_text('{"score": 0.75}', encoding="utf-8")
        (out_dir / "per_video.jsonl").write_text(
            '{"video_path": "sample_02.mp4", "score": 0.8}\n',
            encoding="utf-8",
        )

        class Result:
            returncode = 0
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


def test_vbench_runner_retries_with_rewritten_socks5h_proxy(tmp_path: Path, monkeypatch):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    monkeypatch.setattr(VBenchRunner, "_preflight", lambda self: None)
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:1080")

    calls: list[dict[str, object]] = []

    def fake_run(cmd, check, text, capture_output, cwd, env=None):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        out_dir = Path(cwd)
        if len(calls) == 1:
            class Result:
                returncode = 1
                stdout = ""
                stderr = "Error parsing proxy URL socks5h://127.0.0.1:1080: Unsupported scheme 'socks5h'."

            return Result()

        (out_dir / "summary.json").write_text('{"score": 0.66}', encoding="utf-8")
        (out_dir / "per_video.jsonl").write_text(
            '{"video_path": "sample_03.mp4", "score": 0.7}\n',
            encoding="utf-8",
        )

        class Result:
            returncode = 0
            stdout = "retry ok"
            stderr = ""

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    _, results = run_and_collect_vbench(
        video_root=video_root,
        output_dir=tmp_path / "raw",
        dimensions=["motion_smoothness"],
        model_name="demo_model",
        command="vbench",
        mode="custom_input",
    )

    assert len(calls) == 2
    assert calls[0]["env"] is None
    assert calls[1]["env"]["ALL_PROXY"] == "socks5://127.0.0.1:1080"
    assert results.summary_rows == [{"model": "demo_model", "vbench.motion_smoothness": 0.66}]
    assert results.per_video_rows[0]["vbench.motion_smoothness"] == 0.7


def test_vbench_runner_error_reports_failure_reason_when_returncode_is_zero(tmp_path: Path, monkeypatch):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    monkeypatch.setattr(VBenchRunner, "_preflight", lambda self: None)

    def fake_run(cmd, check, text, capture_output, cwd, env=None):
        class Result:
            returncode = 0
            stdout = ""
            stderr = "Traceback: something went wrong"

        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(VBenchExecutionError, match=r"returncode=0, reason=failure markers in stdout/stderr"):
        run_and_collect_vbench(
            video_root=video_root,
            output_dir=tmp_path / "raw",
            dimensions=["motion_smoothness"],
            model_name="demo_model",
            command="vbench",
            mode="custom_input",
        )


def test_supported_custom_dimensions_constant_is_non_empty():
    assert "aesthetic_quality" in SUPPORTED_CUSTOM_INPUT_DIMENSIONS


def test_vbench_runner_preflight_raises_without_cuda(monkeypatch):
    monkeypatch.setattr("longvideo_eval.metrics.vbench_wrapper._torch_cuda_is_available", lambda: False)
    runner = VBenchRunner(command="vbench", mode="custom_input")
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        runner.run(
            video_root="videos",
            output_dir="out",
            dimensions=["aesthetic_quality"],
            dry_run=False,
        )


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


def test_stage_vbench_inputs_flattens_nested_prompt_dirs(tmp_path: Path):
    src_root = tmp_path / "outputs"
    video_a = src_root / "000001" / "sample_a.mp4"
    video_b = src_root / "000002" / "sample_b.mp4"
    video_a.parent.mkdir(parents=True)
    video_b.parent.mkdir(parents=True)
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    targets = _stage_vbench_inputs(
        [
            ("demo_model", "000001", video_a),
            ("demo_model", "000002", video_b),
        ],
        tmp_path / "staged",
    )

    assert targets == [("demo_model", tmp_path / "staged" / "demo_model")]
    staged_files = sorted((tmp_path / "staged" / "demo_model").iterdir(), key=lambda p: p.name)
    assert [p.name for p in staged_files] == ["sample_a.mp4", "sample_b.mp4"]
