from __future__ import annotations

import argparse
import sys
from pathlib import Path

from longvideo_eval.config import build_config_from_args
from longvideo_eval.metrics.vbench_wrapper import run_and_collect_vbench, write_vbench_merged_outputs
from longvideo_eval.pipeline import run_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="longvideo-eval", description="LongVideoEvalKit v0.1 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run v0.1 evaluation pipeline")
    run.add_argument("--config", default=None, help="YAML config path")
    run.add_argument("--video-root", default=None, help="Root directory containing model subdirectories")
    run.add_argument("--prompt-file", default=None, help="JSONL prompt file")
    run.add_argument("--prompt-dir", default=None, help="Directory containing one prompt text file per prompt_id")
    run.add_argument("--runtime-file", default=None, help="Optional runtime JSONL sidecar")
    run.add_argument("--output-dir", default=None, help="Output directory")
    run.add_argument("--dataset-layout", choices=["standard", "prompt_dirs"], default=None)
    run.add_argument("--model-name", default=None, help="Model name to use for single-model prompt_dirs layouts")
    run.add_argument("--video-selection", choices=["all", "latest"], default=None)
    run.add_argument("--sample-fps", type=float, default=None)
    run.add_argument("--segment-seconds", type=float, default=None)
    run.add_argument("--max-segments", type=int, default=None)
    run.add_argument("--max-frames-per-segment", type=int, default=8)
    run.add_argument("--disable-quality-proxy", action="store_true")
    run.add_argument("--disable-colorhist", action="store_true")
    run.add_argument("--enable-clip", action="store_true", help="Enable CLIP metrics")
    run.add_argument("--enable-dinov2", action="store_true", help="Enable DINOv2 metrics")
    run.add_argument("--clip-model", default=None)
    run.add_argument("--clip-pretrained", default=None)
    run.add_argument("--clip-backend", choices=["open_clip", "openai_clip"], default=None)
    run.add_argument("--clip-repo", default=None, help="Optional local CLIP repo path for openai_clip backend")
    run.add_argument("--clip-cache-dir", default=None, help="Local cache dir for OpenAI CLIP weights")
    run.add_argument("--dinov2-model", default=None)
    run.add_argument("--repetition-min-gap-segments", type=int, default=5)
    run.add_argument("--repetition-threshold", type=float, default=0.95)
    run.add_argument("--enable-vbench", action="store_true", help="Run official VBench and merge results into reports")
    run.add_argument("--vbench-dimensions", nargs="*", default=None, help="Official VBench custom_input dimensions to run")
    run.add_argument("--vbench-command", default=None, help="Command used to invoke official VBench")
    run.add_argument("--vbench-mode", default=None, help="VBench mode, currently only custom_input is supported")
    run.add_argument("--vbench-raw-output-subdir", default=None, help="Subdirectory below output-dir for raw VBench artifacts")

    vbench = sub.add_parser("vbench", help="Run official VBench custom-input evaluation")
    vbench.add_argument("--video-root", required=True, help="Video root passed to VBench")
    vbench.add_argument("--output-dir", required=True, help="Directory for raw and merged VBench outputs")
    vbench.add_argument("--dimensions", nargs="+", required=True, help="Official VBench custom_input dimensions to run")
    vbench.add_argument("--vbench-command", default="vbench", help="Command used to invoke official VBench")
    vbench.add_argument("--mode", default="custom_input", help="VBench mode, currently only custom_input is supported")
    vbench.add_argument("--model-name", default=None, help="Model name used in merged outputs")
    vbench.add_argument("--dry-run", action="store_true", help="Print the VBench commands without executing them")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            cfg = build_config_from_args(args)
            rows = run_eval(cfg)
            ok = sum(1 for r in rows if r.get("status") == "ok")
            err = len(rows) - ok
            print(f"Done. ok={ok}, errors={err}, output_dir={cfg.report.output_dir}")
            return 0 if ok > 0 else 2
        if args.command == "vbench":
            model_name = args.model_name or Path(args.video_root).name
            invocations, results = run_and_collect_vbench(
                video_root=args.video_root,
                output_dir=args.output_dir,
                dimensions=args.dimensions,
                model_name=model_name,
                command=args.vbench_command,
                mode=args.mode,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                for item in invocations:
                    print(" ".join(item.command))
                return 0
            write_vbench_merged_outputs(results, args.output_dir)
            print(
                f"Done. dimensions={len(args.dimensions)}, "
                f"summary_rows={len(results.summary_rows)}, "
                f"per_video_rows={len(results.per_video_rows)}, "
                f"output_dir={args.output_dir}"
            )
            return 0
        parser.error(f"Unknown command: {args.command}")
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
