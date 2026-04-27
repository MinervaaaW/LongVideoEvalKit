from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paired_videos_metrics.batch_eval_paired_videos import run_paired_eval


def run_paired_eval_from_cli(args) -> dict:
    output_dir = args.output_dir
    if output_dir is None:
        gt_name = Path(args.gt_dir).resolve().name
        pred_name = Path(args.pred_dir).resolve().name
        output_dir = str(Path("./outputs") / f"{pred_name}_vs_{gt_name}")

    paired_args = type(
        "PairedArgs",
        (),
        {
            "gt_dir": args.gt_dir,
            "pred_dir": args.pred_dir,
            "output_dir": output_dir,
            "fvd_method": args.fvd_method,
            "device": args.device,
            "max_videos": args.max_videos,
            "fvd_resolution": args.fvd_resolution,
            "skip_fvd": args.skip_fvd,
            "skip_lpips": args.skip_lpips,
        },
    )()
    return run_paired_eval(paired_args)
