#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import lpips  # noqa: E402
from calculate_psnr import img_psnr  # noqa: E402
from calculate_ssim import calculate_ssim_function  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class PairMetrics:
    name: str
    gt_path: str
    pred_path: str
    match_type: str
    matched_suffix_chars: int
    frame_count: int
    gt_frame_count_raw: int
    pred_frame_count_raw: int
    gt_fps: float
    pred_fps: float
    gt_duration_sec: float
    pred_duration_sec: float
    aligned_duration_sec: float
    duration_truncated: bool
    gt_width: int
    gt_height: int
    pred_width: int
    pred_height: int
    psnr_mean: float
    ssim_mean: float
    lpips_mean: float
    fvd_pair: float


@dataclass
class VideoPair:
    name: str
    gt_path: Path
    pred_path: Path
    match_type: str
    matched_suffix_chars: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate paired videos with FVD, SSIM, PSNR, and LPIPS."
    )
    parser.add_argument("--gt-dir", required=True, help="Directory of ground-truth videos.")
    parser.add_argument("--pred-dir", required=True, help="Directory of generated videos.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs. Defaults to outputs/<pred>_vs_<gt>.",
    )
    parser.add_argument(
        "--fvd-method",
        default="styleganv",
        choices=("styleganv", "videogpt"),
        help="PyTorch FVD backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for FVD and LPIPS.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optionally evaluate only the first N matched videos.",
    )
    parser.add_argument(
        "--fvd-resolution",
        type=int,
        default=224,
        help="Resolution used for cached FVD frames before feature extraction.",
    )
    parser.add_argument(
        "--skip-fvd",
        action="store_true",
        help="Skip FVD computation.",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS computation.",
    )
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_videos(directory: Path) -> Dict[str, Path]:
    videos = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos[path.name] = path
    return videos


def normalized_stem(path: Path) -> str:
    return path.stem.lower()


def suffix_match_chars(left: str, right: str) -> int:
    matched = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        matched += 1
    return matched


def suffix_match_min_chars(left: str, right: str) -> int:
    return min(8, len(left), len(right))


def pair_to_row(pair: VideoPair) -> dict:
    return {
        "name": pair.name,
        "gt_path": str(pair.gt_path),
        "pred_path": str(pair.pred_path),
        "match_type": pair.match_type,
        "matched_suffix_chars": pair.matched_suffix_chars,
        "gt_stem": pair.gt_path.stem,
        "pred_stem": pair.pred_path.stem,
    }


def match_video_pairs(
    gt_dir: Path, pred_dir: Path
) -> Tuple[List[VideoPair], List[dict], List[dict], List[dict]]:
    gt_videos = list_videos(gt_dir)
    pred_videos = list_videos(pred_dir)
    pairs: List[VideoPair] = []
    non_exact_rows: List[dict] = []

    remaining_gt = dict(gt_videos)
    remaining_pred = dict(pred_videos)

    for name in sorted(set(gt_videos) & set(pred_videos)):
        pair = VideoPair(
            name=name,
            gt_path=gt_videos[name],
            pred_path=pred_videos[name],
            match_type="exact_filename",
            matched_suffix_chars=len(gt_videos[name].stem),
        )
        pairs.append(pair)
        remaining_gt.pop(name, None)
        remaining_pred.pop(name, None)

    gt_stem_index: Dict[str, List[Path]] = {}
    for path in remaining_gt.values():
        gt_stem_index.setdefault(normalized_stem(path), []).append(path)

    pred_stem_index: Dict[str, List[Path]] = {}
    for path in remaining_pred.values():
        pred_stem_index.setdefault(normalized_stem(path), []).append(path)

    for stem in sorted(set(gt_stem_index) & set(pred_stem_index)):
        gt_paths = sorted(gt_stem_index[stem])
        pred_paths = sorted(pred_stem_index[stem])
        while gt_paths and pred_paths:
            gt_path = gt_paths.pop(0)
            pred_path = pred_paths.pop(0)
            pair = VideoPair(
                name=gt_path.name,
                gt_path=gt_path,
                pred_path=pred_path,
                match_type="exact_stem",
                matched_suffix_chars=len(stem),
            )
            pairs.append(pair)
            non_exact_rows.append(pair_to_row(pair))
            remaining_gt.pop(gt_path.name, None)
            remaining_pred.pop(pred_path.name, None)

    candidate_matches: List[Tuple[int, float, str, str]] = []
    for gt_name, gt_path in remaining_gt.items():
        gt_stem = normalized_stem(gt_path)
        for pred_name, pred_path in remaining_pred.items():
            pred_stem = normalized_stem(pred_path)
            matched = suffix_match_chars(gt_stem, pred_stem)
            if matched < suffix_match_min_chars(gt_stem, pred_stem):
                continue
            ratio = matched / max(len(gt_stem), len(pred_stem))
            candidate_matches.append((matched, ratio, gt_name, pred_name))

    matched_gt_names = set()
    matched_pred_names = set()
    for matched, ratio, gt_name, pred_name in sorted(
        candidate_matches,
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    ):
        if gt_name in matched_gt_names or pred_name in matched_pred_names:
            continue
        gt_path = remaining_gt[gt_name]
        pred_path = remaining_pred[pred_name]
        pair = VideoPair(
            name=gt_path.name,
            gt_path=gt_path,
            pred_path=pred_path,
            match_type="suffix_stem_match",
            matched_suffix_chars=matched,
        )
        pairs.append(pair)
        non_exact_rows.append(pair_to_row(pair))
        matched_gt_names.add(gt_name)
        matched_pred_names.add(pred_name)

    for gt_name in matched_gt_names:
        remaining_gt.pop(gt_name, None)
    for pred_name in matched_pred_names:
        remaining_pred.pop(pred_name, None)

    pairs.sort(key=lambda pair: pair.name)
    non_exact_rows.sort(key=lambda row: row["name"])
    unmatched_gt_rows = [
        {"name": name, "path": str(path), "stem": path.stem}
        for name, path in sorted(remaining_gt.items())
    ]
    unmatched_pred_rows = [
        {"name": name, "path": str(path), "stem": path.stem}
        for name, path in sorted(remaining_pred.items())
    ]
    return pairs, non_exact_rows, unmatched_gt_rows, unmatched_pred_rows


def get_video_meta(path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    fps = meta["fps"]
    if fps > 0:
        meta["duration_sec"] = float(meta["frame_count"] / fps)
    else:
        meta["duration_sec"] = 0.0
    return meta


def resize_with_short_side(frame_rgb: np.ndarray, resolution: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    scale = resolution / min(h, w)
    if h < w:
        new_h, new_w = resolution, int(np.ceil(w * scale))
    else:
        new_h, new_w = int(np.ceil(h * scale)), resolution
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (new_h - resolution) // 2
    left = (new_w - resolution) // 2
    return resized[top : top + resolution, left : left + resolution]


def frame_to_chw_float(frame_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0


def build_lpips_model(device: torch.device):
    warnings.filterwarnings("ignore", message=".*weights.*")
    model = lpips.LPIPS(net="alex", spatial=True)
    model = model.to(device)
    model.eval()
    return model


def load_fvd_backend(method: str, device: torch.device):
    if method == "styleganv":
        from fvd.styleganv.fvd import (  # noqa: E402
            frechet_distance,
            get_fvd_feats,
            load_i3d_pretrained,
        )
    else:
        from fvd.videogpt.fvd import (  # noqa: E402
            frechet_distance,
            get_fvd_logits as get_fvd_feats,
            load_i3d_pretrained,
        )

    return load_i3d_pretrained(device=device), get_fvd_feats, frechet_distance


def evaluate_pair(
    pair: VideoPair,
    lpips_model,
    eval_device: torch.device,
    fvd_i3d,
    get_fvd_feats,
    frechet_distance,
    fvd_resolution: int,
) -> Tuple[PairMetrics, np.ndarray, np.ndarray]:
    name = pair.name
    gt_path = pair.gt_path
    pred_path = pair.pred_path
    gt_meta = get_video_meta(gt_path)
    pred_meta = get_video_meta(pred_path)

    gt_cap = cv2.VideoCapture(str(gt_path))
    pred_cap = cv2.VideoCapture(str(pred_path))
    if not gt_cap.isOpened() or not pred_cap.isOpened():
        raise RuntimeError(f"Failed to open paired videos: {gt_path} | {pred_path}")

    gt_duration_sec = gt_meta["duration_sec"]
    pred_duration_sec = pred_meta["duration_sec"]
    target_duration_sec = min(gt_duration_sec, pred_duration_sec)
    duration_truncated = abs(gt_duration_sec - pred_duration_sec) > 1e-6

    def frames_for_duration(meta: Dict[str, float], duration_sec: float) -> int:
        fps = meta["fps"]
        raw_count = meta["frame_count"]
        if fps <= 0 or duration_sec <= 0:
            return raw_count
        return min(raw_count, max(1, int(round(duration_sec * fps))))

    gt_target_frames = frames_for_duration(gt_meta, target_duration_sec)
    pred_target_frames = frames_for_duration(pred_meta, target_duration_sec)
    frame_count = min(gt_target_frames, pred_target_frames)
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []
    gt_fvd_frames: List[torch.Tensor] = []
    pred_fvd_frames: List[torch.Tensor] = []

    for _ in range(frame_count):
        ok_gt, gt_bgr = gt_cap.read()
        ok_pred, pred_bgr = pred_cap.read()
        if not ok_gt or not ok_pred:
            break

        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

        if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
            pred_rgb = cv2.resize(
                pred_rgb,
                (gt_rgb.shape[1], gt_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        gt_chw = np.transpose(gt_rgb.astype(np.float32) / 255.0, (2, 0, 1))
        pred_chw = np.transpose(pred_rgb.astype(np.float32) / 255.0, (2, 0, 1))

        psnr_values.append(float(img_psnr(gt_chw, pred_chw)))
        ssim_values.append(float(calculate_ssim_function(gt_chw, pred_chw)))

        if lpips_model is not None:
            gt_tensor = torch.from_numpy(gt_chw).unsqueeze(0).to(eval_device)
            pred_tensor = torch.from_numpy(pred_chw).unsqueeze(0).to(eval_device)
            gt_tensor = gt_tensor * 2 - 1
            pred_tensor = pred_tensor * 2 - 1
            with torch.no_grad():
                lpips_score = lpips_model.forward(gt_tensor, pred_tensor).mean().item()
            lpips_values.append(float(lpips_score))

        if fvd_i3d is not None:
            gt_fvd_frame = resize_with_short_side(gt_rgb, fvd_resolution)
            pred_fvd_frame = resize_with_short_side(pred_rgb, fvd_resolution)
            gt_fvd_frames.append(frame_to_chw_float(gt_fvd_frame))
            pred_fvd_frames.append(frame_to_chw_float(pred_fvd_frame))

    gt_cap.release()
    pred_cap.release()

    frame_count = len(psnr_values)
    if frame_count == 0:
        raise RuntimeError(f"No aligned frames found for pair: {name}")

    pair_fvd = None
    gt_feat = None
    pred_feat = None
    if fvd_i3d is not None:
        if len(gt_fvd_frames) < 10:
            raise ValueError(
                f"FVD requires at least 10 frames, but pair {name} only has {len(gt_fvd_frames)}."
            )
        gt_video = torch.stack(gt_fvd_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        pred_video = torch.stack(pred_fvd_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        gt_feat = get_fvd_feats(gt_video, i3d=fvd_i3d, device=eval_device)
        pred_feat = get_fvd_feats(pred_video, i3d=fvd_i3d, device=eval_device)
        pair_fvd = float(frechet_distance(pred_feat, gt_feat))

    metrics = PairMetrics(
        name=name,
        gt_path=str(gt_path),
        pred_path=str(pred_path),
        match_type=pair.match_type,
        matched_suffix_chars=pair.matched_suffix_chars,
        frame_count=frame_count,
        gt_frame_count_raw=gt_meta["frame_count"],
        pred_frame_count_raw=pred_meta["frame_count"],
        gt_fps=gt_meta["fps"],
        pred_fps=pred_meta["fps"],
        gt_duration_sec=gt_duration_sec,
        pred_duration_sec=pred_duration_sec,
        aligned_duration_sec=target_duration_sec,
        duration_truncated=duration_truncated,
        gt_width=gt_meta["width"],
        gt_height=gt_meta["height"],
        pred_width=pred_meta["width"],
        pred_height=pred_meta["height"],
        psnr_mean=float(np.mean(psnr_values)),
        ssim_mean=float(np.mean(ssim_values)),
        lpips_mean=float(np.mean(lpips_values)) if lpips_values else float("nan"),
        fvd_pair=pair_fvd if pair_fvd is not None else float("nan"),
    )
    return metrics, gt_feat, pred_feat


def save_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def save_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean(values: List[float], weights: List[int]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return float("nan")
    return float(sum(v * w for v, w in zip(values, weights)) / total_weight)


def build_pairing_report_rows(
    non_exact_rows: List[dict],
    unmatched_gt_rows: List[dict],
    unmatched_pred_rows: List[dict],
) -> List[dict]:
    rows: List[dict] = []
    for row in non_exact_rows:
        rows.append(
            {
                "record_type": "non_exact_match",
                "name": row.get("name", ""),
                "match_type": row.get("match_type", ""),
                "matched_suffix_chars": row.get("matched_suffix_chars", ""),
                "gt_path": row.get("gt_path", ""),
                "pred_path": row.get("pred_path", ""),
                "gt_stem": row.get("gt_stem", ""),
                "pred_stem": row.get("pred_stem", ""),
                "path": "",
                "stem": "",
            }
        )
    for row in unmatched_gt_rows:
        rows.append(
            {
                "record_type": "unmatched_gt",
                "name": row.get("name", ""),
                "match_type": "",
                "matched_suffix_chars": "",
                "gt_path": row.get("path", ""),
                "pred_path": "",
                "gt_stem": row.get("stem", ""),
                "pred_stem": "",
                "path": row.get("path", ""),
                "stem": row.get("stem", ""),
            }
        )
    for row in unmatched_pred_rows:
        rows.append(
            {
                "record_type": "unmatched_pred",
                "name": row.get("name", ""),
                "match_type": "",
                "matched_suffix_chars": "",
                "gt_path": "",
                "pred_path": row.get("path", ""),
                "gt_stem": "",
                "pred_stem": row.get("stem", ""),
                "path": row.get("path", ""),
                "stem": row.get("stem", ""),
            }
        )
    return rows


def run_paired_eval(args: argparse.Namespace) -> dict:
    gt_dir = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"GT dir not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Pred dir not found: {pred_dir}")

    if args.output_dir is None:
        output_dir = REPO_ROOT / "outputs" / f"{pred_dir.name}_vs_{gt_dir.name}"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    pairs, non_exact_rows, unmatched_gt_rows, unmatched_pred_rows = match_video_pairs(gt_dir, pred_dir)
    if args.max_videos is not None:
        pairs = pairs[: args.max_videos]
        kept_names = {pair.name for pair in pairs}
        non_exact_rows = [
            row for row in non_exact_rows if row["name"] in kept_names
        ]
        unmatched_gt_rows = [row for row in unmatched_gt_rows if row["name"] not in kept_names]
    if not pairs:
        raise ValueError("No matched videos found.")

    lpips_model = None if args.skip_lpips else build_lpips_model(device)
    fvd_i3d = None
    get_fvd_feats = None
    frechet_distance = None
    if not args.skip_fvd:
        fvd_i3d, get_fvd_feats, frechet_distance = load_fvd_backend(args.fvd_method, device)

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "fvd_method": args.fvd_method,
        "fvd_resolution": args.fvd_resolution,
        "skip_fvd": args.skip_fvd,
        "skip_lpips": args.skip_lpips,
        "matched_video_count": len(pairs),
        "non_exact_match_count": len(non_exact_rows),
        "unmatched_gt_count": len(unmatched_gt_rows),
        "unmatched_pred_count": len(unmatched_pred_rows),
    }
    save_json(output_dir / "config.json", config)
    pairing_report = {
        "summary": {
            "non_exact_match_count": len(non_exact_rows),
            "unmatched_gt_count": len(unmatched_gt_rows),
            "unmatched_pred_count": len(unmatched_pred_rows),
        },
        "non_exact_match_pairs": non_exact_rows,
        "unmatched_gt_videos": unmatched_gt_rows,
        "unmatched_pred_videos": unmatched_pred_rows,
    }
    save_json(output_dir / "pairing_report.json", pairing_report)
    save_csv(
        output_dir / "pairing_report.csv",
        build_pairing_report_rows(non_exact_rows, unmatched_gt_rows, unmatched_pred_rows),
    )

    per_video_rows: List[dict] = []
    gt_features: List = []
    pred_features: List = []

    for pair in tqdm(pairs, desc="Evaluating pairs"):
        pair_metrics, gt_feat, pred_feat = evaluate_pair(
            pair=pair,
            lpips_model=lpips_model,
            eval_device=device,
            fvd_i3d=fvd_i3d,
            get_fvd_feats=get_fvd_feats,
            frechet_distance=frechet_distance,
            fvd_resolution=args.fvd_resolution,
        )
        row = {
            "name": pair_metrics.name,
            "gt_path": pair_metrics.gt_path,
            "pred_path": pair_metrics.pred_path,
            "match_type": pair_metrics.match_type,
            "matched_suffix_chars": pair_metrics.matched_suffix_chars,
            "frame_count": pair_metrics.frame_count,
            "gt_frame_count_raw": pair_metrics.gt_frame_count_raw,
            "pred_frame_count_raw": pair_metrics.pred_frame_count_raw,
            "gt_fps": pair_metrics.gt_fps,
            "pred_fps": pair_metrics.pred_fps,
            "gt_duration_sec": pair_metrics.gt_duration_sec,
            "pred_duration_sec": pair_metrics.pred_duration_sec,
            "aligned_duration_sec": pair_metrics.aligned_duration_sec,
            "duration_truncated": pair_metrics.duration_truncated,
            "gt_width": pair_metrics.gt_width,
            "gt_height": pair_metrics.gt_height,
            "pred_width": pair_metrics.pred_width,
            "pred_height": pair_metrics.pred_height,
            "psnr_mean": pair_metrics.psnr_mean,
            "ssim_mean": pair_metrics.ssim_mean,
            "lpips_mean": pair_metrics.lpips_mean,
            "fvd_pair": pair_metrics.fvd_pair,
        }
        per_video_rows.append(row)

        if gt_feat is not None and pred_feat is not None:
            gt_features.append(gt_feat)
            pred_features.append(pred_feat)

    frame_weights = [row["frame_count"] for row in per_video_rows]
    summary = {
        "video_count": len(per_video_rows),
        "total_frames": int(sum(frame_weights)),
        "psnr_mean": weighted_mean([row["psnr_mean"] for row in per_video_rows], frame_weights),
        "ssim_mean": weighted_mean([row["ssim_mean"] for row in per_video_rows], frame_weights),
        "lpips_mean": weighted_mean([row["lpips_mean"] for row in per_video_rows], frame_weights),
    }

    if gt_features and pred_features:
        if args.fvd_method == "styleganv":
            gt_feat_all = np.vstack(gt_features)
            pred_feat_all = np.vstack(pred_features)
        else:
            gt_feat_all = torch.cat(gt_features, dim=0)
            pred_feat_all = torch.cat(pred_features, dim=0)
        summary["fvd_dataset"] = float(frechet_distance(pred_feat_all, gt_feat_all))
    else:
        summary["fvd_dataset"] = float("nan")

    save_jsonl(output_dir / "per_video_metrics.jsonl", per_video_rows)
    save_csv(output_dir / "per_video_metrics.csv", per_video_rows)
    save_json(output_dir / "summary.json", summary)
    save_json(
        output_dir / "final_results.json",
        {
            "config": config,
            "summary": summary,
            "non_exact_match_pairs": non_exact_rows,
            "unmatched_gt_videos": unmatched_gt_rows,
            "unmatched_pred_videos": unmatched_pred_rows,
            "per_video_metrics": per_video_rows,
        },
    )

    result = {"config": config, "summary": summary}
    print(json.dumps(result, indent=2))
    return result


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    run_paired_eval(args)


if __name__ == "__main__":
    main()
