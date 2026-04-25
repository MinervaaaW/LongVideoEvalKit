from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from longvideo_eval.config import EvalConfig
from longvideo_eval.io.prompt_loader import load_prompts
from longvideo_eval.io.runtime_loader import load_runtime_sidecar
from longvideo_eval.io.video_reader import list_video_records, read_video_sampled, segment_indices
from longvideo_eval.metrics.efficiency import compute_efficiency
from longvideo_eval.metrics.long_consistency import (
    clip_t_from_segments,
    compute_image_feature_metric_bundle,
    segment_mean_features,
)
from longvideo_eval.metrics.quality_proxy import compute_quality_proxy
from longvideo_eval.metrics.vbench_wrapper import (
    VBenchMergedResults,
    merge_vbench_per_video_into_rows,
    merge_vbench_summary_into_model_summary,
    run_and_collect_vbench,
    write_vbench_merged_outputs,
)
from longvideo_eval.models.features import ColorHistExtractor, DINOv2Extractor, build_clip_extractor
from longvideo_eval.report.writer import summarize_by_model, write_csv, write_json, write_jsonl


def _build_coverage_rows(
    prompts: Dict[str, Any],
    raw_records: List[tuple[str, str, Path]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt_ids = sorted(prompts.keys())
    video_prompt_ids = sorted({prompt_id for _, prompt_id, _ in raw_records})
    missing_prompt_ids = sorted(set(prompt_ids) - set(video_prompt_ids))
    extra_video_prompt_ids = sorted(set(video_prompt_ids) - set(prompt_ids))
    missing_rows = [
        {
            "prompt_id": prompt_id,
            "prompt": prompts[prompt_id].prompt,
            "category": prompts[prompt_id].category or "",
            "status": "missing_video",
        }
        for prompt_id in missing_prompt_ids
    ]
    coverage = {
        "num_prompt_records": len(prompt_ids),
        "num_video_records": len(raw_records),
        "num_prompts_with_video": len(video_prompt_ids),
        "missing_prompt_ids": missing_prompt_ids,
        "orphan_video_prompt_ids": extra_video_prompt_ids,
    }
    return coverage, missing_rows


def _build_vbench_targets(
    cfg: EvalConfig,
    raw_records: List[tuple[str, str, Path]],
) -> list[tuple[str, Path]]:
    if cfg.dataset.layout == "prompt_dirs":
        model = cfg.dataset.model_name or cfg.dataset.video_root.name
        return [(model, cfg.dataset.video_root)]

    models = sorted({model for model, _, _ in raw_records})
    return [(model, cfg.dataset.video_root / model) for model in models]


def _run_vbench_and_merge(
    cfg: EvalConfig,
    raw_records: List[tuple[str, str, Path]],
    rows: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
    out_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    targets = _build_vbench_targets(cfg, raw_records)
    raw_root = out_dir / cfg.vbench.raw_output_subdir
    merged_summary_rows: list[dict[str, Any]] = []
    merged_per_video_rows: list[dict[str, Any]] = []

    for model, video_root in targets:
        model_out_dir = raw_root / model
        _, results = run_and_collect_vbench(
            video_root=video_root,
            output_dir=model_out_dir,
            dimensions=cfg.vbench.dimensions,
            model_name=model,
            command=cfg.vbench.command,
            mode=cfg.vbench.mode,
        )
        if not results.summary_rows and not results.per_video_rows:
            raise RuntimeError(
                f"VBench produced no parseable results for model={model} in {model_out_dir}"
            )
        merged_summary_rows.extend(results.summary_rows)
        merged_per_video_rows.extend(results.per_video_rows)

    write_vbench_merged_outputs(
        results=VBenchMergedResults(
            summary_rows=merged_summary_rows,
            per_video_rows=merged_per_video_rows,
        ),
        output_dir=out_dir,
    )
    rows = merge_vbench_per_video_into_rows(rows, merged_per_video_rows)
    summary = merge_vbench_summary_into_model_summary(summary, merged_summary_rows)
    return rows, summary


def run_eval(cfg: EvalConfig) -> List[Dict[str, Any]]:
    prompts = load_prompts(cfg.dataset.prompt_file, cfg.dataset.prompt_dir)
    runtime = load_runtime_sidecar(cfg.dataset.runtime_file)
    raw_records = list_video_records(
        cfg.dataset.video_root,
        layout=cfg.dataset.layout,
        model_name=cfg.dataset.model_name,
        video_selection=cfg.dataset.video_selection,
    )
    out_dir = cfg.report.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_records:
        raise RuntimeError(f"No videos found below {cfg.dataset.video_root}")

    coverage, missing_rows = _build_coverage_rows(prompts, raw_records)
    coverage.update(
        {
            "dataset_layout": cfg.dataset.layout,
            "model_name": cfg.dataset.model_name or cfg.dataset.video_root.name,
            "video_root": str(cfg.dataset.video_root),
            "video_selection": cfg.dataset.video_selection,
        }
    )
    print(
        "Dataset coverage: "
        f"prompts={coverage['num_prompt_records']}, "
        f"videos={coverage['num_video_records']}, "
        f"matched_prompts={coverage['num_prompts_with_video']}, "
        f"missing_prompts={len(coverage['missing_prompt_ids'])}"
    )
    write_json(coverage, out_dir / "dataset_coverage.json")
    if missing_rows:
        write_jsonl(missing_rows, out_dir / "missing_prompts.jsonl")

    colorhist = ColorHistExtractor() if cfg.metrics.colorhist else None

    clip = None
    if cfg.metrics.clip:
        clip = build_clip_extractor(
            backend=cfg.models.clip_backend,
            model_name=cfg.models.clip_model,
            pretrained=cfg.models.clip_pretrained,
            clip_repo=cfg.models.clip_repo,
            clip_cache_dir=cfg.models.clip_cache_dir,
        )

    dinov2 = None
    if cfg.metrics.dinov2:
        dinov2 = DINOv2Extractor(model_name=cfg.models.dinov2_model)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for model, prompt_id, path in tqdm(raw_records, desc="Evaluating videos"):
        prompt_rec = prompts.get(prompt_id)
        row: Dict[str, Any] = {
            "model": model,
            "prompt_id": prompt_id,
            "video_path": str(path),
            "prompt": prompt_rec.prompt if prompt_rec else "",
            "category": prompt_rec.category if prompt_rec else "",
        }
        try:
            frames, timestamps, meta = read_video_sampled(
                path,
                sample_fps=cfg.sampling.sample_fps,
            )
            segments = segment_indices(
                timestamps,
                segment_seconds=cfg.sampling.segment_seconds,
                max_segments=cfg.sampling.max_segments,
                max_frames_per_segment=cfg.sampling.max_frames_per_segment,
            )
            if not segments:
                raise RuntimeError("No non-empty segments after sampling")

            row.update(compute_efficiency(meta, runtime.get((model, prompt_id))))
            if cfg.metrics.quality_proxy:
                row.update(compute_quality_proxy(frames, segments))

            if colorhist is not None:
                metrics, _ = compute_image_feature_metric_bundle(
                    frames=frames,
                    segments=segments,
                    extractor=colorhist,
                    prefix="colorhist",
                    repetition_min_gap_segments=cfg.metrics.repetition_min_gap_segments,
                    repetition_threshold=cfg.metrics.repetition_threshold,
                )
                row.update(metrics)

            if clip is not None:
                clip_features = clip.encode_images(list(frames))
                clip_seg_feats = segment_mean_features(clip_features, segments)
                from longvideo_eval.metrics.long_consistency import long_consistency_from_features, drift_from_features, repetition_from_features

                row.update(long_consistency_from_features(clip_seg_feats, "clip_f"))
                row.update(drift_from_features(clip_seg_feats, "drift_clip"))
                row.update(
                    repetition_from_features(
                        clip_seg_feats,
                        "repetition_clip",
                        min_gap_segments=cfg.metrics.repetition_min_gap_segments,
                        threshold=cfg.metrics.repetition_threshold,
                    )
                )
                if prompt_rec is not None and prompt_rec.prompt:
                    text_feat = clip.encode_texts([prompt_rec.prompt])[0]
                    row.update(clip_t_from_segments(clip_seg_feats, text_feat, "clip_t"))

            if dinov2 is not None:
                metrics, _ = compute_image_feature_metric_bundle(
                    frames=frames,
                    segments=segments,
                    extractor=dinov2,
                    prefix="dinov2",
                    repetition_min_gap_segments=cfg.metrics.repetition_min_gap_segments,
                    repetition_threshold=cfg.metrics.repetition_threshold,
                )
                # For clarity, alias dinov2_lc as the canonical DINOv2 long consistency metric.
                for k, v in list(metrics.items()):
                    if k.startswith("dinov2_lc"):
                        row[k] = v
                    elif k.startswith("drift_dinov2"):
                        row[k] = v
                    elif k.startswith("repetition_dinov2"):
                        row[k] = v

            row["status"] = "ok"
            rows.append(row)
        except Exception as exc:  # continue batch evaluation
            row["status"] = "error"
            row["error"] = str(exc)
            errors.append({**row, "traceback": traceback.format_exc()})
            rows.append(row)

    summary = summarize_by_model([r for r in rows if r.get("status") == "ok"])
    if cfg.vbench.enabled:
        rows, summary = _run_vbench_and_merge(cfg, raw_records, rows, summary, out_dir)
    write_jsonl(rows, out_dir / "per_video_metrics.jsonl")
    write_csv(rows, out_dir / "per_video_metrics.csv")
    write_csv(summary, out_dir / "model_summary.csv")
    if errors:
        write_jsonl(errors, out_dir / "errors.jsonl")
    return rows
