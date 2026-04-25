from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence

from longvideo_eval.report.writer import write_csv, write_jsonl

SUPPORTED_CUSTOM_INPUT_DIMENSIONS = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


@dataclass
class VBenchInvocation:
    dimension: str
    command: list[str]
    cwd: Path
    stdout: str = ""
    stderr: str = ""


@dataclass
class VBenchMergedResults:
    summary_rows: list[dict[str, Any]]
    per_video_rows: list[dict[str, Any]]


def _normalize_path_text(value: str) -> str:
    return str(Path(value)).replace("\\", "/")


def _path_name(value: str | None) -> str | None:
    if not value:
        return None
    return Path(value).name


def _path_stem(value: str | None) -> str | None:
    if not value:
        return None
    return Path(value).stem


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _looks_like_video_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    suffix = Path(text).suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return True
    return any(token in text.lower() for token in ("video", "videos")) and "/" in text


def _flatten_records(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        scalar_items = {str(k): v for k, v in payload.items() if not isinstance(v, (dict, list))}
        if scalar_items:
            yield scalar_items
        for value in payload.values():
            if isinstance(value, (dict, list)):
                yield from _flatten_records(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _flatten_records(item)


def _read_structured_rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield from _flatten_records(payload)
        return
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                return
        yield from _flatten_records(payload)


def _extract_score(record: dict[str, Any], dimension: str) -> float | None:
    lower_map = {str(k).strip().lower(): v for k, v in record.items()}
    preferred = [
        dimension,
        f"vbench.{dimension}",
        f"{dimension}_score",
        f"{dimension}.score",
        "score",
        "value",
        "result",
        "mean",
        "avg",
        "average",
    ]
    for key in preferred:
        if key.lower() in lower_map:
            score = _parse_float(lower_map[key.lower()])
            if score is not None:
                return score

    scored_keys = []
    for key, value in lower_map.items():
        score = _parse_float(value)
        if score is None:
            continue
        if dimension.lower() in key or "score" in key or "result" in key or "value" in key:
            scored_keys.append(score)
    if scored_keys:
        return scored_keys[0]

    numeric_values = [_parse_float(value) for value in record.values()]
    numeric_values = [value for value in numeric_values if value is not None]
    if len(numeric_values) == 1:
        return numeric_values[0]
    return None


def _extract_video_info(record: dict[str, Any]) -> dict[str, str]:
    candidates = [
        "video_path",
        "path",
        "file_path",
        "filepath",
        "filename",
        "file",
        "video_file",
        "video",
        "videos_path",
    ]
    lower_map = {str(k).strip().lower(): v for k, v in record.items()}

    path_value: str | None = None
    for key in candidates:
        raw = _string_value(lower_map.get(key))
        if raw and _looks_like_video_path(raw):
            path_value = _normalize_path_text(raw)
            break

    if path_value is None:
        for value in lower_map.values():
            raw = _string_value(value)
            if raw and _looks_like_video_path(raw):
                path_value = _normalize_path_text(raw)
                break

    info: dict[str, str] = {}
    if path_value is not None:
        info["video_path"] = path_value
        info["video_name"] = _path_name(path_value) or path_value
        info["video_stem"] = _path_stem(path_value) or path_value
        return info

    for key in ("video_name", "filename", "file", "video"):
        raw = _string_value(lower_map.get(key))
        if raw and Path(raw).suffix.lower() in VIDEO_SUFFIXES:
            info["video_name"] = Path(raw).name
            info["video_stem"] = Path(raw).stem
            return info
    return info


def _aggregate_summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        model = str(row["model"])
        dimension = str(row["dimension"])
        score = float(row["score"])
        grouped.setdefault((model, dimension), []).append(score)

    merged: dict[str, dict[str, Any]] = {}
    for (model, dimension), values in grouped.items():
        merged.setdefault(model, {"model": model})[f"vbench.{dimension}"] = sum(values) / len(values)
    return [merged[key] for key in sorted(merged.keys())]


def _aggregate_per_video_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        model = str(row["model"])
        if row.get("video_path"):
            key = ("video_path", model, str(row["video_path"]))
        elif row.get("video_name"):
            key = ("video_name", model, str(row["video_name"]))
        else:
            key = ("video_stem", model, str(row["video_stem"]))

        bucket = grouped.setdefault(
            key,
            {
                "model": model,
                "video_path": row.get("video_path", ""),
                "video_name": row.get("video_name", ""),
                "video_stem": row.get("video_stem", ""),
                "_scores": {},
            },
        )
        bucket_scores = bucket["_scores"]
        bucket_scores.setdefault(str(row["dimension"]), []).append(float(row["score"]))

    merged_rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        merged = {
            "model": bucket["model"],
            "video_path": bucket["video_path"],
            "video_name": bucket["video_name"],
            "video_stem": bucket["video_stem"],
        }
        for dimension, values in sorted(bucket["_scores"].items()):
            merged[f"vbench.{dimension}"] = sum(values) / len(values)
        merged_rows.append(merged)

    return sorted(
        merged_rows,
        key=lambda row: (
            str(row.get("model", "")),
            str(row.get("video_path", "")),
            str(row.get("video_name", "")),
            str(row.get("video_stem", "")),
        ),
    )


def merge_vbench_summary_into_model_summary(
    model_summary_rows: Sequence[dict[str, Any]],
    vbench_summary_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(row) for row in model_summary_rows]
    by_model = {str(row.get("model")): row for row in merged}
    for vbench_row in vbench_summary_rows:
        model = str(vbench_row.get("model"))
        target = by_model.get(model)
        if target is None:
            target = {"model": model}
            by_model[model] = target
            merged.append(target)
        for key, value in vbench_row.items():
            if key == "model":
                continue
            target[key] = value
    return sorted(merged, key=lambda row: str(row.get("model", "")))


def merge_vbench_per_video_into_rows(
    rows: Sequence[dict[str, Any]],
    vbench_per_video_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(row) for row in rows]
    indices: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in merged:
        model = str(row.get("model", ""))
        video_path = _normalize_path_text(str(row.get("video_path", ""))) if row.get("video_path") else ""
        keys = []
        if video_path:
            keys.append(("video_path", model, video_path))
            keys.append(("video_name", model, Path(video_path).name))
            keys.append(("video_stem", model, Path(video_path).stem))
        for key in keys:
            indices.setdefault(key, []).append(row)

    for vbench_row in vbench_per_video_rows:
        model = str(vbench_row.get("model", ""))
        candidates = []
        if vbench_row.get("video_path"):
            video_path = _normalize_path_text(str(vbench_row["video_path"]))
            candidates.append(("video_path", model, video_path))
            candidates.append(("video_name", model, Path(video_path).name))
            candidates.append(("video_stem", model, Path(video_path).stem))
        if vbench_row.get("video_name"):
            candidates.append(("video_name", model, str(vbench_row["video_name"])))
        if vbench_row.get("video_stem"):
            candidates.append(("video_stem", model, str(vbench_row["video_stem"])))

        target: dict[str, Any] | None = None
        for candidate in candidates:
            matches = indices.get(candidate, [])
            if len(matches) == 1:
                target = matches[0]
                break
        if target is None:
            continue

        for key, value in vbench_row.items():
            if key in {"model", "video_path", "video_name", "video_stem"}:
                continue
            target[key] = value
    return merged


def build_summary_from_per_video_rows(per_video_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, list[float]]] = {}
    for row in per_video_rows:
        model = str(row.get("model", ""))
        bucket = by_model.setdefault(model, {})
        for key, value in row.items():
            if not key.startswith("vbench."):
                continue
            score = _parse_float(value)
            if score is None:
                continue
            bucket.setdefault(key, []).append(score)

    summary_rows: list[dict[str, Any]] = []
    for model, columns in sorted(by_model.items()):
        row: dict[str, Any] = {"model": model}
        for key, values in sorted(columns.items()):
            row[key] = sum(values) / len(values)
        summary_rows.append(row)
    return summary_rows


def parse_vbench_outputs(
    raw_output_dir: str | Path,
    dimensions: Sequence[str],
    model_name: str,
) -> VBenchMergedResults:
    root = Path(raw_output_dir)
    summary_candidates: list[dict[str, Any]] = []
    per_video_candidates: list[dict[str, Any]] = []

    for dimension in dimensions:
        dim_dir = root / dimension
        if not dim_dir.exists():
            continue
        for path in sorted(dim_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
                continue
            for record in _read_structured_rows(path):
                score = _extract_score(record, dimension)
                if score is None:
                    continue
                model = _string_value(record.get("model")) or model_name
                video_info = _extract_video_info(record)
                if video_info:
                    per_video_candidates.append(
                        {
                            "model": model,
                            "dimension": dimension,
                            "score": score,
                            **video_info,
                        }
                    )
                else:
                    summary_candidates.append(
                        {
                            "model": model,
                            "dimension": dimension,
                            "score": score,
                        }
                    )

    per_video_rows = _aggregate_per_video_rows(per_video_candidates)
    summary_rows = _aggregate_summary_rows(summary_candidates)
    if not summary_rows and per_video_rows:
        summary_rows = build_summary_from_per_video_rows(per_video_rows)
    return VBenchMergedResults(summary_rows=summary_rows, per_video_rows=per_video_rows)


class VBenchRunner:
    """Thin wrapper around official VBench custom-input evaluation."""

    def __init__(self, command: str = "vbench", mode: str = "custom_input") -> None:
        self.command = command
        self.mode = mode

    def _validate_dimensions(self, dimensions: Sequence[str]) -> None:
        if self.mode != "custom_input":
            raise ValueError(f"Unsupported VBench mode: {self.mode}")
        invalid = [dim for dim in dimensions if dim not in SUPPORTED_CUSTOM_INPUT_DIMENSIONS]
        if invalid:
            raise ValueError(
                "Unsupported VBench custom_input dimensions: "
                + ", ".join(sorted(invalid))
            )

    def build_command(
        self,
        video_root: str | Path,
        dimension: str,
        extra_args: Sequence[str] | None = None,
    ) -> list[str]:
        cmd: List[str] = [
            self.command,
            "evaluate",
            "--videos_path",
            str(video_root),
            "--dimension",
            dimension,
            "--mode",
            self.mode,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def run(
        self,
        video_root: str | Path,
        output_dir: str | Path,
        dimensions: Sequence[str],
        extra_args: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> list[VBenchInvocation]:
        self._validate_dimensions(dimensions)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        invocations: list[VBenchInvocation] = []
        for dimension in dimensions:
            dim_dir = root / dimension
            dim_dir.mkdir(parents=True, exist_ok=True)
            cmd = self.build_command(video_root=video_root, dimension=dimension, extra_args=extra_args)
            if dry_run:
                invocations.append(VBenchInvocation(dimension=dimension, command=cmd, cwd=dim_dir))
                continue
            completed = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True,
                cwd=str(dim_dir),
            )
            invocations.append(
                VBenchInvocation(
                    dimension=dimension,
                    command=cmd,
                    cwd=dim_dir,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
        return invocations


def run_and_collect_vbench(
    *,
    video_root: str | Path,
    output_dir: str | Path,
    dimensions: Sequence[str],
    model_name: str,
    command: str = "vbench",
    mode: str = "custom_input",
    extra_args: Sequence[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[VBenchInvocation], VBenchMergedResults]:
    runner = VBenchRunner(command=command, mode=mode)
    invocations = runner.run(
        video_root=video_root,
        output_dir=output_dir,
        dimensions=dimensions,
        extra_args=extra_args,
        dry_run=dry_run,
    )
    if dry_run:
        return invocations, VBenchMergedResults(summary_rows=[], per_video_rows=[])
    results = parse_vbench_outputs(output_dir, dimensions=dimensions, model_name=model_name)
    return invocations, results


def write_vbench_merged_outputs(results: VBenchMergedResults, output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if results.summary_rows:
        write_csv(results.summary_rows, out_dir / "vbench_merged_summary.csv")
        write_jsonl(results.summary_rows, out_dir / "vbench_merged_summary.jsonl")
    if results.per_video_rows:
        write_csv(results.per_video_rows, out_dir / "vbench_merged_per_video.csv")
        write_jsonl(results.per_video_rows, out_dir / "vbench_merged_per_video.jsonl")


def _print_invocations(invocations: Iterable[VBenchInvocation]) -> None:
    for item in invocations:
        print(" ".join(item.command))
        if item.stdout:
            print(item.stdout)
        if item.stderr:
            print(item.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official VBench through a thin wrapper.")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--command", default="vbench")
    parser.add_argument("--mode", default="custom_input")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()

    model_name = args.model_name or Path(args.video_root).name
    invocations, results = run_and_collect_vbench(
        video_root=args.video_root,
        output_dir=args.output_dir,
        dimensions=args.dimensions,
        model_name=model_name,
        command=args.command,
        mode=args.mode,
        extra_args=extra,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        _print_invocations(invocations)
        return

    write_vbench_merged_outputs(results, args.output_dir)
    _print_invocations(invocations)
    print(f"VBench merged summary rows: {len(results.summary_rows)}")
    print(f"VBench merged per-video rows: {len(results.per_video_rows)}")


if __name__ == "__main__":
    main()
