from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from longvideo_eval.model_defaults import default_clip_cache_dir, default_dinov2_model_dir


@dataclass
class DatasetConfig:
    video_root: Path
    prompt_file: Optional[Path] = None
    prompt_dir: Optional[Path] = None
    runtime_file: Optional[Path] = None
    layout: str = "standard"
    model_name: Optional[str] = None
    video_selection: str = "all"


@dataclass
class SamplingConfig:
    sample_fps: float = 2.0
    segment_seconds: float = 2.0
    max_segments: int = 30
    max_frames_per_segment: int = 8


@dataclass
class MetricConfig:
    quality_proxy: bool = True
    colorhist: bool = True
    clip: bool = False
    dinov2: bool = False
    paper_metrics: bool = False
    paper_quality_clip_seconds: float = 5.0
    paper_cf_window_radius_seconds: float = 0.5
    repetition_min_gap_segments: int = 5
    repetition_threshold: float = 0.95


@dataclass
class ModelConfig:
    clip_model: str = "ViT-B/32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    dinov2_model: str | Path = field(default_factory=default_dinov2_model_dir)
    clip_backend: str = "openai_clip"
    clip_repo: Optional[Path] = None
    clip_cache_dir: Optional[Path] = field(default_factory=default_clip_cache_dir)


@dataclass
class ReportConfig:
    output_dir: Path = Path("./outputs")


@dataclass
class VBenchConfig:
    enabled: bool = False
    dimensions: list[str] = field(default_factory=list)
    command: str = "vbench"
    mode: str = "custom_input"
    raw_output_subdir: str = "vbench_raw"


@dataclass
class EvalConfig:
    dataset: DatasetConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    vbench: VBenchConfig = field(default_factory=VBenchConfig)


def _path_or_none(value: Any) -> Optional[Path]:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _validate_dataset_config(dataset: DatasetConfig) -> None:
    if dataset.layout not in {"standard", "prompt_dirs"}:
        raise ValueError(f"Unsupported dataset.layout: {dataset.layout}")
    if dataset.video_selection not in {"all", "latest"}:
        raise ValueError(f"Unsupported dataset.video_selection: {dataset.video_selection}")
    if dataset.prompt_file is None and dataset.prompt_dir is None:
        raise ValueError("Dataset config must include prompt_file or prompt_dir")


def _validate_model_config(models: ModelConfig) -> None:
    if models.clip_backend not in {"open_clip", "openai_clip"}:
        raise ValueError(f"Unsupported models.clip_backend: {models.clip_backend}")
    if models.clip_backend == "open_clip":
        checkpoint = Path(models.clip_pretrained).expanduser()
        if not checkpoint.is_file():
            raise ValueError(
                "models.clip_pretrained must point to a local OpenCLIP checkpoint file when "
                "models.clip_backend='open_clip'"
            )


def _validate_metric_config(metrics: MetricConfig) -> None:
    if metrics.paper_quality_clip_seconds <= 0:
        raise ValueError("metrics.paper_quality_clip_seconds must be positive")
    if metrics.paper_cf_window_radius_seconds < 0:
        raise ValueError("metrics.paper_cf_window_radius_seconds must be non-negative")
    if metrics.repetition_min_gap_segments < 0:
        raise ValueError("metrics.repetition_min_gap_segments must be non-negative")
    if not (0.0 <= metrics.repetition_threshold <= 1.0):
        raise ValueError("metrics.repetition_threshold must be between 0 and 1")


def _normalize_metric_config(metrics: MetricConfig) -> None:
    if metrics.paper_metrics:
        metrics.clip = True
        metrics.dinov2 = True


def _validate_vbench_config(vbench: VBenchConfig) -> None:
    if not vbench.command:
        raise ValueError("VBench command must be non-empty")
    if vbench.mode != "custom_input":
        raise ValueError(f"Unsupported vbench.mode: {vbench.mode}")
    if not vbench.raw_output_subdir:
        raise ValueError("VBench raw_output_subdir must be non-empty")
    if vbench.enabled and not vbench.dimensions:
        raise ValueError("VBench is enabled but no vbench.dimensions were provided")


def load_config(path: str | Path) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    dataset_raw = raw.get("dataset", {})
    sampling_raw = raw.get("sampling", {})
    metrics_raw = raw.get("metrics", {})
    models_raw = raw.get("models", {})
    report_raw = raw.get("report", {})
    vbench_raw = raw.get("vbench", {})

    if "video_root" not in dataset_raw:
        raise ValueError("Config must include dataset.video_root")
    if "prompt_file" not in dataset_raw and "prompt_dir" not in dataset_raw:
        raise ValueError("Config must include dataset.prompt_file or dataset.prompt_dir")

    cfg = EvalConfig(
        dataset=DatasetConfig(
            video_root=Path(dataset_raw["video_root"]).expanduser(),
            prompt_file=_path_or_none(dataset_raw.get("prompt_file")),
            prompt_dir=_path_or_none(dataset_raw.get("prompt_dir")),
            runtime_file=_path_or_none(dataset_raw.get("runtime_file")),
            layout=str(dataset_raw.get("layout", "standard")),
            model_name=dataset_raw.get("model_name"),
            video_selection=str(dataset_raw.get("video_selection", "all")),
        ),
        sampling=SamplingConfig(**sampling_raw),
        metrics=MetricConfig(**metrics_raw),
        models=ModelConfig(
            clip_model=models_raw.get("clip_model", "ViT-B/32"),
            clip_pretrained=models_raw.get("clip_pretrained", "laion2b_s34b_b79k"),
            dinov2_model=_path_or_none(models_raw.get("dinov2_model")) or default_dinov2_model_dir(),
            clip_backend=models_raw.get("clip_backend", "openai_clip"),
            clip_repo=_path_or_none(models_raw.get("clip_repo")),
            clip_cache_dir=_path_or_none(models_raw.get("clip_cache_dir")) or default_clip_cache_dir(),
        ),
        report=ReportConfig(output_dir=Path(report_raw.get("output_dir", "./outputs")).expanduser()),
        vbench=VBenchConfig(
            enabled=bool(vbench_raw.get("enabled", False)),
            dimensions=list(vbench_raw.get("dimensions", []) or []),
            command=str(vbench_raw.get("command", "vbench")),
            mode=str(vbench_raw.get("mode", "custom_input")),
            raw_output_subdir=str(vbench_raw.get("raw_output_subdir", "vbench_raw")),
        ),
    )
    _validate_dataset_config(cfg.dataset)
    _normalize_metric_config(cfg.metrics)
    _validate_metric_config(cfg.metrics)
    _validate_model_config(cfg.models)
    _validate_vbench_config(cfg.vbench)
    return cfg


def build_config_from_args(args: Any) -> EvalConfig:
    if args.config:
        cfg = load_config(args.config)
        if args.video_root:
            cfg.dataset.video_root = Path(args.video_root).expanduser()
        if args.prompt_file:
            cfg.dataset.prompt_file = Path(args.prompt_file).expanduser()
        if args.prompt_dir:
            cfg.dataset.prompt_dir = Path(args.prompt_dir).expanduser()
        if args.runtime_file:
            cfg.dataset.runtime_file = Path(args.runtime_file).expanduser()
        if args.output_dir:
            cfg.report.output_dir = Path(args.output_dir).expanduser()
        if args.dataset_layout:
            cfg.dataset.layout = args.dataset_layout
        if args.model_name:
            cfg.dataset.model_name = args.model_name
        if args.video_selection:
            cfg.dataset.video_selection = args.video_selection
        if args.segment_seconds is not None:
            cfg.sampling.segment_seconds = args.segment_seconds
        if args.sample_fps is not None:
            cfg.sampling.sample_fps = args.sample_fps
        if args.max_segments is not None:
            cfg.sampling.max_segments = args.max_segments
        if args.max_frames_per_segment is not None:
            cfg.sampling.max_frames_per_segment = args.max_frames_per_segment
        if args.enable_clip:
            cfg.metrics.clip = True
        if args.enable_dinov2:
            cfg.metrics.dinov2 = True
        if args.enable_paper_metrics:
            cfg.metrics.paper_metrics = True
        if args.paper_quality_clip_seconds is not None:
            cfg.metrics.paper_quality_clip_seconds = args.paper_quality_clip_seconds
        if args.paper_cf_window_radius_seconds is not None:
            cfg.metrics.paper_cf_window_radius_seconds = args.paper_cf_window_radius_seconds
        if args.clip_model:
            cfg.models.clip_model = args.clip_model
        if args.clip_pretrained:
            cfg.models.clip_pretrained = args.clip_pretrained
        if args.dinov2_model:
            cfg.models.dinov2_model = Path(args.dinov2_model).expanduser()
        if args.clip_backend:
            cfg.models.clip_backend = args.clip_backend
        if args.clip_repo:
            cfg.models.clip_repo = Path(args.clip_repo).expanduser()
        if args.clip_cache_dir:
            cfg.models.clip_cache_dir = Path(args.clip_cache_dir).expanduser()
        if args.enable_vbench:
            cfg.vbench.enabled = True
        if args.vbench_dimensions:
            cfg.vbench.dimensions = list(args.vbench_dimensions)
        if args.vbench_command:
            cfg.vbench.command = args.vbench_command
        if args.vbench_mode:
            cfg.vbench.mode = args.vbench_mode
        if args.vbench_raw_output_subdir:
            cfg.vbench.raw_output_subdir = args.vbench_raw_output_subdir
        _validate_dataset_config(cfg.dataset)
        _normalize_metric_config(cfg.metrics)
        _validate_metric_config(cfg.metrics)
        _validate_model_config(cfg.models)
        _validate_vbench_config(cfg.vbench)
        return cfg

    if not args.video_root or (not args.prompt_file and not args.prompt_dir):
        raise ValueError("Either --config or --video-root with --prompt-file/--prompt-dir is required")

    cfg = EvalConfig(
        dataset=DatasetConfig(
            video_root=Path(args.video_root).expanduser(),
            prompt_file=Path(args.prompt_file).expanduser() if args.prompt_file else None,
            prompt_dir=Path(args.prompt_dir).expanduser() if args.prompt_dir else None,
            runtime_file=Path(args.runtime_file).expanduser() if args.runtime_file else None,
            layout=args.dataset_layout or "standard",
            model_name=args.model_name,
            video_selection=args.video_selection or "all",
        ),
        sampling=SamplingConfig(
            sample_fps=args.sample_fps if args.sample_fps is not None else 2.0,
            segment_seconds=args.segment_seconds if args.segment_seconds is not None else 2.0,
            max_segments=args.max_segments if args.max_segments is not None else 30,
            max_frames_per_segment=args.max_frames_per_segment if args.max_frames_per_segment is not None else 8,
        ),
        metrics=MetricConfig(
            quality_proxy=not args.disable_quality_proxy,
            colorhist=not args.disable_colorhist,
            clip=args.enable_clip,
            dinov2=args.enable_dinov2,
            paper_metrics=args.enable_paper_metrics,
            paper_quality_clip_seconds=args.paper_quality_clip_seconds if args.paper_quality_clip_seconds is not None else 5.0,
            paper_cf_window_radius_seconds=(
                args.paper_cf_window_radius_seconds if args.paper_cf_window_radius_seconds is not None else 0.5
            ),
            repetition_min_gap_segments=args.repetition_min_gap_segments,
            repetition_threshold=args.repetition_threshold,
        ),
        models=ModelConfig(
            clip_model=args.clip_model or "ViT-B/32",
            clip_pretrained=args.clip_pretrained or "laion2b_s34b_b79k",
            dinov2_model=Path(args.dinov2_model).expanduser() if args.dinov2_model else default_dinov2_model_dir(),
            clip_backend=args.clip_backend or "openai_clip",
            clip_repo=Path(args.clip_repo).expanduser() if args.clip_repo else None,
            clip_cache_dir=Path(args.clip_cache_dir).expanduser() if args.clip_cache_dir else default_clip_cache_dir(),
        ),
        report=ReportConfig(output_dir=Path(args.output_dir or "./outputs").expanduser()),
        vbench=VBenchConfig(
            enabled=args.enable_vbench,
            dimensions=list(args.vbench_dimensions or []),
            command=args.vbench_command or "vbench",
            mode=args.vbench_mode or "custom_input",
            raw_output_subdir=args.vbench_raw_output_subdir or "vbench_raw",
        ),
    )
    _validate_dataset_config(cfg.dataset)
    _normalize_metric_config(cfg.metrics)
    _validate_metric_config(cfg.metrics)
    _validate_model_config(cfg.models)
    _validate_vbench_config(cfg.vbench)
    return cfg
