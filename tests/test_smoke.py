import sys
import types
from pathlib import Path

import numpy as np

from longvideo_eval.config import MetricConfig, SamplingConfig, build_config_from_args
from longvideo_eval.io.prompt_loader import load_prompts
from longvideo_eval.io.video_reader import list_video_records
from longvideo_eval.model_defaults import default_clip_cache_dir, default_dinov2_model_dir
from longvideo_eval.models.features import ColorHistExtractor, DINOv2Extractor, OpenAICLIPExtractor


def test_config_defaults():
    s = SamplingConfig()
    assert s.sample_fps == 2.0
    m = MetricConfig()
    assert m.quality_proxy is True


def test_colorhist_extractor_shape():
    frames = [np.zeros((32, 32, 3), dtype=np.uint8), np.ones((32, 32, 3), dtype=np.uint8) * 255]
    extractor = ColorHistExtractor()
    feats = extractor.encode_images(frames)
    assert feats.shape[0] == 2


def test_load_prompts_from_txt_dir(tmp_path: Path):
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()
    (prompt_dir / "000001.txt").write_text("hello world", encoding="utf-8")
    (prompt_dir / "000002.txt").write_text("another prompt", encoding="utf-8")

    prompts = load_prompts(prompt_dir=prompt_dir)

    assert sorted(prompts.keys()) == ["000001", "000002"]
    assert prompts["000001"].prompt == "hello world"


def test_list_video_records_prompt_dirs_latest_selection(tmp_path: Path):
    root = tmp_path / "outputs"
    prompt_dir = root / "000001"
    prompt_dir.mkdir(parents=True)
    older = prompt_dir / "sample_20240101_010101.mp4"
    newer = prompt_dir / "sample_20240101_010102.mp4"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    other_dir = root / "000002"
    other_dir.mkdir()
    only = other_dir / "only.mp4"
    only.write_bytes(b"")

    records = list_video_records(root, layout="prompt_dirs", model_name="demo_model", video_selection="latest")

    assert records == [
        ("demo_model", "000001", newer),
        ("demo_model", "000002", only),
    ]


def test_build_config_from_args_prompt_dir():
    class Args:
        config = None
        video_root = "./videos"
        prompt_file = None
        prompt_dir = "./prompt"
        runtime_file = None
        output_dir = "./outputs"
        dataset_layout = "prompt_dirs"
        model_name = "demo_model"
        video_selection = "latest"
        sample_fps = None
        segment_seconds = None
        max_segments = None
        max_frames_per_segment = 8
        disable_quality_proxy = False
        disable_colorhist = False
        enable_clip = True
        enable_dinov2 = False
        enable_paper_metrics = False
        paper_quality_clip_seconds = None
        paper_cf_window_radius_seconds = None
        clip_model = "ViT-B-32"
        clip_pretrained = "laion2b_s34b_b79k"
        clip_backend = "openai_clip"
        clip_repo = "~/CLIP"
        clip_cache_dir = "/tmp/clip-cache"
        dinov2_model = "facebook/dinov2-base"
        repetition_min_gap_segments = 5
        repetition_threshold = 0.95
        enable_vbench = True
        vbench_dimensions = ["aesthetic_quality", "imaging_quality"]
        vbench_command = "vbench"
        vbench_mode = "custom_input"
        vbench_raw_output_subdir = "vbench_raw"

    cfg = build_config_from_args(Args())

    assert cfg.dataset.layout == "prompt_dirs"
    assert cfg.dataset.video_selection == "latest"
    assert cfg.dataset.prompt_dir == Path("./prompt").expanduser()
    assert cfg.models.clip_backend == "openai_clip"
    assert cfg.models.clip_repo == Path("~/CLIP").expanduser()
    assert cfg.models.clip_cache_dir == Path("/tmp/clip-cache")
    assert cfg.vbench.enabled is True
    assert cfg.vbench.dimensions == ["aesthetic_quality", "imaging_quality"]
    assert cfg.vbench.command == "vbench"


def test_build_config_from_args_uses_local_model_defaults():
    class Args:
        config = None
        video_root = "./videos"
        prompt_file = None
        prompt_dir = "./prompt"
        runtime_file = None
        output_dir = "./outputs"
        dataset_layout = "prompt_dirs"
        model_name = "demo_model"
        video_selection = "latest"
        sample_fps = None
        segment_seconds = None
        max_segments = None
        max_frames_per_segment = 8
        disable_quality_proxy = False
        disable_colorhist = False
        enable_clip = False
        enable_dinov2 = False
        enable_paper_metrics = False
        paper_quality_clip_seconds = None
        paper_cf_window_radius_seconds = None
        clip_model = None
        clip_pretrained = None
        clip_backend = None
        clip_repo = None
        clip_cache_dir = None
        dinov2_model = None
        repetition_min_gap_segments = 5
        repetition_threshold = 0.95
        enable_vbench = False
        vbench_dimensions = None
        vbench_command = None
        vbench_mode = None
        vbench_raw_output_subdir = None

    cfg = build_config_from_args(Args())

    assert cfg.models.clip_backend == "openai_clip"
    assert cfg.models.clip_model == "ViT-B/32"
    assert cfg.models.clip_cache_dir == default_clip_cache_dir()
    assert cfg.models.dinov2_model == default_dinov2_model_dir()


def test_build_config_from_args_paper_metrics_auto_enable_exact_dependencies():
    class Args:
        config = None
        video_root = "./videos"
        prompt_file = None
        prompt_dir = "./prompt"
        runtime_file = None
        output_dir = "./outputs"
        dataset_layout = "prompt_dirs"
        model_name = "demo_model"
        video_selection = "latest"
        sample_fps = None
        segment_seconds = None
        max_segments = None
        max_frames_per_segment = 8
        disable_quality_proxy = False
        disable_colorhist = False
        enable_clip = False
        enable_dinov2 = False
        enable_paper_metrics = True
        paper_quality_clip_seconds = 5.0
        paper_cf_window_radius_seconds = 0.5
        clip_model = None
        clip_pretrained = None
        clip_backend = None
        clip_repo = None
        clip_cache_dir = None
        dinov2_model = None
        repetition_min_gap_segments = 5
        repetition_threshold = 0.95
        enable_vbench = False
        vbench_dimensions = None
        vbench_command = None
        vbench_mode = None
        vbench_raw_output_subdir = None

    cfg = build_config_from_args(Args())

    assert cfg.metrics.paper_metrics is True
    assert cfg.metrics.clip is True
    assert cfg.metrics.dinov2 is True


def test_openai_clip_extractor_with_fake_clip(monkeypatch, tmp_path: Path):
    import torch

    clip_cache = tmp_path / "clip"
    clip_cache.mkdir()
    checkpoint = clip_cache / "ViT-B-32.pt"
    checkpoint.write_bytes(b"fake")
    called = {}

    class FakeModel:
        def eval(self):
            return self

        def encode_image(self, batch):
            return torch.ones((batch.shape[0], 4), dtype=torch.float32, device=batch.device)

        def encode_text(self, tokens):
            return torch.ones((tokens.shape[0], 4), dtype=torch.float32, device=tokens.device)

    def fake_load(model_name, device=None, jit=False):
        called["model_name"] = model_name
        return FakeModel(), lambda image: torch.ones((3, 2, 2), dtype=torch.float32)

    fake_clip = types.SimpleNamespace(
        load=fake_load,
        tokenize=lambda texts, truncate=False: torch.ones((len(texts), 77), dtype=torch.int64),
    )

    monkeypatch.setitem(sys.modules, "clip", fake_clip)
    extractor = OpenAICLIPExtractor(model_name="ViT-B-32", clip_cache_dir=clip_cache)

    image_feats = extractor.encode_images([np.zeros((4, 4, 3), dtype=np.uint8)])
    text_feats = extractor.encode_texts(["hello"])

    assert called["model_name"] == str(checkpoint)
    assert image_feats.shape == (1, 4)
    assert text_feats.shape == (1, 4)


def test_dinov2_extractor_uses_local_files_only(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "dinov2-base"
    model_dir.mkdir()
    calls = []

    class FakeProcessor:
        @staticmethod
        def from_pretrained(path, local_files_only=False):
            calls.append(("processor", path, local_files_only))
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(path, local_files_only=False):
            calls.append(("model", path, local_files_only))
            return FakeModel()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = types.SimpleNamespace(
        AutoImageProcessor=FakeProcessor,
        AutoModel=FakeModel,
    )

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    extractor = DINOv2Extractor(model_name=model_dir)

    assert extractor.model_name == str(model_dir)
    assert calls == [
        ("processor", str(model_dir), True),
        ("model", str(model_dir), True),
    ]
