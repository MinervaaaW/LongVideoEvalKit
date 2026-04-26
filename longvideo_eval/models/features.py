from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from longvideo_eval.model_defaults import (
    OPENAI_CLIP_MODEL_FILENAMES,
    default_clip_cache_dir,
    default_dinov2_model_dir,
)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


class FeatureExtractor:
    """Base class for image/text feature extractors."""

    name: str = "base"
    supports_text: bool = False

    def encode_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support text features")


class ColorHistExtractor(FeatureExtractor):
    """Lightweight fallback feature extractor based on HSV color histograms.

    This is not a semantic feature. It exists so the pipeline can run without heavy model weights.
    """

    name = "colorhist"
    supports_text = False

    def __init__(self, bins: tuple[int, int, int] = (8, 8, 8)) -> None:
        self.bins = bins

    def encode_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        feats: List[np.ndarray] = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
            hist = hist.flatten().astype(np.float32)
            feats.append(hist)
        return l2_normalize(np.stack(feats, axis=0))


class OpenCLIPExtractor(FeatureExtractor):
    """OpenCLIP image/text feature extractor.

    Requires optional dependency: open_clip_torch and torch.
    """

    name = "clip"
    supports_text = True

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str | None = None) -> None:
        try:
            import torch
            import open_clip
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenCLIPExtractor requires: pip install 'longvideo-evalkit[clip]'") from exc

        self.torch = torch
        self.open_clip = open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = Path(pretrained).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "OpenCLIPExtractor is configured for local-only loading, "
                f"but the checkpoint was not found: {checkpoint}"
            )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(checkpoint),
        )
        self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        self.pretrained = str(checkpoint)

    def encode_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        images = []
        for frame in frames:
            pil = Image.fromarray(frame)
            images.append(self.preprocess(pil))
        batch = torch.stack(images, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().float().numpy()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().float().numpy()


class OpenAICLIPExtractor(FeatureExtractor):
    """OpenAI CLIP image/text feature extractor.

    Requires `torch` and a `clip` module, optionally loaded from a local repository.
    """

    name = "clip"
    supports_text = True

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str | None = None,
        clip_repo: str | Path | None = None,
        clip_cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenAICLIPExtractor requires torch") from exc

        repo = Path(clip_repo).expanduser() if clip_repo is not None else None
        if repo is not None:
            if not repo.exists():
                raise FileNotFoundError(f"CLIP repo not found: {repo}")
            repo_str = str(repo)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        try:
            import clip
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenAICLIPExtractor requires the clip package or a local CLIP repo") from exc

        self.torch = torch
        self.clip = clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = _canonical_openai_clip_model_name(model_name)
        cache_dir = Path(clip_cache_dir).expanduser() if clip_cache_dir is not None else default_clip_cache_dir()
        checkpoint = _resolve_openai_clip_checkpoint(self.model_name, cache_dir)
        self.model, self.preprocess = clip.load(str(checkpoint), device=self.device, jit=False)
        self.model.eval()

    def encode_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        images = [self.preprocess(Image.fromarray(frame)) for frame in frames]
        batch = torch.stack(images, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().float().numpy()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        try:
            tokens = self.clip.tokenize(list(texts), truncate=True)
        except TypeError:
            tokens = self.clip.tokenize(list(texts))
        tokens = tokens.to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().float().numpy()


def _canonical_openai_clip_model_name(model_name: str) -> str:
    aliases = {
        "ViT-B-32": "ViT-B/32",
        "ViT-B-16": "ViT-B/16",
        "ViT-L-14": "ViT-L/14",
    }
    return aliases.get(model_name, model_name)


def _resolve_openai_clip_checkpoint(model_name: str, cache_dir: str | Path | None = None) -> Path:
    canonical = _canonical_openai_clip_model_name(model_name)
    if canonical not in OPENAI_CLIP_MODEL_FILENAMES:
        raise ValueError(
            f"Unsupported OpenAI CLIP model for local cache loading: {canonical}. "
            f"Supported models: {', '.join(sorted(OPENAI_CLIP_MODEL_FILENAMES))}"
        )
    root = Path(cache_dir).expanduser() if cache_dir is not None else default_clip_cache_dir()
    checkpoint = root / OPENAI_CLIP_MODEL_FILENAMES[canonical]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "OpenAICLIPExtractor is configured for local-only loading, "
            f"but the checkpoint was not found: {checkpoint}"
        )
    return checkpoint


def build_clip_extractor(
    backend: str = "openai_clip",
    model_name: str = "ViT-B/32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str | None = None,
    clip_repo: str | Path | None = None,
    clip_cache_dir: str | Path | None = None,
) -> FeatureExtractor:
    if backend == "open_clip":
        return OpenCLIPExtractor(model_name=model_name, pretrained=pretrained, device=device)
    if backend == "openai_clip":
        return OpenAICLIPExtractor(
            model_name=model_name,
            device=device,
            clip_repo=clip_repo,
            clip_cache_dir=clip_cache_dir,
        )
    raise ValueError(f"Unknown clip backend: {backend}")


class DINOv2Extractor(FeatureExtractor):
    """DINOv2 image feature extractor via Hugging Face Transformers.

    Requires optional dependency: transformers and torch. The model checkpoint must already be
    available locally.
    """

    name = "dinov2"
    supports_text = False

    def __init__(self, model_name: str | Path = default_dinov2_model_dir(), device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("DINOv2Extractor requires: pip install 'longvideo-evalkit[dino]'") from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_name).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                "DINOv2Extractor is configured for local-only loading, "
                f"but the model directory was not found: {model_path}"
            )
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(self.device).eval()
        self.model_name = str(model_path)

    def encode_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        images = [Image.fromarray(frame) for frame in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Prefer CLS token if available.
            feats = outputs.last_hidden_state[:, 0, :]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().float().numpy()


def make_extractor(kind: str, **kwargs) -> FeatureExtractor:
    kind = kind.lower()
    if kind in {"colorhist", "hist", "fallback"}:
        return ColorHistExtractor()
    if kind in {"clip", "open_clip", "openclip"}:
        return build_clip_extractor(
            backend=kwargs.get("backend", "openai_clip"),
            model_name=kwargs.get("model_name", "ViT-B/32"),
            pretrained=kwargs.get("pretrained", "laion2b_s34b_b79k"),
            device=kwargs.get("device"),
            clip_repo=kwargs.get("clip_repo"),
            clip_cache_dir=kwargs.get("clip_cache_dir"),
        )
    if kind in {"dinov2", "dino"}:
        return DINOv2Extractor(
            model_name=kwargs.get("model_name", default_dinov2_model_dir()),
            device=kwargs.get("device"),
        )
    raise ValueError(f"Unknown feature extractor kind: {kind}")
