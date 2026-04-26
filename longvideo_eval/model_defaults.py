from __future__ import annotations

from pathlib import Path


def default_cache_root() -> Path:
    return Path("~/.cache").expanduser()


def default_clip_cache_dir() -> Path:
    return default_cache_root() / "clip"


def default_dinov2_model_dir() -> Path:
    return default_cache_root() / "dinov2-base"


def default_vbench_cache_dir() -> Path:
    return default_cache_root() / "vbench"


OPENAI_CLIP_MODEL_FILENAMES: dict[str, str] = {
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
}


VBENCH_LOCAL_ASSET_PATHS: dict[str, list[Path]] = {
    "aesthetic_quality": [
        default_vbench_cache_dir() / "clip_model" / "ViT-L-14.pt",
        default_vbench_cache_dir() / "aesthetic_model" / "emb_reader" / "sa_0_4_vit_l_14_linear.pth",
    ],
    "imaging_quality": [
        default_vbench_cache_dir() / "pyiqa_model" / "musiq_spaq_ckpt-358bb6af.pth",
    ],
    "subject_consistency": [
        default_vbench_cache_dir() / "dino_model" / "facebookresearch_dino_main",
        default_vbench_cache_dir() / "dino_model" / "dino_vitbase16_pretrain.pth",
    ],
    "background_consistency": [
        default_vbench_cache_dir() / "clip_model" / "ViT-B-32.pt",
    ],
    "motion_smoothness": [
        default_vbench_cache_dir() / "amt_model" / "amt-s.pth",
    ],
    "dynamic_degree": [
        default_vbench_cache_dir() / "raft_model" / "models" / "raft-things.pth",
    ],
}

