from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class VideoRecord:
    model: str
    prompt_id: str
    path: Path
    prompt: Optional[PromptRecord]


@dataclass
class VideoMetadata:
    duration_sec: float
    video_fps: float
    num_frames: int
    width: int
    height: int
