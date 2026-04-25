from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from longvideo_eval.schema import PromptRecord


def _load_prompt_file(path: str | Path) -> Dict[str, PromptRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts: Dict[str, PromptRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row:
                raise ValueError(f"Prompt file line {line_no} must include 'id' and 'prompt'")
            meta = {k: v for k, v in row.items() if k not in {"id", "prompt", "category"}}
            prompts[row["id"]] = PromptRecord(
                id=str(row["id"]),
                prompt=str(row["prompt"]),
                category=row.get("category"),
                metadata=meta or None,
            )
    return prompts


def _load_prompt_dir(path: str | Path) -> Dict[str, PromptRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Prompt directory is not a directory: {path}")

    prompts: Dict[str, PromptRecord] = {}
    for prompt_path in sorted(path.glob("*.txt")):
        text = prompt_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {prompt_path}")
        prompts[prompt_path.stem] = PromptRecord(
            id=prompt_path.stem,
            prompt=text,
            category=None,
            metadata=None,
        )
    return prompts


def load_prompts(
    prompt_file: str | Path | None = None,
    prompt_dir: str | Path | None = None,
) -> Dict[str, PromptRecord]:
    prompts: Dict[str, PromptRecord] = {}
    if prompt_file is not None:
        prompts.update(_load_prompt_file(prompt_file))
    if prompt_dir is not None:
        prompts.update(_load_prompt_dir(prompt_dir))
    if not prompts:
        raise ValueError("At least one of prompt_file or prompt_dir must be provided")
    return prompts
