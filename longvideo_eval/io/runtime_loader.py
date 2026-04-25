from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_runtime_sidecar(path: str | Path | None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load optional runtime JSONL keyed by (model, prompt_id)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime sidecar not found: {path}")

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "model" not in row or "prompt_id" not in row:
                raise ValueError(f"Runtime file line {line_no} must include model and prompt_id")
            out[(str(row["model"]), str(row["prompt_id"]))] = row
    return out
