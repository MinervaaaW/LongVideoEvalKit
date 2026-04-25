from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def write_video(path: Path, color_shift: int, fps: int = 12, seconds: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 160, 96
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(fps * seconds):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = int((w - 20) * (t / max(fps * seconds - 1, 1)))
        base = (30 + color_shift) % 255
        img[:, :] = (base, 50, 80)
        cv2.rectangle(img, (x, 35), (x + 20, 55), (220, 220 - color_shift, 50 + color_shift), -1)
        cv2.putText(img, f"t={t}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        writer.write(img)
    writer.release()


def main() -> None:
    root = Path("eval_data")
    prompts = [
        {"id": "prompt_0001", "prompt": "A moving square crossing the frame", "category": "toy_motion"},
        {"id": "prompt_0002", "prompt": "A colored block moves from left to right", "category": "toy_motion"},
    ]
    (root).mkdir(exist_ok=True)
    with open(root / "prompts.jsonl", "w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for model, shift in [("model_a", 0), ("model_b", 45)]:
        for i, p in enumerate(prompts):
            write_video(root / model / f"{p['id']}.mp4", color_shift=shift + i * 20)
    print(f"Toy data written to: {root.resolve()}")


if __name__ == "__main__":
    main()
