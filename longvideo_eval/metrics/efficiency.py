from __future__ import annotations

from typing import Any, Dict

from longvideo_eval.schema import VideoMetadata


def compute_efficiency(metadata: VideoMetadata, runtime: Dict[str, Any] | None) -> Dict[str, float | str]:
    out: Dict[str, float | str] = {
        "metadata.duration_sec": float(metadata.duration_sec),
        "metadata.video_fps": float(metadata.video_fps),
        "metadata.num_frames": float(metadata.num_frames),
        "metadata.width": float(metadata.width),
        "metadata.height": float(metadata.height),
    }
    if not runtime:
        return out

    numeric_keys = [
        "runtime_sec",
        "gpu_mem_peak_gb",
        "latency_ms",
        "kv_cache_gb",
        "denoise_sec",
        "decode_sec",
        "io_sec",
    ]
    for key in numeric_keys:
        if key in runtime and runtime[key] is not None:
            try:
                out[f"efficiency.{key}"] = float(runtime[key])
            except (TypeError, ValueError):
                pass

    if "runtime_sec" in runtime and runtime["runtime_sec"]:
        runtime_sec = float(runtime["runtime_sec"])
        if runtime_sec > 0 and metadata.num_frames > 0:
            out["efficiency.generation_fps"] = float(metadata.num_frames / runtime_sec)
    return out
