# LongVideoEvalKit v0.1

`LongVideoEvalKit` is an extensible evaluation toolkit for long-video generation experiments. v0.1 focuses on a stable, runnable pipeline rather than a full benchmark leaderboard.

For a detailed explanation of every metric, formulas, and multi-metric analysis strategies, see [introduction.md](./introduction.md). A Chinese version is available at [介绍.md](./介绍.md).

It provides:

- batch video loading and segmentation;
- prompt matching by `prompt_id`;
- long-range image-feature consistency;
- CLIP-F / CLIP-T with either `open_clip_torch` or OpenAI `clip`;
- DINOv2 long consistency when `transformers` weights are available;
- drift, repetition, start-end quality degradation, segment quality standard deviation;
- efficiency/statistics merging from a sidecar runtime file;
- dataset coverage reporting for missing prompts;
- CSV / JSONL reports;
- optional official VBench execution and merged `vbench.*` report columns.

The toolkit deliberately does **not** reimplement official VBench dimensions. It can invoke official VBench, standardize the raw outputs, and merge them into toolkit reports with a `vbench.*` prefix.

## Installation

Basic installation:

```bash
pip install -e .
```

Optional OpenCLIP metrics:

```bash
pip install -e '.[clip]'
```

OpenAI CLIP metrics can also reuse an existing `clip` install or a local repo such as `~/CLIP`.

Optional DINOv2 metrics:

```bash
pip install -e '.[dino]'
```

Official VBench is installed separately:

```bash
pip install vbench
```

## Supported input layouts

### Standard layout

```text
eval_data/
  model_a/
    prompt_0001.mp4
    prompt_0002.mp4
  model_b/
    prompt_0001.mp4
    prompt_0002.mp4
  prompts.jsonl
```

`prompts.jsonl`:

```json
{"id": "prompt_0001", "prompt": "A cat walking in a rainy street", "category": "animal_motion"}
{"id": "prompt_0002", "prompt": "A robot cooking in a futuristic kitchen", "category": "human_object_interaction"}
```

The video filename stem must match the prompt id. For example, `prompt_0001.mp4` matches `"id": "prompt_0001"`.

### Prompt-directory layout

This layout is useful for local model testing where prompts are stored as `*.txt` files and outputs are grouped by `prompt_id`:

```text
test_model_perf/
  prompt/
    000001.txt
    000002.txt
test_model_perf_161output/
  000001/
    candidate_20260422_210534.mp4
  000002/
    candidate_20260422_211650.mp4
```

Use `--dataset-layout prompt_dirs`, point `--prompt-dir` at the prompt text directory, and set `--video-selection latest` to evaluate the newest video per prompt directory.

## Quick start

Run the default v0.1 pipeline on the standard layout:

```bash
longvideo-eval run \
  --video-root ./eval_data \
  --prompt-file ./eval_data/prompts.jsonl \
  --output-dir ./outputs \
  --segment-seconds 2 \
  --sample-fps 2
```

This produces:

```text
outputs/
  dataset_coverage.json
  missing_prompts.jsonl  # only written when some prompts have no video
  per_video_metrics.jsonl
  per_video_metrics.csv
  model_summary.csv
```

For a config-based run:

```bash
longvideo-eval run --config configs/eval_v0_1.yaml
```

Run the prompt-directory layout used in this repository:

```bash
conda run -n eval longvideo-eval run \
  --video-root ./test_model_perf_161output \
  --prompt-dir ./test_model_perf/prompt \
  --dataset-layout prompt_dirs \
  --model-name test_model_perf_161output \
  --video-selection latest \
  --output-dir ./outputs/test_model_perf_smoke \
  --sample-fps 2 \
  --segment-seconds 2
```

Enable CLIP metrics with OpenAI CLIP from `~/CLIP`:

```bash
conda run -n eval longvideo-eval run \
  --video-root ./test_model_perf_161output \
  --prompt-dir ./test_model_perf/prompt \
  --dataset-layout prompt_dirs \
  --model-name test_model_perf_161output \
  --video-selection latest \
  --output-dir ./outputs/test_model_perf_clip \
  --enable-clip \
  --clip-backend openai_clip \
  --clip-repo ~/CLIP \
  --clip-cache-dir ~/CLIP/.cache/clip \
  --clip-model ViT-B/32
```

Enable official VBench and merge `vbench.*` fields into the same report:

```bash
conda run -n eval longvideo-eval run \
  --video-root ./test_model_perf_161output \
  --prompt-dir ./test_model_perf/prompt \
  --dataset-layout prompt_dirs \
  --model-name test_model_perf_161output \
  --video-selection latest \
  --output-dir ./outputs/test_model_perf_vbench \
  --enable-vbench \
  --vbench-dimensions aesthetic_quality imaging_quality subject_consistency background_consistency motion_smoothness dynamic_degree
```

This writes:

```text
outputs/test_model_perf_vbench/
  per_video_metrics.csv
  model_summary.csv
  vbench_merged_summary.csv
  vbench_merged_per_video.csv  # only when per-video VBench rows are parseable
  vbench_raw/
    test_model_perf_161output/
      aesthetic_quality/
      imaging_quality/
      ...
```

## Metrics in v0.1

### Always available

These use no heavy model weights:

- `metadata.duration_sec`, `metadata.video_fps`, `metadata.width`, `metadata.height`;
- `quality_proxy.mean`, `quality_proxy.segment_std`, `quality_proxy.delta_abs`, `quality_proxy.delta_drop`;
- `colorhist_lc.mean`, `colorhist_lc.end`, `colorhist_lc.drop`;
- `drift_colorhist.mean`, `drift_colorhist.end`;
- `repetition_colorhist.ratio`, `repetition_colorhist.mean_sim`, `repetition_colorhist.max_sim`.

The `quality_proxy` is a lightweight no-reference proxy based on sharpness and exposure. It is useful for smoke tests and relative diagnostics, but it is **not** a replacement for VBench Imaging Quality / MUSIQ.

### Optional CLIP metrics

Require either `open_clip_torch` or OpenAI `clip`:

- `clip_f.mean`, `clip_f.end`, `clip_f.drop` — long-range visual semantic consistency;
- `clip_t.mean`, `clip_t.end`, `clip_t.drop` — prompt alignment over time;
- `drift_clip.mean`, `drift_clip.end`;
- `repetition_clip.ratio`, `repetition_clip.mean_sim`, `repetition_clip.max_sim`.

Backends:

- `open_clip` keeps the original v0.1 behavior and uses `clip_model` + `clip_pretrained`.
- `openai_clip` uses `clip.load()` and optionally prepends `--clip-repo ~/CLIP` to `sys.path`.
- In this environment, `--clip-cache-dir ~/CLIP/.cache/clip` reuses the existing `ViT-B-32.pt` cache.
- You can also point `--clip-cache-dir` to any other writable directory when the default `~/.cache/clip` location is not writable.
- For long prompts, the OpenAI CLIP path tokenizes with truncation enabled to avoid failures on prompt files like `test_model_perf/prompt/*.txt`.

### Optional DINOv2 metrics

Require `transformers` and a DINOv2 checkpoint:

- `dinov2_lc.mean`, `dinov2_lc.end`, `dinov2_lc.drop`;
- `drift_dinov2.mean`, `drift_dinov2.end`.

### Optional official VBench columns

When `--enable-vbench` is used, official VBench is run in `custom_input` mode and merged into reports with a `vbench.*` prefix. Current integrated dimensions:

- `vbench.aesthetic_quality`
- `vbench.imaging_quality`
- `vbench.subject_consistency`
- `vbench.background_consistency`
- `vbench.motion_smoothness`
- `vbench.dynamic_degree`

These fields come from official VBench outputs. They are intentionally namespaced to avoid confusion with toolkit-native proxy metrics such as `quality_proxy.*` or `clip_*`.

## Runtime sidecar file

If you have generation-time statistics, pass a JSONL file:

```bash
longvideo-eval run \
  --video-root ./eval_data \
  --prompt-file ./eval_data/prompts.jsonl \
  --runtime-file ./runtime.jsonl \
  --output-dir ./outputs
```

Each row:

```json
{"model": "model_a", "prompt_id": "prompt_0001", "runtime_sec": 182.3, "gpu_mem_peak_gb": 63.4, "latency_ms": 450.0, "kv_cache_gb": 12.1}
```

The toolkit computes `generation_fps = num_frames / runtime_sec` when `runtime_sec` is available.

## Coverage reporting

Each run emits `dataset_coverage.json` with prompt/video counts, matched prompts, missing prompts, and orphan video prompt ids. When some prompts have no matching video, `missing_prompts.jsonl` is also written. This is expected for partial local test batches such as the current `test_model_perf_161output`, where `000009` and `000010` are still empty.

## Testing

Install the package into the `eval` environment:

```bash
conda run -n eval pip install -e ./longvideo_evalkit_v0_1
```

Run unit tests:

```bash
conda run -n eval pytest longvideo_evalkit_v0_1/tests -q
```

Run a smoke test on the current repo data:

```bash
conda run -n eval longvideo-eval run \
  --video-root ./test_model_perf_161output \
  --prompt-dir ./test_model_perf/prompt \
  --dataset-layout prompt_dirs \
  --model-name test_model_perf_161output \
  --video-selection latest \
  --output-dir ./outputs/test_model_perf_smoke
```

## VBench integration

v0.1 includes a wrapper and a `longvideo-eval vbench` subcommand. Both use official VBench `custom_input` mode and write merged helper files alongside the raw VBench artifacts.

Standalone command:

```bash
longvideo-eval vbench \
  --video-root ./eval_data/model_a \
  --output-dir ./outputs/vbench_model_a \
  --dimensions aesthetic_quality imaging_quality subject_consistency background_consistency motion_smoothness dynamic_degree
```

Module entrypoint:

```bash
python -m longvideo_eval.metrics.vbench_wrapper \
  --video-root ./eval_data/model_a \
  --output-dir ./outputs/vbench_model_a \
  --dimensions aesthetic_quality imaging_quality subject_consistency background_consistency motion_smoothness dynamic_degree
```

Notes:

- Official VBench is still a separate dependency: `pip install vbench`
- The toolkit runs one VBench invocation per requested dimension.
- Integrated mode currently supports only the official custom-input dimensions documented by VBench: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`.
- Raw official outputs are stored below `vbench_raw/`; standardized toolkit files are written as `vbench_merged_summary.csv/jsonl` and, when available, `vbench_merged_per_video.csv/jsonl`.

## Development roadmap

v0.1:

- stable package structure;
- video reading, sampling, segmentation;
- feature-based long-range consistency;
- drift/repetition/degradation metrics;
- runtime merging;
- JSONL/CSV output.

v0.2:

- HTML report;
- time-curve plots;
- calibrated repetition thresholds;
- bootstrap confidence intervals.

v1.0:

- VLM judge;
- human 2AFC / Likert / ELO-K32;
- leaderboard format;
- benchmark prompt suite and reproducibility cards.
