#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EVAL_ENV="${EVAL_ENV:-eval}"
VIDEO_ROOT="${VIDEO_ROOT:-./test_model_perf_161output}"
PROMPT_DIR="${PROMPT_DIR:-./test_model_perf/prompt}"
DATASET_LAYOUT="${DATASET_LAYOUT:-prompt_dirs}"
MODEL_NAME="${MODEL_NAME:-test_model_perf_161output}"
VIDEO_SELECTION="${VIDEO_SELECTION:-latest}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/test_model_perf_all_metrics}"

SAMPLE_FPS="${SAMPLE_FPS:-2}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-2}"
MAX_SEGMENTS="${MAX_SEGMENTS:-30}"
MAX_FRAMES_PER_SEGMENT="${MAX_FRAMES_PER_SEGMENT:-8}"

CLIP_BACKEND="${CLIP_BACKEND:-openai_clip}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-$HOME/.cache/clip}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
DINOV2_MODEL="${DINOV2_MODEL:-$HOME/.cache/dinov2-base}"

REPETITION_MIN_GAP_SEGMENTS="${REPETITION_MIN_GAP_SEGMENTS:-5}"
REPETITION_THRESHOLD="${REPETITION_THRESHOLD:-0.95}"
RUNTIME_FILE="${RUNTIME_FILE:-}"
# 默认基础指标 metadata、quality_proxy、colorhist
# 自定义 CLIP、DINOv2
# 6 个已集成的 VBench 维度
VBENCH_DIMENSIONS=(
  aesthetic_quality
  imaging_quality
  subject_consistency
  background_consistency
  motion_smoothness
  dynamic_degree
)

CMD=(
  conda run -n "${EVAL_ENV}"
  longvideo-eval run
  --video-root "${VIDEO_ROOT}"
  --prompt-dir "${PROMPT_DIR}"
  --dataset-layout "${DATASET_LAYOUT}"
  --model-name "${MODEL_NAME}"
  --video-selection "${VIDEO_SELECTION}"
  --output-dir "${OUTPUT_DIR}"
  --sample-fps "${SAMPLE_FPS}"
  --segment-seconds "${SEGMENT_SECONDS}"
  --max-segments "${MAX_SEGMENTS}"
  --max-frames-per-segment "${MAX_FRAMES_PER_SEGMENT}"
  --enable-clip
  --clip-backend "${CLIP_BACKEND}"
  --clip-cache-dir "${CLIP_CACHE_DIR}"
  --clip-model "${CLIP_MODEL}"
  --enable-dinov2
  --dinov2-model "${DINOV2_MODEL}"
  --repetition-min-gap-segments "${REPETITION_MIN_GAP_SEGMENTS}"
  --repetition-threshold "${REPETITION_THRESHOLD}"
  --enable-vbench
  --vbench-dimensions "${VBENCH_DIMENSIONS[@]}"
)

if [[ -n "${RUNTIME_FILE}" ]]; then
  CMD+=(--runtime-file "${RUNTIME_FILE}")
fi

printf 'Running command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
