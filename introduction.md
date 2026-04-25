# LongVideoEvalKit Introduction

## 1. Scope and positioning

`LongVideoEvalKit` is a feature-based evaluation toolkit for long-video generation experiments. Its goal is to provide a stable and reproducible analysis pipeline for local testing, ablation studies, and regression tracking.

It is important to separate three categories of metrics:

1. Toolkit-native proxy metrics: `quality_proxy.*`, `colorhist_*`, `clip_*`, `dinov2_*`, `drift_*`, `repetition_*`, `efficiency.*`.
2. Metadata and bookkeeping fields: `metadata.*`, `model`, `prompt_id`, `video_path`, `status`.
3. Official VBench metrics: not reimplemented inside the toolkit. The repository invokes official VBench, stores raw outputs, and merges standardized `vbench.*` fields into reports when enabled.

This distinction matters for interpretation:

- Toolkit-native metrics are useful for relative comparison, debugging, ablation, and failure diagnosis.
- Toolkit-native metrics should not be presented as official VBench scores.
- If a score comes from official VBench, it should be explicitly labeled as VBench in reports.

## 2. Evaluation pipeline

For each video, the toolkit follows the same high-level pipeline:

1. Read video metadata and sample frames at `sample_fps`.
2. Split sampled timestamps into equal-length temporal segments of `segment_seconds`.
3. Keep up to `max_frames_per_segment` frames per segment.
4. Extract frame-level features.
5. Average frame features inside each segment.
6. L2-normalize the segment feature.
7. Compute long-range consistency, drift, repetition, prompt alignment, and quality diagnostics from segment-level signals.

Formally, if segment `k` contains frame indices `S_k`, and frame feature is `f_i`, then the segment feature is:

```text
z_k = normalize(mean_{i in S_k}(f_i))
```

Many metrics are then defined relative to the first segment `z_1`, which acts as the anchor for long-range comparison.

## 3. Naming conventions

Understanding the suffixes is the fastest way to read the CSV files.

### 3.1 Per-video metrics

- `*.mean`: average over all temporal segments or all valid pairs.
- `*.end`: value at the last segment.
- `*.drop`: first-segment value minus last-segment value.
- `drift_*`: one minus similarity to the first segment.
- `repetition_*`: similarity statistics over temporally distant segment pairs.
- `quality_proxy.head`: quality of the first segment.
- `quality_proxy.tail`: quality of the last segment.

### 3.2 Model summary metrics

`model_summary.csv` is produced by averaging each numeric column over all successful videos for the same model.

So:

- `clip_f.mean.mean` means "the mean of per-video `clip_f.mean` values".
- `quality_proxy.tail.mean` means "the mean tail quality across all videos of this model".

## 4. Metric-by-metric interpretation

### 4.1 Metadata fields

These are descriptive fields, not quality scores:

- `metadata.duration_sec`: total video duration in seconds.
- `metadata.video_fps`: original video frame rate.
- `metadata.num_frames`: original total frame count.
- `metadata.width`, `metadata.height`: spatial resolution.

They are useful for normalizing efficiency and checking whether comparisons are fair.

### 4.2 `quality_proxy.*`

This is a lightweight no-reference quality proxy implemented in the toolkit. It is not VBench Imaging Quality and not MUSIQ.

For each frame:

```text
gray = grayscale(frame)
sharpness = Var(Laplacian(gray))
sharp_score = sharpness / (sharpness + 100)

mean_luma = mean(gray) / 255
exposure_score = 1 - min(abs(mean_luma - 0.5) / 0.5, 1)

frame_quality = 0.7 * sharp_score + 0.3 * exposure_score
```

For each segment, the segment quality is the mean frame quality inside that segment.

Reported fields:

- `quality_proxy.mean`: average segment quality across the whole video.
- `quality_proxy.head`: quality of the first segment.
- `quality_proxy.tail`: quality of the last segment.
- `quality_proxy.segment_std`: standard deviation of segment quality over time.
- `quality_proxy.delta_abs = |head - tail|`.
- `quality_proxy.delta_drop = head - tail`.

Interpretation:

- Higher `quality_proxy.mean` usually indicates sharper frames and more balanced exposure.
- Lower `quality_proxy.segment_std` usually indicates more stable perceptual quality over time.
- Positive `quality_proxy.delta_drop` means the ending is worse than the beginning.
- Negative `quality_proxy.delta_drop` means the ending is better than the beginning.

Limitations:

- It does not measure semantics, motion realism, temporal coherence, or human preference directly.
- It can overvalue sharp but semantically wrong outputs.
- It can undervalue intentionally dark or stylized scenes.

### 4.3 `colorhist_lc.*`

These are long-range consistency metrics computed from HSV color histograms.

For each frame:

```text
feature = normalize(HSV_histogram(frame, bins=(8, 8, 8)))
```

For each segment:

```text
z_k = normalize(mean frame histogram in segment k)
```

Similarity to the first segment:

```text
s_k = cos(z_k, z_1)
```

Reported fields:

- `colorhist_lc.mean = mean_k(s_k)`
- `colorhist_lc.end = s_last`
- `colorhist_lc.drop = s_1 - s_last`

Interpretation:

- High values mean the global color distribution stays similar to the start of the video.
- This is useful for detecting exposure drift, palette changes, or large scene/background changes.
- It is not a semantic metric. Two semantically different scenes can still have similar color histograms.

### 4.4 `drift_colorhist.*`

These fields convert color similarity into a drift score:

```text
drift_k = 1 - cos(z_k, z_1), for k >= 2
```

Reported fields:

- `drift_colorhist.mean`
- `drift_colorhist.end`
- `drift_colorhist.max`

Interpretation:

- Lower is better.
- High `drift_colorhist.end` means the color statistics at the end differ substantially from the start.
- High `drift_colorhist.max` means there exists at least one segment with strong appearance drift.

### 4.5 `repetition_colorhist.*`

These fields look for repetition using distant segment pairs.

If two segments are at least `repetition_min_gap_segments` apart, their similarity is included:

```text
pair_sim(i, j) = cos(z_i, z_j), with j >= i + min_gap
```

Let the threshold be `repetition_threshold`.

Reported fields:

- `repetition_colorhist.ratio`: fraction of valid distant pairs whose similarity is at least the threshold.
- `repetition_colorhist.mean_sim`: mean similarity over all valid distant pairs.
- `repetition_colorhist.max_sim`: maximum similarity over all valid distant pairs.
- `repetition_colorhist.num_pairs`: number of valid distant pairs.

Interpretation:

- High repetition can indicate looping, frozen backgrounds, cyclic failure, or mode collapse.
- Color-based repetition is sensitive to recurring appearance patterns, but cannot tell whether repeated content is semantically identical.

### 4.6 `clip_f.*`

These are long-range visual semantic consistency metrics based on CLIP image embeddings.

Each sampled frame is encoded by CLIP image encoder. Segment features are then averaged and normalized.

Similarity to the first segment:

```text
s_k = cos(z_k, z_1)
```

Reported fields:

- `clip_f.mean`
- `clip_f.end`
- `clip_f.drop`

Interpretation:

- High `clip_f.mean` means the video stays semantically close to the beginning on average.
- High `clip_f.end` means the ending still resembles the opening semantically.
- High `clip_f.drop` means semantic consistency decays over time.

Compared with `colorhist_lc`, `clip_f` is usually more informative about subject identity, scene semantics, and major content drift.

Limitations:

- CLIP is not optimized specifically for video temporal reasoning.
- It may miss subtle geometric or local identity changes.
- It may be influenced by broad scene semantics rather than exact frame fidelity.

### 4.7 `drift_clip.*`

These metrics turn CLIP similarity into semantic drift:

```text
drift_k = 1 - cos(z_k, z_1), for k >= 2
```

Reported fields:

- `drift_clip.mean`
- `drift_clip.end`
- `drift_clip.max`

Interpretation:

- Lower is better.
- This is often the most practical proxy for semantic drift in long generated videos.
- `drift_clip.max` is especially useful for catching a catastrophic middle segment even when the average looks acceptable.

### 4.8 `repetition_clip.*`

These metrics detect repeated semantic states using CLIP segment features over distant segment pairs.

Reported fields:

- `repetition_clip.ratio`
- `repetition_clip.mean_sim`
- `repetition_clip.max_sim`
- `repetition_clip.num_pairs`

Interpretation:

- High repetition suggests looping or semantic recycling.
- If `repetition_clip` is high while `clip_f` is also high, the model may be too static.
- If `repetition_clip` is high while `clip_f` is low, the model may be oscillating between a few recurrent states rather than progressing naturally.

### 4.9 `clip_t.*`

These metrics measure prompt alignment over time using CLIP image-text similarity.

Let `t` be the CLIP text embedding of the prompt, and `z_k` be the CLIP segment image embedding:

```text
a_k = cos(z_k, t)
```

Reported fields:

- `clip_t.mean`
- `clip_t.end`
- `clip_t.drop`

Interpretation:

- Higher `clip_t.mean` means better average text-video alignment.
- Higher `clip_t.end` means the ending remains aligned with the prompt.
- Larger `clip_t.drop` means the model starts aligned but drifts away from the prompt later.

Limitations:

- CLIP text-image similarity is only a proxy for prompt faithfulness.
- It is weaker for detailed actions, fine-grained chronology, counting, and exact spatial relations.

### 4.10 `dinov2_lc.*`, `drift_dinov2.*`, `repetition_dinov2.*`

These metrics are analogous to the CLIP-image metrics, but use DINOv2 image features instead of CLIP features.

Interpretation:

- `dinov2_lc.*` often emphasizes visual structure and appearance stability.
- `drift_dinov2.*` reflects non-text-conditioned appearance drift.
- Comparing DINOv2 against CLIP can help distinguish purely visual drift from prompt-related semantic drift.

These are toolkit-native metrics, not VBench metrics.

### 4.11 `efficiency.*`

These fields are derived from an optional runtime sidecar JSONL file.

Possible fields include:

- `efficiency.runtime_sec`
- `efficiency.gpu_mem_peak_gb`
- `efficiency.latency_ms`
- `efficiency.kv_cache_gb`
- `efficiency.denoise_sec`
- `efficiency.decode_sec`
- `efficiency.io_sec`
- `efficiency.generation_fps = metadata.num_frames / runtime_sec`

Interpretation:

- These metrics measure cost, not output quality.
- They become most useful when paired with quality or consistency metrics to analyze efficiency-quality tradeoffs.

## 5. How to analyze metrics jointly

Single metrics are often ambiguous. The strongest analysis comes from combining them.

### 5.1 Semantic drift vs. appearance drift

Compare:

- `clip_f.*` and `drift_clip.*`
- `colorhist_lc.*` and `drift_colorhist.*`
- optionally `dinov2_lc.*` and `drift_dinov2.*`

Typical patterns:

- `clip_f` drops but `colorhist_lc` stays high:
  appearance remains similar, but semantic content changes. This often means identity drift, pose drift, or action drift under a stable background.
- `colorhist_lc` drops but `clip_f` stays relatively high:
  semantics remain roughly consistent, but lighting, palette, or exposure shifts strongly.
- both degrade together:
  likely full-scene drift or global generation instability.

### 5.2 Prompt misalignment vs. internal self-consistency

Compare:

- `clip_t.*`
- `clip_f.*`

Typical patterns:

- high `clip_f`, low `clip_t`:
  the video is self-consistent, but consistently wrong relative to the prompt.
- high `clip_t`, low `clip_f`:
  segments may each be individually prompt-like, but the long video does not preserve identity or scene state well.
- both high:
  strong candidate for successful long-video generation.
- both low:
  model fails both prompt following and long-range consistency.

### 5.3 Quality degradation over time

Compare:

- `quality_proxy.head`
- `quality_proxy.tail`
- `quality_proxy.delta_drop`
- `quality_proxy.segment_std`
- `drift_clip.end`

Typical patterns:

- large positive `quality_proxy.delta_drop` with rising semantic drift:
  later segments are both lower quality and more semantically unstable.
- large `quality_proxy.segment_std` with modest drift:
  the video may suffer from local flicker, exposure oscillation, or intermittent sharpness collapse rather than monotonic drift.
- stable quality proxy but poor `clip_t`:
  clean-looking output can still be semantically wrong.

### 5.4 Frozen video, loops, and mode collapse

Compare:

- `repetition_clip.*`
- `repetition_colorhist.*`
- `clip_f.*`
- `quality_proxy.segment_std`

Typical patterns:

- high repetition and extremely high `clip_f`:
  the model may be nearly frozen, repeating the same visual state.
- high `repetition_clip` with moderate or low `clip_f`:
  the video may cycle among a small set of semantic states.
- high `repetition_colorhist` but lower `repetition_clip`:
  the appearance palette repeats, but content still changes.

### 5.5 Detecting early-good, late-bad failures

Watch:

- `clip_f.drop`
- `clip_t.drop`
- `quality_proxy.delta_drop`
- `drift_clip.end`

This pattern is common in long-video generation:

- low early drift,
- decent first-segment quality,
- then worse prompt alignment and worse quality near the end.

When all three drops are large, the model likely cannot sustain long-horizon generation.

### 5.6 Efficiency-quality tradeoff

Compare:

- `efficiency.runtime_sec`
- `efficiency.generation_fps`
- `efficiency.gpu_mem_peak_gb`
- any quality target such as `clip_f.mean`, `clip_t.mean`, `quality_proxy.mean`

Typical use:

- pick the Pareto frontier instead of maximizing one metric blindly.
- a slower model is only justified if it produces materially better consistency or prompt alignment.

## 6. Suggested analysis workflow

For model comparison, a practical workflow is:

1. Start with `model_summary.csv` to find coarse winners and obvious failures.
2. Inspect `clip_f.mean.mean`, `clip_t.mean.mean`, `quality_proxy.mean.mean`, and drift metrics together.
3. Check `per_video_metrics.csv` for variance across prompts.
4. Sort by worst `clip_t.drop`, worst `drift_clip.end`, and worst `quality_proxy.delta_drop`.
5. Open those videos and verify whether the failure is semantic drift, quality collapse, or repetition.

This "summary first, per-video second" workflow avoids overreacting to a single cherry-picked prompt.

## 7. What is and is not VBench

The repository includes a VBench wrapper, but does not reimplement official VBench dimensions.

Therefore:

- `quality_proxy.*` is not VBench `imaging_quality`.
- `clip_t.*` is not official VBench text alignment.
- `clip_f.*`, `colorhist_lc.*`, `drift_*`, `repetition_*`, `dinov2_*` are all toolkit-native proxies.

If you run official VBench through the wrapper or through `longvideo-eval run --enable-vbench`, those imported fields should be labeled explicitly, for example:

- `vbench.imaging_quality`
- `vbench.subject_consistency`
- `vbench.motion_smoothness`

That separation keeps reports scientifically clean and avoids confusing proxy diagnostics with benchmark scores.

## 8. Practical cautions

- Do not compare models across very different resolutions, durations, or prompt sets without normalization.
- Do not treat any single proxy metric as ground truth.
- Low drift does not necessarily mean good motion. It may also mean the video hardly changes.
- High prompt alignment does not guarantee correct chronology or precise action execution.
- Always validate important conclusions by opening representative videos, especially the best and worst cases by each metric family.
