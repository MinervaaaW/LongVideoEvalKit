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

### 4.11 `paper.*`

These fields implement the exact paper-style metrics that were added after v0.1, rather than the earlier lightweight proxies.

They are designed to match the metric definitions commonly used in Context Forcing, Rolling Forcing, Deep Forcing, and Relax Forcing more closely.

#### 4.11.1 `paper.aesthetic_quality_drift_abs`, `paper.imaging_quality_drift_abs`

These are exact first-vs-last quality drift metrics.

For a quality scorer `Q`, define:

```text
DeltaQ(v) = |Q(v_first_5s) - Q(v_last_5s)|
```

The toolkit computes two concrete variants:

- `paper.aesthetic_quality_drift_abs`: uses official VBench `aesthetic_quality` on the first and last clips.
- `paper.imaging_quality_drift_abs`: uses official VBench `imaging_quality` on the first and last clips.

Interpretation:

- Lower is better.
- High values mean the ending quality differs substantially from the opening.
- These are more faithful to the long-video literature than `quality_proxy.delta_abs`, because they use official quality scorers instead of a lightweight heuristic.

Important note:

- These fields are derived from official VBench scoring on extracted clips, but they are not direct whole-video `vbench.*` fields.

#### 4.11.2 `paper.imaging_quality_drift_std`, `paper.imaging_quality_num_clips`

This measures quality fluctuation over time instead of only start-vs-end drift.

The video is partitioned into consecutive clips. For each clip `c_i`, official VBench `imaging_quality` is computed:

```text
q_i = IQ(c_i)
paper.imaging_quality_drift_std = Std(q_1, ..., q_n)
```

Interpretation:

- Lower is better.
- High values indicate unstable quality over time, such as intermittent blur, artifact bursts, or exposure oscillation.
- `paper.imaging_quality_num_clips` records how many clips contributed to the standard deviation.

#### 4.11.3 `paper.drift_clip_first_last`

This is the exact CLIP first-vs-last semantic drift metric:

```text
Drift(v) = 1 - cos(phi_CLIP(v_first_5s), phi_CLIP(v_last_5s))
```

Here each clip feature is computed by averaging frame-level CLIP image embeddings within the clip and then normalizing.

Interpretation:

- Lower is better.
- This is stricter than `drift_clip.end` because it compares explicit first and last clips rather than averaging over all later segments.

#### 4.11.4 `paper.repetition_clip_global`, `paper.repetition_clip_global.num_pairs`

This is a global CLIP repetition metric over clip pairs:

```text
Rep(v) = Avg_{i<j} cos(phi_CLIP(c_i), phi_CLIP(c_j))
```

Interpretation:

- Higher values mean the video revisits similar semantic states more often.
- Very high repetition can indicate looping, freezing, or semantic recycling.
- `paper.repetition_clip_global.num_pairs` records the number of clip pairs used.

Compared with `repetition_clip.*`, this metric is closer to the paper-style all-pairs formulation and does not depend on a similarity threshold.

#### 4.11.5 `paper.balance.mean`

This implements the Relax Forcing-style balance metric:

```text
Balance = minmax(Drift) + minmax(Repetition)
```

Interpretation:

- Lower is better.
- A model with low drift but extremely high repetition is not truly good.
- A model with low repetition but very high drift is also not good.
- This metric captures the tradeoff between those two failure modes.

Important note:

- `paper.balance.mean` is computed at the model-summary level, not as a raw per-video primitive.
- Because it uses min-max scaling, it is only meaningful when comparing models within the same evaluation batch.

#### 4.11.6 `paper.dinov2_cf.*`

These fields implement the Context Forcing-style DINOv2 consistency metric using window-based sampling.

For each evaluation time `t`, define a centered window `[t-0.5s, t+0.5s]`. Let `V_0` be the first frame and `V_tau` be frames inside that window:

```text
DINOv2(t) = Avg_{tau in [t-0.5s, t+0.5s]} cos(phi_DINO(V_tau), phi_DINO(V_0))
```

The toolkit reports:

- `paper.dinov2_cf.mean`
- `paper.dinov2_cf.end`
- `paper.dinov2_cf.drop`

Interpretation:

- High values mean the video remains structurally similar to the opening frame under the Context Forcing protocol.
- Compared with `dinov2_lc.*`, this version is closer to the paper definition because it uses frame-centered local windows instead of segment-mean features alone.

#### 4.11.7 Comparison with earlier toolkit proxies

The exact paper-style metrics do not make the earlier toolkit-native metrics obsolete. They serve different roles.

| Exact paper-style metric | Earlier toolkit metric | Main similarity | Main difference | Recommended use |
|---|---|---|---|---|
| `paper.aesthetic_quality_drift_abs` / `paper.imaging_quality_drift_abs` | `quality_proxy.delta_abs`, `quality_proxy.delta_drop` | Both measure start-vs-end quality change. | `paper.*` uses official VBench quality scorers on extracted clips; `quality_proxy.*` uses a lightweight sharpness/exposure heuristic on sampled frames. | Use `paper.*` for benchmarking and paper-style reporting; use `quality_proxy.*` for cheap debugging. |
| `paper.imaging_quality_drift_std` | `quality_proxy.segment_std` | Both measure quality fluctuation over time. | `paper.*` uses official `imaging_quality`; `quality_proxy.*` uses the lightweight proxy. | Prefer `paper.*` when quality rigor matters. |
| `paper.drift_clip_first_last` | `drift_clip.end` | Both describe semantic change from the beginning toward the end. | `paper.*` compares explicit first and last clips only; `drift_clip.end` comes from the segment-vs-first-segment formulation. | Use `paper.*` for exact first-last drift; keep `drift_clip.*` for richer temporal diagnostics. |
| `paper.repetition_clip_global` | `repetition_clip.mean_sim`, `repetition_clip.ratio`, `repetition_clip.max_sim` | Both measure semantic repetition across time. | `paper.*` is an all-pairs clip-average without thresholds; `repetition_clip.*` uses distant-pair filtering and can expose thresholded repetition events. | Use `paper.*` for paper-style repetition score; use `repetition_clip.*` to inspect looping behavior in more detail. |
| `paper.dinov2_cf.*` | `dinov2_lc.*` | Both use DINOv2 to measure long-range structural consistency. | `paper.*` follows the Context Forcing window-centered protocol against the first frame; `dinov2_lc.*` compares segment-mean features against the first segment. | Use `paper.*` when matching Context Forcing; use `dinov2_lc.*` for a simpler stable proxy. |
| `paper.balance.mean` | none | none | `paper.balance.mean` is a new derived summary metric combining drift and repetition after min-max normalization. | Use only for within-batch model comparison, not as a standalone per-video score. |

Practical rule of thumb:

- If you are writing a paper table, prefer the `paper.*` fields.
- If you are diagnosing why a model failed, keep the older proxy fields as supporting evidence.

### 4.12 `efficiency.*`

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

### 4.13 `vbench.*`

These fields come from official VBench. The toolkit does not reimplement them. Instead, it invokes official VBench in `custom_input` mode, stores the raw artifacts, and merges standardized `vbench.*` columns back into toolkit reports.

Currently integrated dimensions are:

- `vbench.aesthetic_quality`
- `vbench.imaging_quality`
- `vbench.subject_consistency`
- `vbench.background_consistency`
- `vbench.motion_smoothness`
- `vbench.dynamic_degree`

Interpretation:

- `vbench.aesthetic_quality`: official VBench judgment of overall visual appeal.
- `vbench.imaging_quality`: official VBench judgment of visual quality. Use this instead of `quality_proxy.*` when you need an official benchmark quality number.
- `vbench.subject_consistency`: official VBench judgment of whether the main subject remains consistent over time.
- `vbench.background_consistency`: official VBench judgment of whether the background stays coherent and stable.
- `vbench.motion_smoothness`: official VBench judgment of whether motion appears smooth and temporally natural.
- `vbench.dynamic_degree`: official VBench judgment of motion intensity. This helps separate "stable because it is correct" from "stable because it barely moves."

Reporting behavior:

- `model_summary.csv` stores merged model-level `vbench.*` columns whenever the official outputs can be parsed.
- `per_video_metrics.csv` stores `vbench.*` only when official outputs contain stable per-video rows that can be matched back to generated videos.
- Raw official artifacts stay under `vbench_raw/`.
- Standardized toolkit exports are written to `vbench_merged_summary.csv/jsonl` and, when available, `vbench_merged_per_video.csv/jsonl`.

## 5. How to analyze metrics jointly

Single metrics are often ambiguous. The strongest analysis comes from combining them.

### 5.1 Semantic drift vs. appearance drift

Compare:

- `clip_f.*` and `drift_clip.*`
- `colorhist_lc.*` and `drift_colorhist.*`
- optionally `dinov2_lc.*` and `drift_dinov2.*`
- for exact paper-style consistency, optionally `paper.dinov2_cf.*`

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

- `paper.aesthetic_quality_drift_abs`
- `paper.imaging_quality_drift_abs`
- `paper.imaging_quality_drift_std`
- `quality_proxy.head`
- `quality_proxy.tail`
- `quality_proxy.delta_drop`
- `quality_proxy.segment_std`
- `drift_clip.end`

Typical patterns:

- large `paper.imaging_quality_drift_abs` or `paper.aesthetic_quality_drift_abs`:
  the end of the video is noticeably worse than the beginning under official quality scorers.
- large `paper.imaging_quality_drift_std`:
  quality fluctuates throughout the video, even if the start and end happen to look similar on average.
- large positive `quality_proxy.delta_drop` with rising semantic drift:
  later segments are both lower quality and more semantically unstable.
- large `quality_proxy.segment_std` with modest drift:
  the video may suffer from local flicker, exposure oscillation, or intermittent sharpness collapse rather than monotonic drift.
- stable quality proxy but poor `clip_t`:
  clean-looking output can still be semantically wrong.

When official VBench is available:

- high `vbench.imaging_quality` with low `quality_proxy.mean`:
  the lightweight proxy may be underestimating the video; trust the official VBench score more for benchmarking.
- low `vbench.imaging_quality` with acceptable `quality_proxy.mean`:
  the video may be sharp enough but still contain artifacts or unnatural details that the lightweight proxy misses.

### 5.4 Frozen video, loops, and mode collapse

Compare:

- `paper.repetition_clip_global`
- `paper.balance.mean` when comparing multiple models
- `repetition_clip.*`
- `repetition_colorhist.*`
- `clip_f.*`
- `quality_proxy.segment_std`

Typical patterns:

- high repetition and extremely high `clip_f`:
  the model may be nearly frozen, repeating the same visual state.
- low `paper.drift_clip_first_last` but high `paper.repetition_clip_global`:
  the model may preserve the opening semantics mainly by not evolving enough.
- high `repetition_clip` with moderate or low `clip_f`:
  the video may cycle among a small set of semantic states.
- high `repetition_colorhist` but lower `repetition_clip`:
  the appearance palette repeats, but content still changes.

When official VBench is available:

- high `vbench.motion_smoothness` but low `vbench.dynamic_degree`:
  motion may be smooth mainly because the video is too static.
- high `vbench.dynamic_degree` but low `vbench.motion_smoothness`:
  the model produces enough motion, but the motion is jerky, unstable, or visually implausible.

### 5.5 Detecting early-good, late-bad failures

Watch:

- `clip_f.drop`
- `clip_t.drop`
- `paper.aesthetic_quality_drift_abs`
- `paper.imaging_quality_drift_abs`
- `quality_proxy.delta_drop`
- `drift_clip.end`

This pattern is common in long-video generation:

- low early drift,
- decent first-segment quality,
- then worse prompt alignment and worse quality near the end.

When all three drops are large, the model likely cannot sustain long-horizon generation.

If official VBench is also available, compare these failures against:

- `vbench.subject_consistency`
- `vbench.background_consistency`

This helps separate "late-stage semantic drift" from "late-stage visual degradation." For example, low `vbench.subject_consistency` with moderate `vbench.background_consistency` often indicates that the subject drifts while the broader scene scaffold remains relatively intact.

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
- `paper.*` fields are paper-style derived metrics. Some of them reuse official VBench clip-level scorers, but they are still derived toolkit outputs rather than native whole-video `vbench.*` fields.

If you run official VBench through the wrapper or through `longvideo-eval run --enable-vbench`, those imported fields should be labeled explicitly, for example:

- `vbench.aesthetic_quality`
- `vbench.imaging_quality`
- `vbench.background_consistency`
- `vbench.subject_consistency`
- `vbench.motion_smoothness`
- `vbench.dynamic_degree`

That separation keeps reports scientifically clean and avoids confusing proxy diagnostics with benchmark scores.

## 8. Practical cautions

- Do not compare models across very different resolutions, durations, or prompt sets without normalization.
- Do not treat any single proxy metric as ground truth.
- Low drift does not necessarily mean good motion. It may also mean the video hardly changes.
- High prompt alignment does not guarantee correct chronology or precise action execution.
- Always validate important conclusions by opening representative videos, especially the best and worst cases by each metric family.
