# Development Roadmap

This roadmap is intentionally written as a TODO list. The first priority is to test and stabilize the functionality that already exists before expanding the surface area.

## Working Principles

- [x] Prioritize fast end-to-end validation over completeness.
- [x] Use the current 2-sample fixture as the default first-line test path.
- [x] Enable optional modules only after the base pipeline passes.
- [x] Defer hardening, fallback, and extra documentation until the main test path is done.

## v0.1 Stabilization

Goal: run the current toolkit successfully and verify the main output paths with the smallest realistic dataset.

### v0.1.A Core pipeline smoke coverage

- [x] Test prompt-directory layout dataset loading.
- [x] Test `video_selection=latest`.
- [x] Test prompt matching against `prompt_dir`.
- [x] Test dataset coverage reporting.
- [x] Test per-video report writing.
- [x] Test model-summary report writing.

#### Current 2-sample smoke-test baseline

Use the current 2-video local dataset as the default smoke-test fixture for v0.1 validation. The purpose is not statistical evaluation; it is end-to-end pipeline validation on the smallest realistic batch already available in the repo.

This subsection defines the shared fixture, base-pipeline expectations, and debugging order only.

Recommended debugging order for the 2-sample fixture:

- [x] First run the base pipeline with the 2-sample fixture and treat it as the default first-line debugging path.
- [x] Use the base-pipeline smoke test to quickly isolate chain-level issues before enabling any heavy optional models.
- [x] Only after the base pipeline passes, enable optional modules one at a time in this order:
  - native CLIP
  - native DINOv2
  - integrated official VBench
- [x] If a later stage fails, treat the failure as scoped to the newly enabled module until proven otherwise.

- [x] Freeze the current 2-sample fixture as the default smoke-test dataset and document its exact paths.
- [x] Record expected runtime artifacts for the 2-sample smoke test.
- [x] Record expected coverage behavior.
- [x] Record expected base success criteria.
- [x] Record acceptable smoke-test exclusions.

### v0.1.B Native lightweight metrics

- [x] Test metadata extraction.
- [x] Test `quality_proxy.*`.
- [x] Test `colorhist_lc.*`.
- [x] Test `drift_colorhist.*`.
- [x] Test `repetition_colorhist.*`.

### v0.1.C Toolkit-native CLIP support

Current scope:

- [x] OpenAI CLIP local-cache loading
- [x] local cache default at `~/.cache/clip`
- [x] optional local `clip` source repo via `--clip-repo`
- [x] `clip_f.*`
- [x] `clip_t.*`
- [x] `drift_clip.*`
- [x] `repetition_clip.*`

Remaining validation tasks:

- [x] Smoke-test `--enable-clip` with local `~/.cache/clip/ViT-B-32.pt`.
- [ ] Confirm whether the current 2-sample CLIP run covers all expected CLIP columns.

### v0.1.D Toolkit-native DINOv2 support

Current scope:

- [x] local-only loading from `~/.cache/dinov2-base`
- [x] Hugging Face `local_files_only=True`
- [x] `dinov2_lc.*`
- [x] `drift_dinov2.*`

Remaining validation tasks:

- [x] Smoke-test `--enable-dinov2` with the current local cache.
- [x] Confirm DINOv2 columns appear in output.

### v0.1.E Official VBench integration

Current integrated VBench dimensions:

- [x] `aesthetic_quality`
- [x] `imaging_quality`
- [x] `subject_consistency`
- [x] `background_consistency`
- [x] `motion_smoothness`
- [x] `dynamic_degree`

Current wrapper behavior:

- [x] run official VBench in `custom_input`
- [x] local-only checkpoint mode via `--load_ckpt_from_local True`
- [x] merged model-level summary export
- [x] merged per-video export when parseable
- [x] proxy retry for `socks5h://` parse failures

Remaining validation tasks:

- [ ] Smoke-test each VBench dimension individually with real local assets.
- [ ] Verify that all six dimensions succeed from the integrated `longvideo-eval run --enable-vbench` path.
- [ ] Verify local cache reuse for:
  - `~/.cache/clip/*`
  - `~/.cache/vbench/clip_model/*` symlinks

### v0.1.F Model cache and environment docs

- [x] add `MODEL_CACHE_SETUP.md`
- [x] document local CLIP cache
- [x] document local DINOv2 cache
- [x] document local VBench cache
- [x] document official download URLs



## v0.2 VBench-Long integration

Goal: support long-video evaluation through `VBench-Long` in a way that matches the current wrapper UX.

### v0.2.A Wrapper and configuration

- [ ] Add a `vbench_variant` config option:
  - `vbench`
  - `vbench_long`
- [ ] Add CLI support for selecting the VBench variant.
- [ ] Teach the wrapper to dispatch to:
  - `vbench evaluate` for standard VBench
  - `python -m vbench2_beta_long.eval_long` for VBench-Long
- [ ] Add support for `long_custom_input`.
- [ ] Add support for `long_vbench_standard`.

### v0.2.B VBench-Long-specific arguments

- [ ] Add config and CLI plumbing for:
  - `use_semantic_splitting`
  - `dev_flag`
  - `static_filter_flag`
  - `clip_length_config`
  - `slow_fast_eval_config`
  - `bg_clip2clip_feat_extractor`
  - `sb_clip2clip_feat_extractor`
  - `w_inclip`
  - `w_clip2clip`
  - `num_of_samples_per_prompt`

### v0.2.C Output parsing

- [ ] Implement `parse_vbench_long_outputs()`.
- [ ] Parse `*_eval_results.json` emitted by `VBench-Long`.
- [ ] Merge summary-level `vbench.*` scores first.
- [ ] Decide whether clip-level details should be exposed in toolkit reports.
- [ ] Add tests for:
  - summary-only parsing
  - fused `subject_consistency`
  - fused `background_consistency`
  - dimensions returning reorganized long-video results

### v0.2.D Dependencies and preflight

- [ ] Add preflight checks for:
  - `vbench2_beta_long`
  - `ffmpeg`
  - `scenedetect`
  - `dreamsim`
  - other long-path dependencies as needed
- [ ] Add local-model documentation for any extra VBench-Long assets.
- [ ] Add smoke tests on one small long-video fixture.

## v0.3 VBench++ integration list

This milestone tracks the broader VBench++ family, not just VBench-Long.

### v0.3.A VBench++ inventory

- [ ] Document the supported upstream components and their status:
  - `VBench`
  - `VBench-Long`
  - `VBench-I2V`
  - `VBench-Trustworthiness`

### v0.3.B VBench-I2V

- [ ] Evaluate whether current dataset abstractions can represent image-to-video inputs cleanly.
- [ ] Add a separate wrapper path for `vbench2_beta_i2v`.
- [ ] Decide how to represent I2V-specific outputs in the current CSV schema.
- [ ] Add tests for I2V command construction and parsing.

### v0.3.C VBench-Trustworthiness

- [ ] Evaluate whether trustworthiness dimensions belong in the same report namespace or a separate one.
- [ ] Add wrapper support for `vbench2_beta_trustworthiness`.
- [ ] Add model-cache documentation for any trustworthiness-specific assets.
- [ ] Add tests for command construction and summary import.

## v0.4 Reporting and diagnostics

Goal: make the current metrics easier to inspect and compare.

- [ ] Add HTML report generation.
- [ ] Add per-model summary tables in HTML.
- [ ] Add time-curve plots for long-range consistency and drift metrics.
- [ ] Add per-prompt comparison views across models.
- [ ] Add report annotations for partial / missing metrics.
- [ ] Add bootstrap confidence intervals for model-level aggregates.
- [ ] Calibrate repetition thresholds by metric family.
- [ ] Add a diagnostic section that groups metrics by:
  - quality
  - consistency
  - motion
  - efficiency

## v1.0 Benchmark productization

Goal: move from an internal evaluation toolkit to a reproducible benchmark package.

- [ ] Add VLM-judge support for qualitative scoring.
- [ ] Add human evaluation formats:
  - 2AFC
  - Likert
  - ELO / K-factor ranking
- [ ] Define leaderboard-ready output schema.
- [ ] Add reproducibility cards per run:
  - model version
  - prompt source
  - cache state
  - environment
  - commit hash
- [ ] Add benchmark prompt-suite versioning.
- [ ] Add release checklist for public benchmark submissions.

## Immediate Next Actions

- [x] Run the current test suite and confirm it passes after every roadmap-related change.
- [x] Perform base smoke test.
- [x] Perform native CLIP smoke test.
- [x] Perform native DINOv2 smoke test.
- [ ] Finish integrated VBench smoke test.

## Deferred Hardening

- [ ] Test no-video failure path and error messaging.
- [ ] Test `video_selection=all`.
- [ ] Test standard-layout dataset loading.
- [ ] Test prompt matching against `prompt_file`.
- [ ] Add edge-case tests for short or empty segment cases.
- [ ] Smoke-test `--clip-cache-dir` override.
- [ ] Test long prompt truncation behavior for OpenAI CLIP tokenization.
- [ ] Test local-file failure path when the CLIP checkpoint is missing.
- [ ] Test optional `open_clip` backend with a local checkpoint file.
- [ ] Document a known-good OpenCLIP checkpoint naming convention for local files.
- [ ] Add a regression test for missing `~/.cache/dinov2-base`.
- [ ] Confirm expected files for both `model.safetensors` and `pytorch_model.bin` cases.
- [ ] Verify behavior on CPU-only fallback for native DINOv2 metrics.
- [ ] Add a test for missing local VBench assets preflight.
- [ ] Add a test for merged summary generation across multiple dimensions.
- [ ] Add a test for empty-or-unparseable raw VBench outputs.
- [ ] Document which VBench assets can be symlinks and which must remain separate.
- [ ] Add troubleshooting and cache-maintenance notes.
