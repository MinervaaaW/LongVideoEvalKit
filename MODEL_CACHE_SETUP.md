# Local Model Cache Setup

This repository is now configured for local-only model loading during evaluation. Runtime downloads are no longer the default path for CLIP, DINOv2, or the integrated official VBench dimensions.

In addition to the prompt-aware long-video pipeline, the repository also includes a paired GT-vs-pred evaluation workflow exposed through:

- `longvideo-eval paired-run`

That workflow depends on:

- bundled FVD I3D weights inside `paired_videos_metrics/`
- LPIPS linear weights inside `paired_videos_metrics/lpips/weights/`
- torchvision ImageNet backbone cache for LPIPS, typically under `~/.cache/torch/hub/checkpoints/`

## Code Locations

Toolkit-native model loading:

- `longvideo_eval/pipeline.py`
  - builds the optional feature extractors used by the evaluation pipeline.
- `longvideo_eval/models/features.py`
  - `OpenAICLIPExtractor`: local-only OpenAI CLIP checkpoint loading from `~/.cache/clip`
  - `OpenCLIPExtractor`: local-only checkpoint loading from an explicit local file path
  - `DINOv2Extractor`: local-only Hugging Face directory loading from `~/.cache/dinov2-base`
- `longvideo_eval/config.py`
  - default local paths for `clip_cache_dir`, `dinov2_model`, and the default `openai_clip` backend
- `longvideo_eval/model_defaults.py`
  - shared cache path defaults and VBench asset layout

Paired GT-vs-pred metric loading:

- `paired_videos_metrics/batch_eval_paired_videos.py`
  - paired evaluation entrypoint used by `longvideo-eval paired-run`
- `paired_videos_metrics/fvd/styleganv/fvd.py`
  - loads the bundled `i3d_torchscript.pt`
- `paired_videos_metrics/fvd/videogpt/fvd.py`
  - loads the bundled `i3d_pretrained_400.pt`
- `paired_videos_metrics/lpips/lpips.py`
  - loads LPIPS linear weights from `paired_videos_metrics/lpips/weights/`
  - uses torchvision pretrained image backbones, which may read from `~/.cache/torch/hub/checkpoints/`

Integrated official VBench loading:

- `longvideo_eval/metrics/vbench_wrapper.py`
  - always appends `--load_ckpt_from_local True`
  - validates local VBench asset paths before execution

Official VBench package modules used at runtime:

- `vbench/utils.py`
  - defines the checkpoint paths and download URLs for the VBench dimensions
- `vbench/aesthetic_quality.py`
  - loads the LAION aesthetic head and CLIP backbone
- `vbench/imaging_quality.py`
  - loads MUSIQ-SPAQ
- `vbench/subject_consistency.py`
  - loads the DINO subject-consistency backbone
- `vbench/background_consistency.py`
  - loads the CLIP backbone for background consistency
- `vbench/motion_smoothness.py`
  - loads the AMT-S checkpoint
- `vbench/dynamic_degree.py`
  - loads the RAFT checkpoint

## Metric To Model Mapping

### Toolkit-native optional metrics

| Metric family | Model(s) | Local path |
| --- | --- | --- |
| `clip_f.*`, `clip_t.*`, `drift_clip.*`, `repetition_clip.*` | OpenAI CLIP checkpoint | `~/.cache/clip/ViT-B-32.pt` by default |
| `dinov2_lc.*`, `drift_dinov2.*`, `repetition_dinov2.*` | DINOv2 base directory | `~/.cache/dinov2-base/` |

### Paired GT-vs-pred metrics

| Metric family | Model(s) | Local path |
| --- | --- | --- |
| `fvd_pair`, `fvd_dataset` (`styleganv`) | I3D TorchScript | bundled at `paired_videos_metrics/fvd/styleganv/i3d_torchscript.pt` |
| `fvd_pair`, `fvd_dataset` (`videogpt`) | I3D PyTorch weights | bundled at `paired_videos_metrics/fvd/videogpt/i3d_pretrained_400.pt` |
| `lpips_mean` | LPIPS linear weights | bundled at `paired_videos_metrics/lpips/weights/v0.1/*.pth` |
| `lpips_mean` backbone | torchvision AlexNet by default | `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` |

### Official VBench dimensions integrated by this repo

| VBench dimension | Model(s) | Local path |
| --- | --- | --- |
| `aesthetic_quality` | CLIP `ViT-L-14.pt` + LAION aesthetic head | `~/.cache/vbench/clip_model/ViT-L-14.pt` and `~/.cache/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth` |
| `imaging_quality` | MUSIQ-SPAQ | `~/.cache/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` |
| `subject_consistency` | DINO repo + DINO ViT-B/16 weights | `~/.cache/vbench/dino_model/facebookresearch_dino_main/` and `~/.cache/vbench/dino_model/dino_vitbase16_pretrain.pth` |
| `background_consistency` | CLIP `ViT-B-32.pt` | `~/.cache/vbench/clip_model/ViT-B-32.pt` |
| `motion_smoothness` | AMT-S | `~/.cache/vbench/amt_model/amt-s.pth` |
| `dynamic_degree` | RAFT Things | `~/.cache/vbench/raft_model/models/raft-things.pth` |

For the two VBench CLIP files, the `~/.cache/vbench/clip_model/` paths do not need to be separate physical copies. They can be symlinks to the shared toolkit-native CLIP cache under `~/.cache/clip/`.

## Official Download Sources

### Toolkit-native CLIP

- `ViT-B/32`: `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`
- `ViT-B/16`: `https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e693f9e85e4ecb61988df416f/ViT-B-16.pt`
- `ViT-L/14`: `https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt`

### Toolkit-native DINOv2

- Hugging Face repo: `https://huggingface.co/facebook/dinov2-base`

### Paired LPIPS backbone

- torchvision AlexNet checkpoint
  - `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`

### Official VBench integrated assets

- VBench aesthetic CLIP `ViT-L-14.pt`
  - `https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt`
- VBench background CLIP `ViT-B-32.pt`
  - `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`
- VBench aesthetic head
  - `https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true`
- VBench imaging quality MUSIQ-SPAQ
  - `https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth`
- VBench subject consistency DINO repo
  - `https://github.com/facebookresearch/dino`
- VBench subject consistency DINO weights
  - `https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth`
- VBench motion smoothness AMT-S
  - `https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth`
- VBench dynamic degree RAFT zip
  - `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`

## Download Commands

### 1. Toolkit-native CLIP cache

```bash
mkdir -p ~/.cache/clip
wget -P ~/.cache/clip https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

Optional alternative CLIP checkpoints:

```bash
wget -P ~/.cache/clip https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e693f9e85e4ecb61988df416f/ViT-B-16.pt
wget -P ~/.cache/clip https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

### 2. Toolkit-native DINOv2 cache

```bash
git clone https://huggingface.co/facebook/dinov2-base ~/.cache/dinov2-base
```

### 2.5 Paired LPIPS backbone cache

If you use `longvideo-eval paired-run` with LPIPS enabled, pre-populate the torchvision cache:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
wget -O ~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth \
  https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
```

If you run inside a restricted environment where `~/.cache` is not writable, point `TORCH_HOME` to a writable directory before running `paired-run`.

### 3. Official VBench integrated cache

```bash
mkdir -p ~/.cache/vbench/clip_model
mkdir -p ~/.cache/vbench/aesthetic_model/emb_reader
mkdir -p ~/.cache/vbench/pyiqa_model
mkdir -p ~/.cache/vbench/dino_model
mkdir -p ~/.cache/vbench/amt_model
mkdir -p ~/.cache/vbench/raft_model
```

Recommended CLIP reuse via symlinks:

```bash
mkdir -p ~/.cache/clip ~/.cache/vbench/clip_model
wget -P ~/.cache/clip https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget -P ~/.cache/clip https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
ln -s ~/.cache/clip/ViT-B-32.pt ~/.cache/vbench/clip_model/ViT-B-32.pt
ln -s ~/.cache/clip/ViT-L-14.pt ~/.cache/vbench/clip_model/ViT-L-14.pt
```

If you prefer real files instead of symlinks, use the direct-download variant below.

```bash
wget -P ~/.cache/vbench/clip_model https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget -P ~/.cache/vbench/clip_model https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
wget -O ~/.cache/vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
wget -P ~/.cache/vbench/pyiqa_model https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
git clone https://github.com/facebookresearch/dino ~/.cache/vbench/dino_model/facebookresearch_dino_main
wget -P ~/.cache/vbench/dino_model https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
wget -P ~/.cache/vbench/amt_model https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth
wget -P ~/.cache/vbench/raft_model https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip -d ~/.cache/vbench/raft_model ~/.cache/vbench/raft_model/models.zip
rm ~/.cache/vbench/raft_model/models.zip
```

## Expected Final `~/.cache` Layout

```text
~/.cache/
├── clip/
│   ├── ViT-B-32.pt
│   ├── ViT-B-16.pt                       # optional
│   └── ViT-L-14.pt                       # optional for toolkit-native CLIP
├── dinov2-base/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── model.safetensors
│   ├── pytorch_model.bin
│   └── README.md
└── vbench/
    ├── aesthetic_model/
    │   └── emb_reader/
    │       └── sa_0_4_vit_l_14_linear.pth
    ├── amt_model/
    │   └── amt-s.pth
    ├── clip_model/
    │   ├── ViT-B-32.pt                  # file or symlink to ~/.cache/clip/ViT-B-32.pt
    │   └── ViT-L-14.pt                  # file or symlink to ~/.cache/clip/ViT-L-14.pt
    ├── dino_model/
    │   ├── dino_vitbase16_pretrain.pth
    │   └── facebookresearch_dino_main/
    │       ├── hubconf.py
    │       ├── vision_transformer.py
    │       └── ...
    ├── pyiqa_model/
    │   └── musiq_spaq_ckpt-358bb6af.pth
    └── raft_model/
        └── models/
            └── raft-things.pth
```

## Runtime Notes

- Toolkit-native CLIP metrics now default to the `openai_clip` backend plus `~/.cache/clip`.
- Toolkit-native DINOv2 metrics now default to `~/.cache/dinov2-base`.
- The integrated VBench wrapper now forces `--load_ckpt_from_local True`.
- VBench `clip_model/ViT-B-32.pt` and `clip_model/ViT-L-14.pt` can be symlinks to the shared `~/.cache/clip/` files.
- If a required local file is missing, the toolkit raises a local-path error before evaluation instead of silently downloading.
- The paired workflow uses bundled FVD weights from the repository, but LPIPS may still require the local torchvision AlexNet cache.
