# paired_videos_metrics

`paired_videos_metrics` 是 `LongVideoEvalKit` 中负责**成对视频评测**的子模块，用于评测：

- GT 视频 vs 预测视频
- 模型 A 输出 vs 模型 B 输出
- 视频预测 / 重建 / 对齐生成任务

它输出四类常用指标：

- `PSNR`
- `SSIM`
- `LPIPS`
- `FVD`

与主工具 `longvideo-eval run` 的区别是：

- `longvideo-eval run` 面向“长视频集合评测”，结合 prompt 和长时一致性指标
- `paired_videos_metrics` 面向“成对视频评测”，衡量一个预测目录与一个参考目录的接近程度

## 推荐入口

推荐通过主工具 CLI 调用，而不是直接单独运行脚本：

```bash
longvideo-eval paired-run \
  --gt-dir /path/to/gt \
  --pred-dir /path/to/pred \
  --output-dir ./outputs/paired_eval \
  --device cuda
```

如果需要，也可以直接运行本目录下的脚本：

```bash
python paired_videos_metrics/batch_eval_paired_videos.py \
  --gt-dir /path/to/gt \
  --pred-dir /path/to/pred \
  --output-dir ./outputs/paired_eval \
  --device cuda
```

## 适用场景

适合以下任务：

- 视频预测
- 视频重建
- 有明确 GT 的 video-to-video 任务
- 两个模型输出结果的逐视频对比

不适合直接替代长视频集合评测，因为它回答的问题不同：

- paired 指标回答的是“这组预测和这组参考有多接近”
- 长视频集合指标回答的是“这个模型作为整体生成器表现如何”

## 指标说明

本模块输出：

- `psnr_mean`
- `ssim_mean`
- `lpips_mean`
- `fvd_pair`
- `fvd_dataset`

方向如下：

- `PSNR`：越高越好
- `SSIM`：越高越好
- `LPIPS`：越低越好
- `FVD`：越低越好

更详细的指标含义、公式、结果分析方法，请看：

- [METRICS_GUIDE_ZH.md](./METRICS_GUIDE_ZH.md)

## 输入要求

输入是两个目录：

- `--gt-dir`
- `--pred-dir`

目录内支持的视频后缀：

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.webm`

## 配对规则

模块会自动按以下顺序配对：

1. 完整文件名精确匹配
2. 去掉扩展名后的 stem 精确匹配
3. 去掉扩展名后的 stem 从后向前后缀匹配

这使它能处理类似下面这种情况：

- GT: `057dfd1fffc9b7771a5c9c1c12c5acab.mp4`
- Pred: `001_step-6000_057dfd1fffc9b7771a5c9c1c12c5acab_057dfd1fffc9b7771a5c9c1c12c5acab.mp4`

如果某些视频无法配对：

- 未匹配的 GT 视频不会参与评测
- 未匹配的预测视频也不会参与评测
- 配对报告中会明确记录这些条目

## 时长对齐规则

如果配对成功的两个视频时长不一致：

- 取较短时长作为对齐长度
- 将长视频从**开始位置**截取到相同时长
- 再做后续指标计算

输出字段中会保留：

- `gt_duration_sec`
- `pred_duration_sec`
- `aligned_duration_sec`
- `duration_truncated`

## 输出文件

运行完成后，输出目录通常包含：

```text
output_dir/
  config.json
  pairing_report.json
  pairing_report.csv
  per_video_metrics.jsonl
  per_video_metrics.csv
  summary.json
  final_results.json
```

各文件作用如下：

- `config.json`
  - 本次评测配置和统计摘要
- `pairing_report.json`
  - 配对过程的完整 JSON 报告
- `pairing_report.csv`
  - 配对过程的表格版本，便于人工筛查
- `per_video_metrics.jsonl`
  - 每条视频的详细评测结果，JSONL 格式
- `per_video_metrics.csv`
  - 每条视频的详细评测结果，CSV 格式
- `summary.json`
  - 数据集级汇总结果
- `final_results.json`
  - 汇总配置、配对信息和逐视频结果的一份总 JSON

## 配对报告说明

`pairing_report.json` 和 `pairing_report.csv` 合并记录三类信息：

- `non_exact_match`
  - 成功参与评测，但不是通过完整文件名精确匹配上的视频
- `unmatched_gt`
  - GT 中存在、但预测目录中没有匹配上的视频
- `unmatched_pred`
  - 预测目录中存在、但 GT 中没有匹配上的视频

也就是说，现在不再拆成多个零散的小文件，而是统一看一份配对报告即可。

## 依赖与模型权重

运行本模块通常需要：

- `torch`
- `torchvision`
- `numpy`
- `opencv-python-headless` 或 `opencv-python`
- `scipy`
- `tqdm`

FVD 权重：

- 仓库内已自带 `styleganv` 与 `videogpt` 的 I3D 权重

LPIPS 权重：

- LPIPS 线性层权重已随仓库提供
- 默认使用 AlexNet backbone
- 需要本地 torchvision AlexNet 缓存，通常位于：
  - `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`

更详细的本地缓存说明请看：

- [../MODEL_CACHE_SETUP.md](../MODEL_CACHE_SETUP.md)

## 常用命令示例

### 1. 常规 paired 评测

```bash
conda run -n eval longvideo-eval paired-run \
  --gt-dir /data/gt \
  --pred-dir /data/pred \
  --output-dir ./outputs/my_paired_eval \
  --device cuda
```

### 2. 只计算 PSNR / SSIM / LPIPS，跳过 FVD

```bash
conda run -n eval longvideo-eval paired-run \
  --gt-dir /data/gt \
  --pred-dir /data/pred \
  --output-dir ./outputs/my_paired_eval_no_fvd \
  --device cuda \
  --skip-fvd
```

### 3. 只抽样测试前 N 条配对

```bash
conda run -n eval longvideo-eval paired-run \
  --gt-dir /data/gt \
  --pred-dir /data/pred \
  --output-dir ./outputs/my_paired_eval_smoke \
  --device cpu \
  --max-videos 5
```

## 结果解读建议

如果你的任务是有明确 GT 的重建或预测：

- 优先看 `PSNR / SSIM / LPIPS`

如果你更关心整组视频的动态分布差异：

- 结合看 `fvd_pair` 和 `fvd_dataset`

特别注意：

- `fvd_pair` 和 `fvd_dataset` 可能给出不同倾向
- `PSNR / SSIM` 高，不一定表示视频时序自然
- `LPIPS` 更偏感知相似性

因此建议：

- 帧级指标和视频级指标一起看
- 数值分析和人工视频检查一起做

## 相关文件

- 主工具 README：[../README.md](../README.md)
- 中文总介绍：[../介绍.md](../介绍.md)
- 模型缓存说明：[../MODEL_CACHE_SETUP.md](../MODEL_CACHE_SETUP.md)
- 中文指标指南：[./METRICS_GUIDE_ZH.md](./METRICS_GUIDE_ZH.md)
