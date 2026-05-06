# AEFM-Net: 自适应增强选择融合情绪识别网络

基于 Event 相机与 RGB 图像融合的单眼情绪识别，面向 VR/AR 智能头戴设备场景。在 SEE 数据集上实现 UAR 91.6%、WAR 91.1%，覆盖正常/过曝/低光/HDR 四种光照条件。


## 环境安装

```bash
conda create -n snn_cmm python=3.10
conda activate snn_cmm
pip install -r requirements.txt
```

> 原始环境：Python 3.10.19 + PyTorch 2.4.0+cu118 + CUDA 11.8，在 RTX 3090 (24GB) 上运行。
> `mamba-ssm` 仅支持 Linux + CUDA，Windows 上无法直接安装。若未安装，SS2D 会静默 fallback 为恒等映射，CMM 跨模态融合模块失效。

## 数据集

使用 SEE (Single-eye Event-based Emotion) 数据集，包含 111 名志愿者、2,405 个序列、128,712 帧，覆盖 7 种情绪（angry, disgust, fear, happiness, neutral, sadness, surprise）和 4 种光照条件（normal, overexposure, low-light, HDR）。训练/测试划分为 1,638 / 767 序列。

数据集需按以下结构放置：

```
SEE/
├── frame/                    # RGB 帧
│   ├── angry/
│   │   └── {video_id}/
│   │       ├── 00001.jpg
│   │       └── ...
│   ├── happiness/
│   ├── sadness/
│   ├── neutral/
│   ├── fear/
│   ├── surprise/
│   └── disgust/
├── event_30/                 # Event 帧（对应结构）
│   └── ...
└── emotion_new_adjust2.json  # 标注文件
```

标注 JSON 格式：

```json
{
  "labels": ["angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
  "database": {
    "{video_id}": {
      "subset": "training" | "testing",
      "annotations": {
        "label": "angry",
        "segment": [start_frame, end_frame]
      }
    }
  }
}
```

## 训练与推理

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --root_path /root/Single-eye-Emotion-Recognition/single-eye-emotion/SEE \
  --event_video_path event_30 \
  --frame_video_path frame \
  --annotation_path emotion_new_adjust2.json \
  --result_path /root/Single-eye-Emotion-Recognition/single-eye-emotion/results/resnet_cmm_test \
  --dataset emotion \
  --n_classes 7 \
  --batch_size 32 \
  --n_threads 16 \
  --inference \
  --no_val \
  --sample_size 90 \
  --no_hflip \
  --sample_duration 4 \
  --inference_batch_size 120 \
  --inference_stride 0 \
  --sample_t_stride 4 \
  --inference_sample_duration 4 \
  --thresh 0.3 \
  --lens 0.5 \
  --decay 0.2 \
  --colorjitter \
  --n_epochs 140
```

| 参数 | 说明 |
|---|---|
| `--root_path` | 数据集根目录 |
| `--sample_size` | 输入帧尺寸 (90×90) |
| `--sample_duration` | 训练时采样帧数 |
| `--sample_t_stride` | 帧采样间隔 |
| `--inference_sample_duration` | 推理时采样帧数 |
| `--inference` | 训练期间开启定期推理评测（从 epoch 40 开始，每 3 epoch 一次） |
| `--no_val` | 关闭 validation |

> **注意**：`--thresh`、`--lens`、`--decay` 三个参数是 SEEN 框架遗留的 SNN 超参，在当前模型中已无实际作用，保留仅为兼容性。学习率调度器在代码中硬编码为 `StepLR(step_size=1, gamma=0.94)`，`--lr_scheduler`、`--multistep_milestones`、`--plateau_patience` 参数同样无效。

## 可视化

修改各脚本中的数据集路径和 `CHECKPOINT_PATH` 后运行：

| 脚本 | 功能 |
|---|---|
| `visualize_hdr.py` | LACM 低光补偿效果对比 |
| `visualize_safm_heatmap.py` | SAFM 噪声抑制热力图 |
| `visualize_asfm_weights.py` | 四种光照下 RGB/Event 模态贡献率 |
| `visualize_tsne.py` | 七种情绪 t-SNE 特征聚类 |

## 实验结果

| 指标 | 数值 |
|---|---|
| UAR | 91.6% |
| WAR | 91.1% |

| 正常 | 过度曝光 | 低光照 | 高动态 |
|---|---|---|---|
| 87.3% | 93.9% | 94.4% | 88.5% |

## 模型架构

```
RGB帧 ──► LACM ──► ResNet18 ──► SAFM ──┐
                                        ├──► ACMF (Mamba) ──► ConvLSTM ──► 分类
Event帧 ──► ResNet18 ──────────► SAFM ──┘
```

## 代码 ↔ 论文对应

| 代码类名 | 论文名称 |
|---|---|
| `HDREnhancer` | LACM 低光自适应补偿模块 |
| `MFCM` / `MSC` | SAFM 稀疏注意力融合模块 |
| `CMMBlock` / `SS2D` | ACMF 特征增强（Mamba 部分） |
| `ASFM` | ACMF 自适应特征选择机制 |
| `ConvLSTM` | ConvLSTM 时序特征建模 |
