# YOLOv1 ~ YOLOv3: 奠基时代

> **阅读时间**: ~12 min | **前置知识**: CNN 基础、目标检测概述
>
> 从 2016 年到 2018 年，YOLO 系列完成了从"概念验证"到"工业级检测器"的蜕变。本文梳理 YOLOv1、YOLOv2/YOLO9000、YOLOv3 三代模型的核心思想、网络结构与关键技术。

---

## 目录

1. [YOLOv1: 开创性的单阶段检测器](#1-yolov1-开创性的单阶段检测器)
2. [YOLOv2/YOLO9000: 全面升级](#2-yolov2yolo9000-全面升级)
3. [YOLOv3: 多尺度检测的里程碑](#3-yolov3-多尺度检测的里程碑)
4. [三代 YOLO 对比总结](#4-三代-yolo-对比总结)
5. [延伸阅读](#5-延伸阅读)
6. [思考题](#6-思考题)

---

## 1. YOLOv1: 开创性的单阶段检测器

**论文**: *You Only Look Once: Unified, Real-Time Object Detection* (Redmon et al., CVPR 2016)

### 1.1 核心思想

在 YOLO 之前，主流检测器（如 R-CNN 系列）采用 **两阶段** 流水线：先生成候选区域，再对每个区域分类与回归。YOLO 的突破在于将目标检测 **直接转化为一个回归问题**，用单个卷积网络一次性完成边界框定位和类别预测。

```
输入图像 --> 单次前向传播 --> S x S 网格 --> [bbox 坐标 + 置信度 + 类别概率]
```

### 1.2 网络结构

YOLOv1 的网络由 **24 个卷积层** 和 **2 个全连接层** 组成，灵感来源于 GoogLeNet。

```
输入: 448 x 448 x 3
  |
  v
+-----------------------------+
| Conv 7x7, 64, stride 2     |   --> 224x224x64
| MaxPool 2x2, stride 2      |
| Conv 3x3, 192               |   --> 112x112x192
| MaxPool 2x2, stride 2      |
+-----------------------------+
| 8x [Conv 1x1 + Conv 3x3]   |   --> 28x28x512
+-----------------------------+
| MaxPool 2x2, stride 2      |
| 4x [Conv 1x1 + Conv 3x3]   |   --> 14x14x1024
+-----------------------------+
| Conv 3x3 + Conv 3x3         |   --> 7x7x1024
| FC: 4096 -> 1470            |
+-----------------------------+
  |
  v
输出: 7 x 7 x 30 (S=7, B=2, C=20)
```

### 1.3 输出编码详解

YOLOv1 将输入图像划分为 **S x S** 的网格（S=7）。对于每个网格单元（cell）：

- 预测 **B 个边界框**（B=2），每个 bbox 包含 5 个值：`(x, y, w, h, confidence)`
  - `(x, y)`: bbox 中心相对于该 cell 的偏移（归一化到 [0,1]）
  - `(w, h)`: bbox 宽高相对于整张图像的归一化值
  - `confidence = Pr(Object) * IoU(pred, truth)`
- 预测 **C 个类别条件概率** `Pr(Class_i | Object)`（VOC 数据集 C=20）

每个 cell 的输出向量长度：

```
B * 5 + C = 2 * 5 + 20 = 30
```

因此整体输出张量为 `7 x 7 x 30`。

### 1.4 损失函数

YOLOv1 使用 **平方和损失**（sum-squared error），包含三部分：

```
Loss = lambda_coord * Σ [定位损失]
     + Σ [置信度损失(有目标)]
     + lambda_noobj * Σ [置信度损失(无目标)]
     + Σ [分类损失]

其中 lambda_coord = 5, lambda_noobj = 0.5
```

宽度/高度损失使用 sqrt 以惩罚小框的误差：

```
(sqrt(w_i) - sqrt(w_hat_i))^2 + (sqrt(h_i) - sqrt(h_hat_i))^2
```

### 1.5 YOLOv1 的局限性

| 局限 | 说明 |
|------|------|
| 每个 cell 仅预测 2 个 bbox | 密集目标或同类多目标靠近时漏检严重 |
| 小目标检测能力弱 | 7x7 网格粒度太粗，小物体特征被池化丢失 |
| 定位精度不高 | 平方和损失对大框和小框的误差一视同仁 |
| 泛化能力有限 | 对不常见长宽比或新尺寸的目标表现不佳 |

---

## 2. YOLOv2/YOLO9000: 全面升级

**论文**: *YOLO9000: Better, Faster, Stronger* (Redmon & Farhadi, CVPR 2017)

YOLOv2 是一次 **系统性的工程改进**，作者逐项分析并解决了 YOLOv1 的不足。

### 2.1 关键改进一览

| 改进项 | 说明 | 效果 |
|--------|------|------|
| **Batch Normalization** | 所有卷积层后加 BN | mAP +2%，可去除 Dropout |
| **High Resolution Classifier** | 分类预训练用 448x448（v1 用 224x224） | mAP +4% |
| **Anchor Boxes** | 去掉 FC 层，用 anchor 机制预测偏移量 | 召回率 81% -> 88% |
| **Dimension Clusters** | k-means 聚类最优 anchor 尺寸 | 更好匹配数据分布 |
| **Direct Location Prediction** | 偏移限制在 cell 范围内 | 训练更稳定 |
| **Fine-Grained Features** | Passthrough 层拼接高分辨率特征 | 小目标检测改善 |
| **Multi-Scale Training** | 每 10 batch 随机切换输入尺寸 | 多尺度鲁棒性增强 |

### 2.2 Anchor Boxes 机制

YOLOv2 借鉴 Faster R-CNN 的 anchor 思想，但用 **k-means IoU 距离** 聚类：

```
d(box, centroid) = 1 - IoU(box, centroid)

COCO 数据集 k=5 聚类结果:
  (0.57273, 0.677385)    # 小目标
  (1.87446, 2.06253)     # 中目标
  (3.33843, 5.47434)     # 中大目标
  (7.88282, 3.52778)     # 宽扁目标
  (9.77052, 9.16828)     # 大目标
```

**Direct Location Prediction**（约束在 cell 内，训练稳定）：

```
b_x = sigma(t_x) + c_x      # sigma 压缩到 (0,1)
b_y = sigma(t_y) + c_y      # c_x, c_y: cell 左上角坐标
b_w = p_w * e^(t_w)          # p_w, p_h: anchor 宽高
b_h = p_h * e^(t_h)
```

### 2.3 Darknet-19 骨干网络

YOLOv2 用 **Darknet-19**（19 卷积层 + 5 MaxPool）替代 VGG 风格骨干：

```
Darknet-19: 全部 3x3 和 1x1 卷积 + BN

Layer 0-2:   Conv 3x3(32) -> MaxPool -> Conv 3x3(64) -> MaxPool
Layer 4-6:   Conv 3x3(128) -> Conv 1x1(64) -> Conv 3x3(128) -> MaxPool
Layer 8-10:  Conv 3x3(256) -> Conv 1x1(128) -> Conv 3x3(256) -> MaxPool
Layer 12-16: Conv 3x3(512) -> [1x1(256)->3x3(512)] x2 -> MaxPool (Passthrough)
Layer 18-22: Conv 3x3(1024) -> [1x1(512)->3x3(1024)] x2
```

**Passthrough Layer**: 将 26x26x512 reshape 为 13x13x2048，与 13x13x1024 拼接，保留细粒度定位信息。

### 2.4 Multi-Scale Training

训练时每 10 个 mini-batch 随机选取输入尺寸：

```
{320, 352, 384, 416, 448, 480, 512, 544, 576, 608}
（均为 32 的倍数，因网络下采样 5 次，32 = 2^5）
```

---

## 3. YOLOv3: 多尺度检测的里程碑

**论文**: *YOLOv3: An Incremental Improvement* (Redmon & Farhadi, 2018)

YOLOv3 引入了 **多尺度预测** 和 **更强的骨干网络**，使 YOLO 首次具备与顶级检测器可比的精度。

### 3.1 Darknet-53 骨干网络

YOLOv3 用 **Darknet-53** 替代 Darknet-19，核心引入 **残差连接**：

```
Darknet-53 结构:

Input: 416x416x3
  |
  v
Conv 3x3, 32, stride 1    --> 416x416x32
Conv 3x3, 64, stride 2    --> 208x208x64     (下采样 1)
  +--- Residual Block x1
Conv 3x3, 128, stride 2   --> 104x104x128    (下采样 2)
  +--- Residual Block x2
Conv 3x3, 256, stride 2   --> 52x52x256      (下采样 3) --> Scale 1
  +--- Residual Block x8
Conv 3x3, 512, stride 2   --> 26x26x512      (下采样 4) --> Scale 2
  +--- Residual Block x8
Conv 3x3, 1024, stride 2  --> 13x13x1024     (下采样 5) --> Scale 3
  +--- Residual Block x4
```

**残差块**: `y = x + F(x)`，梯度可直接从输出传到输入，解决深层网络梯度消失问题。

**Darknet-53 vs ResNet 对比**：

| 网络 | 层数 | Top-1 Acc | Top-5 Acc | BFLOP/s | FPS |
|------|------|-----------|-----------|---------|-----|
| Darknet-19 | 19 | 74.1% | 91.8% | 1246 | - |
| ResNet-101 | 101 | 77.1% | 93.7% | 1520 | - |
| ResNet-152 | 152 | 77.6% | 93.8% | 1890 | - |
| **Darknet-53** | **53** | **77.2%** | **93.8%** | **1870** | **78** |

Darknet-53 精度与 ResNet-152 相当，但推理速度更快。

### 3.2 多尺度预测 (Multi-Scale Prediction)

YOLOv3 最关键的改进是 **三个不同尺度的检测头**：

```
                    +-----------------------+
                    |     Scale 3 (大目标)   |
                    |     13 x 13 网格       |
                    |     3 anchors          |
                    +-----------+-----------+
                                |
                          Upsample x2 + Concat
                                |
                    +-----------+-----------+
                    |     Scale 2 (中目标)   |
                    |     26 x 26 网格       |
                    |     3 anchors          |
                    +-----------+-----------+
                                |
                          Upsample x2 + Concat
                                |
                    +-----------+-----------+
                    |     Scale 1 (小目标)   |
                    |     52 x 52 网格       |
                    |     3 anchors          |
                    +-----------------------+

总 anchor 数: 3 x (13x13 + 26x26 + 52x52) = 10920
```

### 3.3 FPN 特征融合

YOLOv3 采用 **Feature Pyramid Network (FPN)** 自顶向下特征融合：

```
13x13x1024 --> Conv 1x1 --> 检测头 (大目标)
    |
    v
Upsample 2x + Concat <-- 26x26x512
    |
    v
26x26x768 --> Conv 1x1 --> 检测头 (中目标)
    |
    v
Upsample 2x + Concat <-- 52x52x256
    |
    v
52x52x512 --> Conv 1x1 --> 检测头 (小目标)
```

### 3.4 Anchor 分配

YOLOv3 为每个尺度分配 3 个 anchor（共 9 个），通过 k-means 聚类获得：

| 尺度 | 特征图 | 感受野 | Anchor 尺寸 (COCO) |
|------|--------|--------|---------------------|
| Scale 1 (小目标) | 52x52 | 小 | (10x13), (16x30), (33x23) |
| Scale 2 (中目标) | 26x26 | 中 | (30x61), (62x45), (59x119) |
| Scale 3 (大目标) | 13x13 | 大 | (116x90), (156x198), (373x326) |

### 3.5 分类方式: Softmax -> Sigmoid

YOLOv3 将分类从 **softmax 多分类** 改为 **独立 sigmoid 二分类**：

```
YOLOv1/v2: softmax  -> 类别互斥，同一目标只能属于一个类别
YOLOv3:    sigmoid  -> 每个类别独立判断，支持多标签分类

Pr(Class_i) = 1 / (1 + exp(-z_i))
```

这允许一个目标同时被标记为多个类别（如 "人" 和 "女人"）。

---

## 4. 三代 YOLO 对比总结

### 4.1 核心架构对比

| 特性 | YOLOv1 (2016) | YOLOv2 (2017) | YOLOv3 (2018) |
|------|---------------|---------------|---------------|
| **骨干网络** | 24 Conv + 2 FC | Darknet-19 | Darknet-53 |
| **输入尺寸** | 448x448 (固定) | 320~608 (多尺度) | 320~608 (多尺度) |
| **网格大小** | 7x7 | 13x13 | 13/26/52 (3 尺度) |
| **Anchor 机制** | 无 | 有 (k-means, 5 个) | 有 (k-means, 9 个) |
| **每 cell 预测 bbox** | 2 | 5 | 3 (每个尺度) |
| **检测尺度数** | 1 | 1 | 3 |
| **残差连接** | 无 | 无 | 有 |
| **特征融合** | 无 | Passthrough | FPN |
| **分类方式** | Softmax | Softmax | Sigmoid (多标签) |
| **BN 层** | 无 | 全部 | 全部 |

### 4.2 性能对比 (COCO test-dev)

| 模型 | mAP@0.5 | mAP@[0.5:0.95] | FPS (Titan X) |
|------|---------|----------------|---------------|
| YOLOv2 (416) | 44.0% | 21.6% | 67 |
| YOLOv3 (320) | 51.5% | 28.2% | 45 |
| YOLOv3 (416) | 55.3% | 33.0% | 35 |
| YOLOv3 (608) | 57.9% | 35.4% | 20 |

### 4.3 技术演进路线图

```
YOLOv1 (2016)
  |  核心突破: 单阶段回归检测
  |  主要不足: 精度低、小目标差
  v
YOLOv2 (2017)
  |  工程优化: BN + 高分辨率 + Anchor + k-means
  |  骨干升级: Darknet-19
  |  训练技巧: Multi-Scale Training
  v
YOLOv3 (2018)
  |  骨干升级: Darknet-53 + 残差连接
  |  检测升级: 多尺度预测 (3 scales)
  |  特征融合: FPN
  |  分类改进: Sigmoid 多标签
  v
后续: YOLOv4, v5, v6, v7, v8, v9, v10, v11...
```

---

## 5. 延伸阅读

- **YOLOv1 原文**: [arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
- **YOLOv2 原文**: [arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- **YOLOv3 原文**: [arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- **FPN 论文**: [arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144) -- 特征金字塔网络
- **Faster R-CNN**: [arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497) -- Anchor 机制来源
- **Darknet 框架**: [pjreddie.com/darknet](https://pjreddie.com/darknet/) -- YOLO 原始实现框架
- **Batch Normalization**: [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

---

## 6. 思考题

<details>
<summary>Q1: YOLOv1 为什么用 sqrt(w) 和 sqrt(h) 来计算宽高损失？</summary>

直接使用 w 和 h 计算平方差时，同样 1 像素的误差对大框和小框的惩罚是一样的，但从 IoU 角度看，1 像素误差对小框的影响远大于大框。使用 sqrt 可以放大 小框的误差贡献，使得模型对小目标的定位更敏感。
</details>

<details>
<summary>Q2: YOLOv2 的 k-means 为什么用 IoU 而不是欧氏距离？</summary>

欧氏距离对大框和小框的误差敏感度不同：两个大框即使 IoU 很高，欧氏距离也可能很大。使用 `d = 1 - IoU` 作为距离度量，可以确保聚类结果在 IoU 意义上是最优的，与检测任务的评估指标一致。
</details>

<details>
<summary>Q3: YOLOv3 为什么从 softmax 改为 sigmoid？</summary>

softmax 假设类别互斥，而实际场景中可能存在多标签情况。sigmoid 对每个类别独立计算概率，允许一个目标同时属于多个类别。此外，sigmoid 的训练信号更直接，每个类别的梯度独立，不受其他类别影响。
</details>

<details>
<summary>Q4: Darknet-53 的残差连接解决了什么问题？</summary>

残差连接解决了深层网络的梯度消失和退化问题。`y = x + F(x)` 提供了一条"梯度高速公路"，梯度可以直接从输出传到输入，不经过中间的非线性层。这使得网络可以安全地加深到 53 层而不出现训练困难。
</details>

<details>
<summary>Q5: 如果让你设计 YOLOv4，你会在 YOLOv3 的基础上做哪些改进？（开放题）</summary>

可以考虑的方向：更强的数据增强（Mosaic、CutMix）、训练策略（SAT、余弦退火）、更高效的骨干（CSPNet）、更精确的检测头（Decoupled Head）、更好的 NMS（DIoU-NMS）、损失函数改进（CIoU Loss）。实际上 YOLOv4 恰好采用了上述大部分改进。
</details>

---

*本文为 YOLO Survey 系列第二章第一节，下一节将介绍 YOLOv4 ~ YOLOv5 的工业时代。*
