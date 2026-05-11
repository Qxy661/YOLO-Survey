# YOLOv4 与 YOLOv5: 目标检测的工业时代 (2020)

> **阅读时间**: ~12 min | **前置知识**: YOLOv1-v3 基本原理、CSPDarknet、PANet/FPN

---

## 目录

1. [YOLOv4: 技巧的百科全书](#1-yolov4-技巧的百科全书)
2. [YOLOv5: 工程化的里程碑](#2-yolov5-工程化的里程碑)
3. [YOLOv4 vs YOLOv5 全面对比](#3-yolov4-vs-yolov5-全面对比)
4. [延伸阅读](#4-延伸阅读)
5. [思考题](#5-思考题)

---

## 1. YOLOv4: 技巧的百科全书

**论文**: *YOLOv4: Optimal Speed and Accuracy of Object Detection* (Bochkovskiy et al., 2020)

### 1.1 核心思想

YOLOv4 的核心贡献并非全新架构，而是 **系统性地验证和组合大量已有技巧**。作者将技巧分为两类：

| 分类 | 英文名称 | 含义 | 影响 |
|------|---------|------|------|
| **训练技巧** | Bag of Freebies (BoF) | 仅增加训练成本，不增加推理成本 | 提升精度 |
| **推理技巧** | Bag of Specials (BoS) | 少量增加推理成本，显著提升精度 | 精度/速度权衡 |

### 1.2 Bag of Freebies (训练技巧)

### Mosaic 数据增强

将 4 张训练图像拼接成一张：

```
    +----------+----------+
    |          |          |
    |  img1    |  img2    |
    |          |          |
    +----------+----------+
    |          |          |
    |  img3    |  img4    |
    |          |          |
    +----------+----------+
     Mosaic: 4 图拼接
```

**优势**: 丰富图像背景、等效增大 batch size、减少对大 batch 的依赖。

### CutMix 数据增强

将一张图像的区域裁剪后粘贴到另一张图像，按面积比例混合标签：

```
y = lambda * y_a + (1 - lambda) * y_b
lambda = 1 - (裁剪区域面积 / 总面积)
```

### CIoU Loss 详解

```
CIoU = IoU - rho^2(b, b_gt) / c^2 - alpha * v

其中:
  rho^2(b, b_gt) = 预测框与真实框中心点欧氏距离平方
  c^2            = 最小外接矩形对角线长度平方
  v  = (4/pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2   # 宽高比一致性
  alpha = v / ((1 - IoU) + v)                              # 权衡因子

L_CIou = 1 - CIoU
```

**IoU Loss 演进链**:

```
IoU Loss  -->  仅重叠区域
                 + 距离
GIoU Loss -->  重叠区域 + 最小外接矩形
                 + 聚焦中心
DIoU Loss -->  重叠区域 + 中心距离
                 + 宽高比
CIoU Loss -->  重叠区域 + 中心距离 + 宽高比  <-- YOLOv4 采用
```

### 其他训练技巧

| 技巧 | 作用 | 效果 |
|------|------|------|
| **DropBlock** | 随机丢弃特征图连续区域（比 Dropout 更有效） | +1.0% mAP |
| **Label Smoothing** | 硬标签 (0/1) 转软标签 (0.05/0.95) | 防止过拟合 |
| **Self-Adversarial Training (SAT)** | 先对输入做对抗扰动，再用扰动图像训练 | 增强鲁棒性 |
| **Cosine Annealing LR** | 学习率按余弦函数衰减 | 更平滑收敛 |

### 1.3 Bag of Specials (推理技巧)

### Mish 激活函数

```
Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

对比:
  ReLU:  x<0 时输出 0 (死神经元问题)
  Mish:  x<0 时输出接近 0 但非零 (平滑, 梯度不中断)
```

### CSPDarknet53: YOLOv4 的 Backbone

将 CSP (Cross Stage Partial) 结构应用于 Darknet53：

```
CSP 结构:

输入 --+---------------------------+
       |                           |
       v                           |
  +---------+                      |
  | Part 1  | (直通分支)            |
  +----+----+                      |
       |                           v
       |                    +-----------+
       |                    |  Part 2   | (处理分支)
       |                    | Dense Block|
       |                    +-----+-----+
       |                          |
       v                          v
  +-----------------------------------+
  |     Concat + 1x1 Conv (压缩)      |
  +------------------+----------------+
                     v
                   输出
```

**CSP 核心优势**: 减少约 20% FLOPs，精度几乎不降，梯度通过直通分支更有效回传。

### SPP (Spatial Pyramid Pooling)

```
输入特征图
    |
    +---- MaxPool 5x5 ----+
    +---- MaxPool 9x9 ----+----> Concat + 1x1 Conv ----> 输出
    +---- MaxPool 13x13 --+
```

增大感受野，捕获多尺度上下文，对不同输入尺寸鲁棒。

### PANet (双向特征融合)

```
Backbone         FPN (自顶向下)       PANet (双向)

C3 (大尺度) --> P3 -----------> P3 (最终)
  |               ^ (上采样+add)   ^ (下采样+add)
C4 (中尺度) --> P4 -----------> P4 (最终)
  |               ^               ^
C5 (小尺度) --> P5 -----------> P5 (最终)

FPN 自顶向下传递语义信息
PAN 自底向上传递定位信息
双向融合同时增强语义与定位
```

### SAM (Spatial Attention Module)

```
输入特征图 F (CxHxW)
    |
    +-- MaxPool (通道维度) --> F_max --+
    +-- AvgPool (通道维度) --> F_avg --+--> Concat + Conv 7x7 + Sigmoid
                                                                |
输出 = F * M(F)  (逐元素相乘)  <-------------------------------+
```

### DIoU-NMS

```
传统 NMS:   if IoU(box_i, box_j) > threshold --> 抑制
DIoU-NMS:   if IoU - rho^2(b_i, b_j) / c^2 > threshold --> 抑制

优势: 即使 IoU 较大但中心距离远时，不会被误抑制
     适用于密集遮挡场景
```

### 1.4 YOLOv4 完整网络架构

```
输入图像 (416x416x3)
        |
        v
+-------------------------------+
|       CSPDarknet53            |  <-- Backbone
|  (Mish 激活 + CSP 结构)        |     特征提取
|  输出: C3, C4, C5             |
+---------------+---------------+
                |
+---------------v---------------+
|    SPP + PANet + SAM          |  <-- Neck
|  (多尺度池化 + 双向融合)       |     特征增强
+---------------+---------------+
                |
+---------------v---------------+
|   YOLO Head (3 个检测头)       |  <-- Head
|   P3: 小目标 (52x52)          |     多尺度检测
|   P4: 中目标 (26x26)          |
|   P5: 大目标 (13x13)          |
+---------------+---------------+
                |
          DIoU-NMS 后处理
                v
          检测结果输出
```

### 1.5 YOLOv4 性能

| 指标 | YOLOv4 (416) | YOLOv4 (608) |
|------|:------------:|:------------:|
| mAP@0.5 (COCO) | 65.7% | 68.4% |
| AP@0.5:0.95 | 43.5% | 47.3% |
| FPS (V100) | 62 | 42 |
| 参数量 | 64.4M | 64.4M |
| FLOPs | 58.5B | 127.5B |

---

## 2. YOLOv5: 工程化的里程碑

**发布**: Ultralytics, 2020 年 6 月（无正式论文，代码即文档）

### 2.1 Focus Layer

YOLOv5 在输入端使用 Focus 层进行切片操作：

```
输入图像 (H x W x 3)
每隔一个像素取样，得到 4 个子图:

原始像素排列:             切片结果:
+--------------+         +------+ +------+
| a b c d e f  |   -->   |a c e | |b d f |
| g h i j k l  |         |g i k | |h j l |
| m n o p q r  |         |m o q | |n p r |
+--------------+         +------+ +------+
                          拼接后通道数 x4

Focus: (H, W, 3) --> (H/2, W/2, 12)
Conv 3x3:          --> (H/2, W/2, 64)
```

**优势**: 无信息丢失地将空间信息转为通道信息，减少 FLOPs。

### 2.2 CSP 结构变体

YOLOv5 中 CSP 有两种变体：

```
CSP1_X (用于 Backbone)          CSP2_X (用于 Neck)
输入 --+--------+                输入 --+--------+
       |        |                       |        |
       v        v                       v        v
     Conv    Conv                    Conv    Conv
       |        |                       |        |
       |   X x ResBlock                |   X x ConvBlock
       |   (有残差连接)                 |   (无残差)
       |        |                       |        |
       v        v                       v        v
    Concat + Conv 1x1               Concat + Conv 1x1
```

| 特性 | CSP1_X (Backbone) | CSP2_X (Neck) |
|------|:-----------------:|:-------------:|
| 内部块类型 | ResBlock | Conv Block |
| 残差连接 | 有 | 无 |
| 典型 X 值 | 1, 2, 8, 8, 4 | 1, 2, 3 |

### 2.3 自适应 Anchor 计算

YOLOv5 实现了 **自动 anchor 计算**：

```
1. 加载训练集所有标注框的 (w, h)
2. 运行 K-Means 聚类 (k=9), 距离度量: 1 - IoU
3. 遗传算法进化, 适应度函数: 最大化召回率
4. 按面积分配到 3 个检测层 (P3/P4/P5)
```

### 2.4 YOLOv5 完整网络结构

```
输入 (640x640x3)
        |
        v
+-----------------------------------+
|          Focus Layer              |  <-- 输入端
|  (640,640,3) --> (320,320,64)     |
+-----------------+-----------------+
                  |
+-----------------v-----------------+
|       CSPDarknet Backbone         |  <-- Backbone
|  Conv 3x3 --> CSP1_1              |
|  Conv 3x3 --> CSP1_2              |
|  Conv 3x3 --> CSP1_8              |
|  Conv 3x3 --> CSP1_8              |
|  Conv 3x3 --> CSP1_4              |
|  输出: P3(80x80) P4(40x40) P5(20x20)|
+-----------------+-----------------+
                  |
+-----------------v-----------------+
|       SPP + PANet Neck            |  <-- Neck
|  SPP: MaxPool (5/9/13) + Concat   |
|  PAN: 自顶向下 + 自底向上          |
|  CSP2_X: 特征融合                  |
+-----------------+-----------------+
                  |
+-----------------v-----------------+
|     Detect Head (3 个检测头)       |  <-- Head
|  P3 (80x80): 小目标, 3 anchors    |
|  P4 (40x40): 中目标, 3 anchors    |
|  P5 (20x20): 大目标, 3 anchors    |
+-----------------+-----------------+
                  |
            NMS 后处理
```

### 2.5 Scaled Variants: 从 Nano 到 Extra Large

YOLOv5 提供 5 种缩放变体：

```
变体     width_multiple  depth_multiple   定位
-------  --------------  --------------   --------
YOLOv5n  0.25            0.33             Nano (嵌入式)
YOLOv5s  0.50            0.33             Small (边缘设备)
YOLOv5m  0.75            0.67             Medium (平衡)
YOLOv5l  1.00            1.00             Large (高精度)
YOLOv5x  1.25            1.33             Extra Large (最高精度)

width_multiple: 控制通道数
depth_multiple: 控制 CSP 中 ResBlock 重复次数
```

**性能对比 (COCO val2017)**：

| 变体 | 输入 | mAP@0.5 | mAP@0.5:0.95 | 推理 (V100) | 参数量 | FLOPs |
|------|:----:|:-------:|:------------:|:-----------:|:------:|:-----:|
| YOLOv5n | 640 | 45.7% | 28.0% | 3.0ms | 1.9M | 4.5B |
| YOLOv5s | 640 | 56.8% | 37.4% | 3.4ms | 7.2M | 16.5B |
| YOLOv5m | 640 | 64.1% | 45.4% | 5.3ms | 21.2M | 49.0B |
| YOLOv5l | 640 | 67.4% | 49.0% | 7.2ms | 46.5M | 109.1B |
| YOLOv5x | 640 | 69.1% | 50.7% | 10.1ms | 86.7M | 205.7B |

### 2.6 YOLOv5 完整生态

```
数据准备        训练            验证            部署
+--------+    +--------+    +--------+    +--------+
|标注工具 | --> |train   | --> |val     | --> |export  |
|格式转换 |     |超参进化 |     |混淆矩阵|     |推理引擎 |
|数据增强 |     |多GPU   |     |PR曲线  |     |量化    |
+--------+    +--------+    +--------+    +--------+
                                              |
                       +----------+-----------+-----------+
                       |          |           |           |
                   TensorRT   CoreML      TFLite      NCNN
                   OpenVINO   ONNX
```

---

## 3. YOLOv4 vs YOLOv5 全面对比

| 对比维度 | YOLOv4 | YOLOv5 |
|---------|:------:|:------:|
| **框架** | Darknet (C/CUDA) | PyTorch (Python) |
| **论文** | 有完整论文 | 无正式论文 (代码即文档) |
| **Backbone** | CSPDarknet53 | CSPDarknet (可配置深度) |
| **Neck** | SPP + PANet + SAM | SPP + CSP2_X + PANet |
| **Head** | YOLOv3 Head + DIoU-NMS | 解耦头 + NMS |
| **输入尺寸** | 416/512/608 (固定) | 320-1280 (自适应) |
| **Anchor** | K-Means 手动聚类 | AutoAnchor 自动进化 |
| **激活函数** | Mish (backbone) | SiLU (全网络) |
| **Loss** | CIoU Loss | CIoU Loss + BCE (分类) |
| **部署格式** | Darknet 权重 | ONNX/TensorRT/CoreML 等 |
| **易用性** | 较低 (需编译) | 极高 (pip install) |
| **精度 (AP)** | 43.5% | 50.7% (v5x) |
| **速度 (FPS)** | 62 (V100, 416) | 100+ (V100, v5s) |

---

## 4. 延伸阅读

- **YOLOv4 论文**: Bochkovskiy et al., [arXiv:2004.10934](https://arxiv.org/abs/2004.10934), 2020
- **CSPNet 论文**: Wang et al., "CSPNet: A New Backbone that can Enhance Learning Capability of CNN", CVPR Workshops 2020
- **PANet 论文**: Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018
- **YOLOv5 仓库**: [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Mish 激活函数**: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function", BMVC 2020
- **CIoU Loss**: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020

---

## 5. 思考题

<details>
<summary>Q1: Mosaic vs CutMix -- 两种数据增强策略各适用于什么场景？</summary>

Mosaic 通过 4 图拼接丰富背景上下文，适合场景理解任务；CutMix 通过区域替换增强局部特征的鲁棒性，适合遮挡场景。两者互补：Mosaic 增加空间多样性，CutMix 增加语义多样性。YOLOv4 同时使用两者以获得更全面的增强效果。
</details>

<details>
<summary>Q2: CSP 结构为什么能减少计算量？从信息冗余角度分析。</summary>

传统 DenseNet 中所有层的输出都传递给后续层，存在大量信息冗余。CSP 将特征分为两部分：一部分直通保留原始信息，一部分经过 Dense Block 提取新特征。两者拼接后融合，避免了重复学习已有特征，减少约 20% FLOPs。
</details>

<details>
<summary>Q3: Focus Layer 为什么不会丢失信息？相比直接 3x3 Conv 下采样有何优势？</summary>

Focus 将空间像素按奇偶位置分为 4 组，每组变为独立通道，信息完全保留（可逆操作）。优势：无信息丢失地完成 2x 下采样；减少 FLOPs（无需 MaxPool）；每个输出像素的输入来自原始图像的 4 个不同位置，隐式扩大感受野。
</details>

<details>
<summary>Q4: YOLOv4 有论文但复现困难，YOLOv5 无论文但开箱即用。哪种方式对社区贡献更大？</summary>

两者贡献不同维度：YOLOv4 的论文系统性总结了训练/推理技巧，为后续研究提供方法论框架；YOLOv5 的代码极大降低工业应用门槛，推动目标检测的普及。从实际影响看，YOLOv5 的 GitHub stars 和工业采用率远超 YOLOv4，但 YOLOv4 的学术引用更持久。两者共同推动了 YOLO 生态繁荣。
</details>

<details>
<summary>Q5: DIoU-NMS 在什么情况下可能不如传统 NMS？</summary>

当两个目标物理上重叠但属于不同类别时（如行人手持物体），DIoU-NMS 因中心距离近而可能错误保留两个框。此外，目标尺寸差异极大时，DIoU 距离的归一化可能产生偏差。传统 NMS 在类别感知模式下更稳定。
</details>

---

*本文为 YOLO Survey 系列第二章第二节，下一节将介绍 YOLOv6 ~ YOLOv7 的竞争时代。*

## 延伸阅读

### 相关文档

- [01-YOLOv1-v3奠基时代](01-YOLOv1-v3奠基时代.md) — YOLO 早期演进
- [03-YOLOv6-v7竞争时代](03-YOLOv6-v7竞争时代.md) — YOLOv6/v7 竞争格局
- [04-YOLOv8统一架构](04-YOLOv8统一架构.md) — YOLOv8 统一架构设计
