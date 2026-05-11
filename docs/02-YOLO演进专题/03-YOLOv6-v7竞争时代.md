# YOLOv6-v7 竞争时代：工业落地与学术前沿的碰撞

> **阅读时间**: ~15 min | **前置知识**: YOLOv4-v5 基本架构、CSPDarknet、PANet/FPN、Anchor-based 与 Anchor-free 概念

2022 年是 YOLO 家族"百花齐放"的一年。美团推出 YOLOv6 主打工业部署，Wang 等人发布 YOLOv7 刷新学术 SOTA，百度 PP-YOLOE 则在国产生态中占据一席之地。三者几乎同时出现，形成了 YOLO 历史上最激烈的竞争格局。

---

## 目录

1. [YOLOv6 — 美团的工业级方案](#1-yolov6--美团的工业级方案)
2. [YOLOv7 — 学术 SOTA 的捍卫者](#2-yolov7--学术-sota-的捍卫者)
3. [PP-YOLOE — 百度飞桨生态](#3-pp-yoloe--百度飞桨生态)
4. [横向对比](#4-横向对比)
5. [延伸阅读](#5-延伸阅读)
6. [思考题](#6-思考题)

---

## 1. YOLOv6 — 美团的工业级方案

### 1.1 设计哲学

YOLOv6 的核心理念是 **Hardware-aware Design**：不仅追求 mAP，更关注在实际硬件上的推理延迟。美团团队针对不同部署场景（GPU / CPU / NPU）提供了 S / M / L / X 多个尺度的模型。

### 1.2 RepVGG-style Backbone

YOLOv6 的骨干网络采用 **重参数化（Re-parameterization）** 思想，源自 RepVGG。

**训练阶段**：每个 Block 包含三条并行分支：

```
Input
  |
  +---> 3x3 Conv (主分支)
  |        |
  +---> 1x1 Conv (shortcut)  --->  Add ---> Output
  |        |
  +---> Identity (BN only)
```

**推理阶段**：三条分支合并为单个 3x3 卷积核：

```
合并前 (训练时):                合并后 (推理时):
┌──────────────┐               ┌──────────────┐
│  3x3 Conv    │               │              │
│  1x1 Conv    │  ──合并──>    │  单个 3x3 Conv│
│  Identity    │               │              │
└──────────────┘               └──────────────┘
   3 条分支                        1 条分支
   推理较慢                        推理极快
```

合并的数学原理（以卷积核为例）：

```
W_merged = W_3x3 + pad(W_1x3) + pad(W_identity)

其中:
  W_3x3      : 原始 3x3 卷积核
  W_1x1      : 1x1 卷积核, 零填充至 3x3
  W_identity : 恒等映射, 即中心为 1 其余为 0 的 3x3 核

偏置同样合并:
  b_merged = b_3x3 + b_1x1 + 0
```

这种设计让模型在训练时拥有更强的表达能力（多分支梯度流），推理时则获得极致的单路速度。

### 1.3 Efficient Decoupled Head

YOLOv6 借鉴了 YOLOX 的 Decoupled Head 思想，但做了效率优化：

```
Feature Map
    |
    +---> 1x1 Conv (通道压缩)
    |        |
    |        +---> Cls Branch ---> Classification Score
    |        |
    |        +---> Reg Branch ---> Bounding Box (l, t, r, b)
    |        |
    |        +---> Obj Branch ---> Objectness (仅早期版本)
    |
    +---> 共享特征提取层 (减少冗余计算)
```

与 YOLOX 的关键区别：
- YOLOv6 **去掉了 Objectness 分支**，采用 Anchor-free 直接回归
- 共享前几层卷积，减少参数量和计算量
- 使用更少的卷积层数，降低延迟

### 1.4 SIoU Loss

YOLOv6 提出了 **SIoU（SCYLLA-IoU）Loss**，在传统 IoU Loss 基础上引入角度感知。

SIoU 由四个分量组成：

```
SIoU = 1 - IoU + (Delta + Omega) / 2

其中:
  IoU    : 交并比
  Delta  : 距离惩罚项 (考虑角度)
  Omega  : 形状惩罚项
```

**角度感知（Angle-aware）** 的核心思想：

```
          预测框
        ┌─────────┐
        │    ●----│----->  GT 框
        │   /     │       ┌─────────┐
        │  / θ    │       │         │
        │ /       │       │         │
        └─────────┘       └─────────┘

当 θ = 0 或 90 度: 沿坐标轴方向, 收敛更快
当 θ = 45 度:     对角线方向, 需要额外惩罚
```

角度惩罚项 Delta 的定义：

```
Delta = 1 - 2 * sin^2(arcsin(sin_theta) - pi/4)

其中:
  sin_theta = |y_gt - y_pred| / distance(center_gt, center_pred)

效果:
  theta -> 0 或 pi/2 时, Delta -> 0  (无额外惩罚)
  theta -> pi/4       时, Delta -> 1  (最大惩罚)
```

形状惩罚项 Omega：

```
Omega = (1 - e^(-w_w/w_gt))^theta_w + (1 - e^(-h_h/h_gt))^theta_h

其中:
  w_w = |w_pred - w_gt|, h_h = |h_pred - h_gt|
  theta_w, theta_h 为可调超参数 (通常取 4)
```

### 1.5 自研量化方案

YOLOv6 的量化部署包含两个关键技术：

| 技术 | 说明 |
|------|------|
| **RepOptimizer** | 基于重参数化的优化器，训练时显式引入稀疏正则化，使权重分布更利于量化 |
| **QAT (Quantization-Aware Training)** | 在训练后期插入伪量化节点，模拟 INT8 精度损失，微调恢复精度 |

量化流程：

```
Step 1: 正常训练 (RepOptimizer, FP32)
   |
Step 2: 重参数化合并 (RepVGG -> 单路 3x3)
   |
Step 3: QAT 微调 (插入 FakeQuant 节点)
   |
Step 4: 导出 INT8 模型 (TensorRT / ONNX Runtime)
   |
Step 5: 部署推理
```

---

## 2. YOLOv7 — 学术 SOTA 的捍卫者

### 2.1 设计哲学

YOLOv7 由 Wang 等人提出，核心目标是在 COCO 上刷新 SOTA，同时保持实时推理速度。论文标题 "Trainable bag-of-freebies" 强调了其策略：通过可训练的技巧（不增加推理成本）来提升精度。

### 2.2 E-ELAN 架构

**E-ELAN（Extended Efficient Layer Aggregation Network）** 是 YOLOv7 的核心 Neck 结构，改进自 ELAN（来自 YOLOv7 的前身论文）。

```
传统 ELAN:
  Input ---> split ---> [Conv, Conv, Conv, Conv] ---> concat ---> Output

E-ELAN:
  Input ---> split ---> [Group 1: Conv x4]
            |    |        [Group 2: Conv x4]
            |    |        [Group 3: Conv x4]
            |    +-----> shuffle (通道重排)
            +---------> concat ---> Output
```

E-ELAN 的关键创新：

1. **分组卷积（Group Conv）**：将通道分成多组并行处理，减少计算量
2. **Shuffle 机制**：跨组信息交换，避免信息孤立
3. **扩展（Extend）**：在不改变梯度路径长度的前提下，增加计算块数量

```
计算复杂度对比:

传统堆叠 Conv:    O(n * C^2 * H * W)     n = Conv 层数
E-ELAN (g 组):    O(n * (C/g)^2 * H * W * g) = O(n * C^2 * H * W / g)

当 g = 4 时, 计算量约为传统方式的 1/4
```

### 2.3 模型缩放（Compound Scaling）

YOLOv7 提出了 **复合缩放** 策略，同时调整深度和宽度：

```
缩放因子:
  depth(d) : 控制模块重复次数
  width(w) : 控制通道数

YOLOv7 系列:
  ┌──────────┬───────┬───────┬──────────┬───────────┐
  │ Model    │ depth │ width │ Params   │ FLOPs     │
  ├──────────┼───────┼───────┼──────────┼───────────┤
  │ YOLOv7-t │ 0.25  │ 0.375 │ 6.2M     │ 13.2G     │
  │ YOLOv7   │ 1.0   │ 1.0   │ 36.9M    │ 104.7G    │
  │ YOLOv7-x │ 1.0   │ 1.25  │ 71.3M    │ 189.9G    │
  │ YOLOv7-w6│ -     │ -     │ 70.4M    │ 360.0G    │
  │ YOLOv7-e6│ -     │ -     │ 97.2M    │ 515.2G    │
  │ YOLOv7-d6│ -     │ -     │ 154.7M   │ 806.8G    │
  │ YOLOv7-e6e│ -    │ -     │ 154.7M   │ 843.2G    │
  └──────────┴───────┴───────┴──────────┴───────────┘
```

缩放公式：

```
d_new = round(d_base * d_factor)   # 深度: 模块重复次数
c_new = round(c_base * w_factor)   # 宽度: 通道数

注意: 通道数必须能被特定值整除 (如 8 或 16), 以适配硬件加速
```

### 2.4 重参数化策略

YOLOv7 的重参数化与 YOLOv6 类似但有差异，体现在 **RepConv** 的设计上：

```
RepConv 训练时:
┌─────────────────────────────────┐
│  Input                          │
│    |                            │
│    +---> 3x3 Conv + BN          │
│    |                            │
│    +---> 1x1 Conv + BN          │ ---> Add ---> Output
│    |                            │
│    +---> Identity (BN)          │
│                                  │
│  (当 stride=1 时存在 Identity)   │
└─────────────────────────────────┘

RepConv 推理时:
┌─────────────────────────────────┐
│  Input ---> 单个 3x3 Conv ---> Output │
└─────────────────────────────────┘
```

YOLOv7 中 **并非所有层都使用 RepConv**。论文发现：
- 在某些位置使用普通 Conv 效果更好
- RepConv 中的 Identity 分支在推理时被去掉后，对某些层有负面影响
- 因此提出了 **"Planned re-parameterized convolution"**，有选择地使用重参数化

### 2.5 Auxiliary Head 与 Lead Head

这是 YOLOv7 最具创新性的训练策略之一：

```
                    Lead Head (主检测头)
                         |
                    ┌────┴────┐
                    │  输出层  │  <-- 最终推理使用
                    └────┬────┘
                         |
                    ┌────┴────┐
                    │  Neck   │
                    └────┬────┘
                         |
              ┌──────────┼──────────┐
              |          |          |
         [主干特征]  [中间特征]  [浅层特征]
                         |
                         |
                    ┌────┴────┐
                    │Aux Head │  <-- 仅训练使用, 推理时去掉
                    │(辅助头) │
                    └─────────┘

训练时: Lead Head + Auxiliary Head 同时监督
推理时: 仅保留 Lead Head
```

**Lead Head** 负责最终输出，**Auxiliary Head** 位于较浅层，作用是：

1. **梯度辅助**：为较深的网络层提供额外的梯度信号，缓解梯度消失
2. **Soft Label 生成**：Lead Head 的输出作为 Auxiliary Head 的软标签

软标签的数学表达：

```
Y_aux = alpha * Y_gt + (1 - alpha) * Y_lead_pred

其中:
  Y_gt       : Ground Truth 标签 (one-hot)
  Y_lead_pred: Lead Head 的预测输出 (概率分布)
  alpha      : 混合系数 (通常 0.05 ~ 0.2)

效果:
  - 为 Auxiliary Head 提供更丰富的监督信号
  - 类似于 Knowledge Distillation 的思想
  - Lead Head 本身也从这种协同训练中受益
```

### 2.6 COCO 上的成果

YOLOv7 在发布时（2022年7月）达到了 COCO 上的 SOTA：

```
精度-速度权衡 (V100 GPU, COCO val2017):

  AP (%)
  58 ┤                              * YOLOv7-e6e (56.8% AP)
     │                           *
  56 ┤                        *
     │                     *
  54 ┤                  * YOLOv7-x (53.1% AP)
     │               *
  52 ┤            *
     │         * YOLOv7 (51.4% AP)
  50 ┤      *
     │   *
  48 ┤* YOLOv7-t (38.7% AP)
     │
  46 ┼────┬────┬────┬────┬────┬────┬──> FPS
     50  100  150  200  250  300  350

关键数据点:
  YOLOv7:     51.4% AP, 161 FPS (V100)
  YOLOv7-x:   53.1% AP, 114 FPS
  YOLOv7-e6e: 56.8% AP,  36 FPS
```

---

## 3. PP-YOLOE — 百度飞桨生态

### 3.1 概述

PP-YOLOE 是百度 PaddleDetection 团队推出的系列模型，定位为 **飞桨生态下的高性能检测器**。它不追求 SOTA 数值，而是注重工程实用性和国产硬件适配。

### 3.2 CSPRepResNet Backbone

PP-YOLOE 的骨干网络融合了 CSPNet 和 RepResNet：

```
CSPRepResNet Block:
┌──────────────────────────────────────┐
│  Input                               │
│    |                                 │
│    +---> Branch 1: [1x1 Conv]        │
│    |                                 │
│    +---> Branch 2: [1x1 Conv]        │
│              |                       │
│              +---> RepResBlock x N    │
│              |      (内含重参数化)     │
│              |                       │
│              +---> Transition Layer   │
│    |                                 │
│    +---> Concat(Branch1, Branch2)    │
│    |                                 │
│    +---> 1x1 Conv (融合)             │
│    |                                 │
│    v                                 │
│  Output                              │
└──────────────────────────────────────┘
```

RepResBlock 在训练时使用多分支（类似 RepVGG），推理时合并为单路，兼顾训练性能和推理效率。

### 3.3 ETAHead（Efficient Task Alignment Head）

ETAHead 是 PP-YOLOE 的检测头，核心思想是 **任务对齐（Task Alignment）**：

```
传统检测头的问题:
  - 分类任务偏好: 特征图上语义响应最强的位置
  - 定位任务偏好: 与 GT 边界最接近的位置
  - 两者往往不一致！

ETAHead 的解决方案:
┌──────────────────────────────────┐
│  Feature Map                     │
│    |                             │
│    +---> Cls Score ──┐           │
│    |                  |           │
│    +---> IoU Score ──┼--> TAP ──> 最终得分
│                       |           │
│    TAP = cls_score^alpha          │
│          * iou_score^beta         │
│                                  │
│    alpha, beta: 可调超参数        │
└──────────────────────────────────┘
```

**TAP (Task Alignment Product)** 的定义：

```
TAP(s_cls, s_iou) = (s_cls)^alpha * (s_iou)^beta

其中:
  s_cls : 分类得分
  s_iou : IoU 预测得分 (定位质量)
  alpha : 分类权重 (通常 1.0)
  beta  : 定位权重 (通常 1.0)

作用:
  训练时: TAP 用于选择正样本 (高分类+高定位)
  推理时: TAP 作为最终置信度排序依据
```

### 3.4 TAL（Task Alignment Learning）

TAL 是配套 ETAHead 的训练策略：

```
传统标签分配:            TAL 标签分配:
  基于 IoU 阈值            基于 TAP 得分
  静态阈值 (如 0.5)        动态自适应

  ┌─────────┐              ┌─────────┐
  │ IoU>0.5? │              │ TAP 排名 │
  │  Y -> 正 │              │ Top-K -> 正│
  │  N -> 负 │              │ 其余 -> 负│
  └─────────┘              └─────────┘

优势:
  - 分类好的框不一定是定位好的框
  - TAL 自动选择"两项都好"的框作为正样本
  - 避免了 IoU 阈值的人工调节
```

TAL 的损失函数：

```
L_total = lambda_cls * L_cls + lambda_iou * L_iou + lambda_dfl * L_dfl

其中:
  L_cls : Varifocal Loss (分类损失)
  L_iou : IoU Loss (定位损失)
  L_dfl : Distribution Focal Loss (分布回归损失)

  lambda_cls, lambda_iou, lambda_dfl : 权重系数
```

---

## 4. 横向对比

### 4.1 核心架构对比

| 维度 | YOLOv6 | YOLOv7 | PP-YOLOE |
|------|--------|--------|----------|
| **提出机构** | 美团 | Wang 等人 | 百度 |
| **骨干网络** | EfficientRep (RepVGG 风格) | E-ELAN + CSP | CSPRepResNet |
| **检测头** | Efficient Decoupled Head | Decoupled Head | ETAHead |
| **标签分配** | TAL (SimOTA 改进) | Auxiliary Head + Soft Label | TAL (自研) |
| **回归损失** | SIoU Loss | CIoU Loss | IoU Loss + DFL |
| **重参数化** | 全骨干 RepVGG Block | 选择性 RepConv | RepResBlock |
| **量化支持** | RepOptimizer + QAT (自研) | 社区方案 | PaddleSlim (飞桨) |
| **部署框架** | TensorRT / ONNX | TensorRT / ONNX / CoreML | Paddle Inference |

### 4.2 精度与速度对比

| 模型 | 输入尺寸 | AP (COCO) | 参数量 | FLOPs | 推理速度 |
|------|----------|-----------|--------|-------|----------|
| YOLOv6-S | 640 | 43.1% | 17.2M | 24.5G | 442 FPS (TensorRT) |
| YOLOv6-M | 640 | 49.5% | 34.3M | 60.0G | 234 FPS |
| YOLOv6-L | 640 | 51.7% | 58.5M | 116.8G | 137 FPS |
| YOLOv7-t | 640 | 38.7% | 6.2M | 13.2G | 350+ FPS |
| YOLOv7 | 640 | 51.4% | 36.9M | 104.7G | 161 FPS (V100) |
| YOLOv7-x | 640 | 53.1% | 71.3M | 189.9G | 114 FPS |
| YOLOv7-e6e | 1280 | 56.8% | 154.7M | 843.2G | 36 FPS |
| PP-YOLOE-S | 640 | 43.1% | 7.9M | 17.4G | 333 FPS (V100) |
| PP-YOLOE-M | 640 | 49.0% | 23.4M | 49.0G | 208 FPS |
| PP-YOLOE-L | 640 | 51.4% | 52.2M | 110.1G | 149 FPS |
| PP-YOLOE-X | 640 | 52.3% | 98.4M | 206.6G | 96 FPS |

### 4.3 技术路线对比

```
                YOLOv6                YOLOv7              PP-YOLOE
             ┌──────────┐         ┌──────────┐         ┌──────────┐
  Backbone   │ RepVGG   │         │ E-ELAN   │         │ CSPRep   │
             │ 重参数化  │         │ 分组扩展  │         │ ResNet   │
             └────┬─────┘         └────┬─────┘         └────┬─────┘
                  │                    │                    │
             ┌────┴─────┐         ┌────┴─────┐         ┌────┴─────┐
  Neck       │ PAN +    │         │ E-ELAN   │         │ PAN +    │
             │ RepBlock │         │ + Shuffle│         │ CSPRep   │
             └────┬─────┘         └────┬─────┘         └────┬─────┘
                  │                    │                    │
             ┌────┴─────┐         ┌────┴─────┐         ┌────┴─────┐
  Head       │ Efficient│         │ Decoupled│         │ ETAHead  │
             │ Decoupled│         │ + Aux    │         │ + TAL    │
             └────┬─────┘         └────┬─────┘         └────┬─────┘
                  │                    │                    │
             ┌────┴─────┐         ┌────┴─────┐         ┌────┴─────┐
  Loss       │ SIoU     │         │ CIoU +   │         │ VFL +    │
             │          │         │ Soft Label│        │ IoU + DFL│
             └──────────┘         └──────────┘         └──────────┘

  关键差异:
  YOLOv6 → 重参数化为核心, 量化部署为优先
  YOLOv7 → 多分支聚合 + 训练技巧为优先
  PP-YOLOE → 任务对齐为核心, 飞桨生态为依托
```

### 4.4 适用场景推荐

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| GPU 服务器部署 (高精度) | YOLOv7-e6e | AP 最高 (56.8%)，适合精度优先场景 |
| GPU 边缘部署 (平衡) | YOLOv7 / YOLOv6-L | 精度与速度兼顾 |
| 移动端 / NPU 部署 | YOLOv6-S | 自研量化方案，硬件适配好 |
| 国产芯片 / 飞桨生态 | PP-YOLOE-L | PaddleSlim 量化，国产硬件支持完善 |
| CPU 部署 | PP-YOLOE-S | 参数少、FLOPs 低，CPU 友好 |

---

## 5. 延伸阅读

- [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976) — 美团技术团队，2022
- [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696) — Wang et al., 2022
- [PP-YOLOE: An Evolved Version of YOLO](https://arxiv.org/abs/2203.16250) — 百度 PaddleDetection, 2022
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697) — 重参数化思想的奠基工作
- [SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740) — SIoU 损失原始论文
- [Assigning Orders for Dense Object Detection](https://arxiv.org/abs/2108.07707) — 任务对齐学习的理论基础

---

## 6. 思考题

<details>
<summary>1. YOLOv6 的 RepVGG 合并后为什么推理速度更快？</summary>

训练时三个分支（3x3 Conv、1x1 Conv、Identity）需要分别计算并相加，推理时合并为单个 3x3 Conv 后：
- 减少了内存访问次数（一次而非三次）
- 消除了中间结果的加法操作
- 更利于硬件流水线优化（单路连续计算无依赖）
- 对 GPU/CPU 缓存更友好，减少了 cache miss
</details>

<details>
<summary>2. YOLOv7 的 Auxiliary Head 为什么不增加推理成本？</summary>

Auxiliary Head 仅在训练阶段存在，用于：
- 提供额外的监督梯度信号到浅层
- 生成 Soft Label 辅助 Lead Head 训练

推理时直接移除 Auxiliary Head，网络结构等价于没有辅助头的模型。这是一种典型的"训练时增加复杂度，推理时保持简洁"的策略，属于 "bag-of-freebies"。
</details>

<details>
<summary>3. SIoU 的角度感知机制如何帮助小目标检测？</summary>

小目标的预测框与 GT 框的中心距离相对较大，微小偏移就会导致角度偏差。SIoU 的角度惩罚机制使得：
- 当预测框接近 GT 但方向不对时，额外的 Delta 惩罚促使模型更快地沿正确方向收敛
- 小目标的边界框回归对角度更敏感，角度感知的损失设计能更有效地引导梯度方向
- 相比 CIoU 只考虑中心距离和宽高比，SIoU 的方向信息对小目标更有价值
</details>

<details>
<summary>4. PP-YOLOE 的 TAL 与 YOLOX 的 SimOTA 有什么异同？</summary>

相同点：
- 都是动态标签分配策略，不依赖固定的 IoU 阈值
- 都综合考虑分类和定位质量来选择正样本

不同点：
- SimOTA 使用 Sinkhorn-Knopp 算法求解最优传输问题，计算开销较大
- TAL 使用 TAP（分类得分 x IoU 得分）直接排序，计算更简单高效
- TAL 显式引入了"任务对齐"的概念，而 SimOTA 更侧重于匹配优化
</details>

<details>
<summary>5. 为什么 YOLOv7 没有在所有层都使用 RepConv？</summary>

Wang 等人在实验中发现：
- RepConv 的 Identity 分支在推理时被移除后，对网络深层的特征表达有负面影响
- 在网络的某些关键位置（如第一层或跳跃连接处），保持原始的多分支结构更有利于特征传播
- 因此提出了 "Planned re-parameterized convolution"，根据层的位置和功能选择性地应用重参数化
- 这体现了"并非所有技巧都适用于所有位置"的工程思想
</details>
