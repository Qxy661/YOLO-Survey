# YOLO 系列损失函数论文导读

> 预计阅读时间：~10 分钟 | 前置知识：基本损失函数概念（MSE、交叉熵）、IoU 计算方式、目标检测流程

---

## 目录

1. [1. Focal Loss — 解决正负样本极度不均衡](#1-focal-loss-解决正负样本极度不均衡)
2. [2. GIoU Loss — 引入最小外接矩形的 IoU 改进](#2-giou-loss-引入最小外接矩形的-iou-改进)
3. [3. DIoU / CIoU Loss — 距离与纵横比的双重约束](#3-diou-ciou-loss-距离与纵横比的双重约束)
4. [4. SIoU Loss — 引入角度感知的损失函数](#4-siou-loss-引入角度感知的损失函数)
5. [5. WIoU Loss — 基于动态非单调聚焦机制的 IoU](#5-wiou-loss-基于动态非单调聚焦机制的-iou)
6. [6. DFL（Distribution Focal Loss）— 分布式边界框回归](#6-dfldistribution-focal-loss-分布式边界框回归)
7. [7. VariFocal Loss — 统一分类与定位质量](#7-varifocal-loss-统一分类与定位质量)
8. [8. Quality Focal Loss（QFL）— 分类质量联合监督](#8-quality-focal-lossqfl-分类质量联合监督)
9. [9. Inner-IoU — 基于辅助框的 IoU 变体](#9-inner-iou-基于辅助框的-iou-变体)

---


## 1. Focal Loss — 解决正负样本极度不均衡

| 字段 | 内容 |
|------|------|
| 作者 | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar |
| 年份 | 2017 |
| 会议 | ICCV 2017 |
| 关键词 | 类别不均衡、难易样本、RetinaNet |

**一句话总结：** 在交叉熵基础上加入调制因子 `(1-p_t)^gamma`，降低易分类样本权重，让模型聚焦于难分类样本。

**核心贡献：** 提出 Focal Loss 及平衡因子 `alpha`；构建 RetinaNet，首次使单阶段检测器精度超越两阶段方法。

**方法亮点：** gamma=0 退化为标准交叉熵；gamma=2.0, alpha=0.25 为最优超参组合；实现仅需修改一行代码。

**关键结果：** RetinaNet 在 COCO test-dev 达到 39.1 AP（ResNet-101-FPN），证明单阶段检测器的精度瓶颈在 loss 设计而非架构。

---

## 2. GIoU Loss — 引入最小外接矩形的 IoU 改进

| 字段 | 内容 |
|------|------|
| 作者 | Hamid Rezatofighi, Nathan Tsoi, JunYoung Goyal 等 |
| 年份 | 2019 |
| 会议 | CVPR 2019 |
| 关键词 | IoU 变体、边界框回归、最小外接矩形 |

**一句话总结：** 在 IoU 基础上引入最小外接矩形（smallest enclosing box），解决非重叠框梯度为零的问题。

**核心贡献：** 提出 GIoU = IoU - (C-U)/C，其中 C 为最小外接矩形面积，U 为并集面积；保持尺度不变性。

**方法亮点：** 值域 [-1, 1]，两框重合时 GIoU=1，相距无穷远时 GIoU=-1；loss 最小化 `1-GIoU`，值域 [0, 2]。

**关键结果：** 在 Faster R-CNN 和 SSD 上替代 L1/L2 loss，AP 提升 1-2 点；非重叠框场景下收敛速度显著优于 smooth L1。

---

## 3. DIoU / CIoU Loss — 距离与纵横比的双重约束

| 字段 | 内容 |
|------|------|
| 作者 | Zhaohui Zheng, Ping Wang, Wei Liu 等 |
| 年份 | 2020 |
| 会议 | AAAI 2020 |
| 关键词 | 中心点距离、纵横比、DIoU-NMS |

**一句话总结：** DIoU 直接最小化中心点距离；CIoU 进一步考虑纵横比一致性，实现更快更准的边界框回归。

**核心贡献：** DIoU = IoU - d^2/c^2（d 中心距离，c 对角线），解决 GIoU 在包含关系下退化的问题；CIoU = DIoU - alpha*v 加入纵横比惩罚；提出 DIoU-NMS。

**方法亮点：** CIoU loss 对 v 的梯度直接作用于 w 和 h，避免间接优化；DIoU-NMS 保留中心距离远的框减少误删。

**关键结果：** CIoU 在 YOLOv3 上提升 2-3 AP，收敛速度比 GIoU 快 50% 以上；DIoU-NMS 各种检测器上提升 0.5-1.0 AP。

---

## 4. SIoU Loss — 引入角度感知的损失函数

| 字段 | 内容 |
|------|------|
| 作者 | Siliang Gevorgyan |
| 年份 | 2022 |
| 关键词 | 角度惩罚、方向感知、SCYLLA-IoU |

**一句话总结：** 在 IoU 基础上引入角度惩罚，使模型感知预测框与目标框之间的方向关系，优化回归路径。

**核心贡献：** 提出 SIoU，首次在 IoU 系列损失中引入角度因素；设计四部分损失：Angle Cost、Distance Cost、Shape Cost、IoU Cost。

**方法亮点：** Angle Cost 通过 sin 函数惩罚非最优方向，引导预测框沿最短路径移动；Shape Cost 同时优化宽高和纵横比。

**关键结果：** 在 YOLOv5/v7 上替代 CIoU，AP 提升 1-2 点；密集场景和小目标场景下提升更明显。

---

## 5. WIoU Loss — 基于动态非单调聚焦机制的 IoU

| 字段 | 内容 |
|------|------|
| 作者 | Zijia Tong, Jungang Xu 等 |
| 年份 | 2023 |
| 关键词 | 动态权重、非单调聚焦、梯度增益 |

**一句话总结：** 采用动态非单调聚焦机制，根据 IoU 值自适应分配梯度增益，减少低质量样本的有害梯度。

**核心贡献：** 提出 WIoU（Wise-IoU），首次在 IoU 损失中引入动态非单调权重分配；将样本分为高质量/中等/低质量三类分别策略。

**方法亮点：** Sigmoid 函数构造梯度增益曲线，低 IoU 时抑制梯度；中等 IoU 权重最大（主要学习对象）；非单调设计避免极端行为。

**关键结果：** WIoU-v2/v3 在 YOLOv5/v7/v8 上均有稳定提升；对低质量标注和噪声标签更鲁棒；小目标密集场景优势显著。

---

## 6. DFL（Distribution Focal Loss）— 分布式边界框回归

| 字段 | 内容 |
|------|------|
| 作者 | Xiang Li, Wenhai Wang 等（mmdetection 团队） |
| 年份 | 2020 |
| 会议 | ICLR 2021 |
| 关键词 | 分布式回归、FCOS、积分表示 |

**一句话总结：** 将边界框回归从单一值预测改为离散分布预测，通过 Focal Loss 风格的交叉熵监督分布学习。

**核心贡献：** 边界框坐标建模为离散概率分布；设计 DFL 对分布施加类似 Focal Loss 的加权；积分表示将离散分布转为连续坐标。

**方法亮点：** [0, reg_max] 离散化为 reg_max+1 个值，softmax 预测概率；DFL 对接近真值的相邻位置施加更大权重；最终坐标 = sum(prob_i * value_i)。

**关键结果：** 在 FCOS/ATSS 上 AP 提升 1.0-1.5 点；大目标回归精度提升显著；已被 YOLOv5/v7/v8/RT-DETR 广泛采用。

---

## 7. VariFocal Loss — 统一分类与定位质量

| 字段 | 内容 |
|------|------|
| 作者 | Haoyang Zhang, Ying Wang, Feras Dayoub, Niko Sunderhauf |
| 年份 | 2021 |
| 会议 | ICCV 2021 |
| 关键词 | IoU 感知分类、质量估计、不对称加权 |

**一句话总结：** 将分类分支的监督信号从 binary label 改为 IoU-aware 的连续值，实现分类与定位质量的统一建模。

**核心贡献：** 提出 VFL，对正样本使用 IoU 值作为分类目标（而非 1）；对正负样本采用不对称加权策略。

**方法亮点：** VFL 对正样本 target=IoU（连续值），负样本仍用 0 但降低权重；alpha 和 gamma 分别控制权重和聚焦程度；与 Focal Loss 的区别在于不对称性。

**关键结果：** 在 FCOS/RetinaNet 上 AP 提升 1-2 点；密集预测场景效果显著；已被 YOLOv8 采用作为默认分类损失。

---

## 8. Quality Focal Loss（QFL）— 分类质量联合监督

| 字段 | 内容 |
|------|------|
| 作者 | Xiang Li, Wenhai Wang 等（FCOS 团队） |
| 年份 | 2020 |
| 会议 | CVPR 2021（GFL 论文中） |
| 关键词 | 质量估计、IoU 分支、Generalized Focal Loss |

**一句话总结：** 在 Focal Loss 基础上引入 IoU 作为分类目标的调制因子，使分类分支同时输出置信度和定位质量。

**核心贡献：** 提出 QFL，将 IoU 质量分数融入分类损失；与 DFL 共同构成 GFL 框架；推理时分类分数直接作为 NMS 排序依据。

**方法亮点：** QFL = -|y-sigma|^beta * ((1-y)*log(1-sigma) + y*log(sigma))，y 为 IoU 质量标签，sigma 为预测分数，beta 为聚焦参数。

**关键结果：** 在 FCOS/ATSS 上 AP 提升 1.0-1.5 点；与 DFL 组合效果最佳，GFL 整体提升约 2-3 AP。

---

## 9. Inner-IoU — 基于辅助框的 IoU 变体

| 字段 | 内容 |
|------|------|
| 作者 | Jiabo Zhang, Qi Chu 等 |
| 年份 | 2023 |
| 关键词 | 辅助内框、比率因子、多尺度回归 |

**一句话总结：** 引入与原始框成比例的辅助 inner 框计算 IoU，通过比率因子控制不同尺度下 IoU 的计算范围。

**核心贡献：** 提出 Inner-IoU，构造按比例缩小的 inner 框来计算 IoU；引入 ratio 因子控制缩放比例，适配不同尺度需求。

**方法亮点：** inner 框中心不变、宽高按 ratio 缩放；ratio=1.0 退化为标准 IoU；可与 GIoU/DIoU/CIoU/SIoU 任意组合。

**关键结果：** Inner-CIoU（ratio=0.7）在 YOLOv5/v7/v8 上比标准 CIoU 提升 0.5-1.5 AP；小目标提升约 1-2 AP；计算开销可忽略。

---

## 延伸阅读

- **Generalized Focal Loss (GFL)**：QFL + DFL 的统一框架，Li et al., CVPR 2021
- **Wise-IoU 系列**：WIoU v1/v2/v3 的演进，关注动态非单调加权策略
- **Efficient IoU 系列**：EIoU、SIoU 等对 CIoU 的改进方向
- **Powerful-IoU (PIoU)**：面向旋转目标检测的 IoU 变体
- **3D IoU 变体**：IoU3D、GIoU3D 等在点云/3D 检测中的应用
- **Loss 函数设计综述**：A Survey of Loss Functions in Computer Vision

---

<details>
<summary><b>思考题</b></summary>

1. Focal Loss 中 gamma 参数对易/难样本的调节机制是什么？为什么 gamma=2 是常用选择？
2. GIoU 在什么场景下会退化为接近普通 IoU 的行为？DIoU 是如何解决这个问题的？
3. CIoU 中的纵横比惩罚项 v 对 w 和 h 的梯度是如何推导的？这种设计有什么优势？
4. SIoU 引入角度惩罚后，对回归路径的优化有什么具体影响？在什么场景下收益最大？
5. WIoU 的非单调权重分配与 Focal Loss 的单调权重分配相比，各自适用什么类型的训练数据？
6. DFL 将边界框回归建模为离散分布而非单点值，这种设计的优势和劣势分别是什么？
7. VariFocal Loss 和 Quality Focal Loss 都涉及 IoU-aware 分类，它们的核心区别是什么？各自适合什么检测架构？
8. Inner-IoU 中 ratio 参数的选择对不同尺度目标的回归精度有何影响？如何确定最优 ratio？
9. 综合比较：如果你要为一个新的密集小目标检测任务选择损失函数组合，你会如何搭配？为什么？
10. 从 Focal Loss (2017) 到 WIoU (2023)，IoU 系列损失函数的演进体现了哪些设计趋势？

</details>
