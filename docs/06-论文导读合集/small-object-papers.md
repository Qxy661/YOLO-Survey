# 小目标检测经典论文导读

> **阅读时长**: ~15 min | **前置知识**: small object detection basics, multi-scale feature fusion, anchor-based/anchor-free detectors

小目标检测 (Small Object Detection) 是目标检测领域最具挑战性的任务之一。COCO 将面积小于 32x32 像素的目标定义为小目标，这类目标在特征提取过程中信息极易丢失。本文梳理了 12 篇针对小目标检测的核心论文，从多尺度特征融合、尺度归一化、切片推理、聚类检测到查询驱动等多条技术路线展开解读。

---

## 目录

1. [1. Feature Pyramid Networks (FPN)](#1-feature-pyramid-networks-fpn)
2. [2. SNIP / SNIPER](#2-snip-sniper)
3. [3. SAHI (Slicing Aided Hyper Inference)](#3-sahi-slicing-aided-hyper-inference)
4. [4. ClusDet](#4-clusdet)
5. [5. DMNet](#5-dmnet)
6. [6. GLSAN](#6-glsan)
7. [7. QueryDet](#7-querydet)
8. [8. FocusNet](#8-focusnet)
9. [9. AugFPN](#9-augfpn)
10. [10. DetectoRS](#10-detectors)
11. [11. EfficientDet](#11-efficientdet)
12. [12. FCOS](#12-fcos)

---


## 1. Feature Pyramid Networks (FPN)

| 字段 | 内容 |
|------|------|
| **论文** | Feature Pyramid Networks for Object Detection |
| **作者** | Tsung-Yi Lin, Piotr Dollar, Ross Girshick et al. |
| **会议** | CVPR 2017 |
| **引用** | ~25000+ |
| **论文链接** | https://arxiv.org/abs/1612.03144 |

**一句话总结**: 提出自顶向下 + 横向连接的特征金字塔结构，使检测器在多尺度上均获得语义丰富的特征。

**核心贡献**:
- 提出 FPN 架构：自顶向下路径 (top-down pathway) 与横向连接 (lateral connection) 结合，将高层语义信息逐层传递到低层高分辨率特征
- 首次证明**单张图片前向传播**即可构建高质量多尺度特征表示，抛弃了传统图像金字塔的多尺度输入方案
- 确立了"每层特征图独立预测对应尺度目标"的范式 (per-layer prediction)

**方法亮点**:
- 横向连接将 backbone 各阶段输出 (C2-C5) 与上采样后的高层特征逐元素相加，生成 P2-P5
- P6 由 P5 经 3x3 stride-2 卷积获得，用于 RetinaNet 等大目标检测
- 结构极其简洁，几乎不增加计算量，却显著提升小目标 AP_S

**关键结果**:
- 在 COCO 上以 ResNet-101 backbone 获得 36.7 AP，其中 AP_S 提升超过 8 个点
- 成为后续几乎所有检测器的标准组件，影响深远

---

## 2. SNIP / SNIPER

| 字段 | 内容 |
|------|------|
| **论文** | An Analysis of Scale Invariance in Object Detection -- SNIP; SNIPER: Efficient Multi-Scale Training |
| **作者** | Bharat Singh, Larry S. Davis |
| **会议** | CVPR 2018 / ECCV 2018 |
| **引用** | ~1500+ / ~800+ |
| **论文链接** | https://arxiv.org/abs/1711.08189 / https://arxiv.org/abs/1805.09300 |

**一句话总结**: SNIP 在多尺度图像上只对"尺寸合适"的目标进行训练；SNIPER 将其改进为对图像区域的高效裁剪方案。

**核心贡献**:
- 提出 **Scale Normalization for Image Pyramids (SNIP)**: 在多尺度图像金字塔上训练时，只对与当前尺度匹配的目标计算梯度，忽略过大/过小目标
- SNIPER (Scale Normalization for Image Pyramids with Efficient Resampling): 将 SNIP 的图像金字塔替换为对 **chip** (裁剪区域) 的选择性训练，大幅降低计算量
- 揭示了预训练模型分辨率与检测尺度之间的关系

**方法亮点**:
- SNIP 定义了有效尺度范围 (e.g., 12-256 pixels for 800-pixel 输入)，每个尺度分支只处理落入该范围的目标
- SNIPER 在正样本 chip 和负样本 chip 上采样，保证背景多样性的同时避免全图计算
- 使用 context padding 扩展 chip 区域以保留上下文信息

**关键结果**:
- SNIP 在 COCO test-dev 上以单模型获得 45.7 AP (2018 年 SOTA)
- SNIPER 在保持 SNIP 精度的同时训练速度提升约 3 倍，推理速度提升约 5 倍

---

## 3. SAHI (Slicing Aided Hyper Inference)

| 字段 | 内容 |
|------|------|
| **论文** | Slicing Aided Hyper Inference for Small Object Detection |
| **作者** | Fatih Cagatay Akyon, Sinan Onur Altinuc et al. |
| **年份** | 2022 (WACV 2022 Workshop) |
| **引用** | ~800+ |
| **论文链接** | https://arxiv.org/abs/2202.06934 |

**一句话总结**: 提出切片辅助推理框架，将大图切成重叠小图分别检测后合并，无需重新训练即可显著提升小目标检测性能。

**核心贡献**:
- 提出 **Slicing Aided Hyper Inference (SAHI)** 框架：推理时将图像切分为有重叠 (overlap) 的小图 (slice)，分别运行检测器，再用 NMS 合并结果
- 提供切片辅助训练 (Slicing Aided Hyper Fine-tuning, SAHFT) 策略，用切片裁剪区域微调模型
- **即插即用**：适用于任意检测器 (YOLO, DETR, Faster R-CNN 等)，无需修改模型结构

**方法亮点**:
- slice_size 和 overlap_ratio 是两个核心超参数，通常 slice_size=640, overlap_ratio=0.2
- 标准 NMS 无法正确处理跨 slice 边界的目标，需使用 **postprocess_type='NMS'** 或 **'SNMS'** (soft-NMS 变体)
- 提供 Python 包 `sahi`，pip install 即可使用，与主流检测框架无缝集成

**关键结果**:
- 在 VisDrone 数据集上，使用 YOLOv5 + SAHI 推理，AP_S 提升 15-20 个点
- 在自定义遥感数据集上平均提升 10+ AP，零训练成本
- 开源项目 GitHub Star 4000+，工业界广泛采用

---

## 4. ClusDet

| 字段 | 内容 |
|------|------|
| **论文** | Clustered Object Detection in Aerial Images |
| **作者** | Fan Yang, Heng Fan, Peng Chu et al. |
| **会议** | ICCV 2019 |
| **引用** | ~500+ |
| **论文链接** | https://arxiv.org/abs/1901.01866 |

**一句话总结**: 针对航拍图像中目标密集分布的特点，先聚类定位目标密集区域，再对这些区域高分辨率裁剪后精细检测。

**核心贡献**:
- 提出 **Clustered Detection (ClusDet)** 框架：先用全局检测器定位目标密度高的 cluster 区域，再对这些区域裁剪放大后用精细检测器二次检测
- 设计 **Density Map-based Cluster Estimation (DMCE)** 模块，利用密度图估计聚类区域
- 引入 **Cluster Region Detection Network (Cluster-Net)**，对裁剪后的高分辨率区域执行精细检测

**方法亮点**:
- 全局检测器负责稀疏分布的大目标，聚类检测器负责密集分布的小目标，两者互补
- Cluster 区域大小由密度图自适应确定，避免固定大小裁剪的局限
- 最终结果由全局检测和聚类检测的 NMS 合并输出

**关键结果**:
- 在 VisDrone 数据集上 AP 提升约 6-8 个点
- 在 UAVDT 数据集上 AP_S 提升约 10 个点
- 有效解决了航拍图像中小目标密集分布的难题

---

## 5. DMNet

| 字段 | 内容 |
|------|------|
| **论文** | Dynamic Multi-scale Convolution for Object Detection |
| **作者** | Xiang Li, Wenhai Wang et al. |
| **会议** | AAAI 2020 |
| **引用** | ~200+ |
| **论文链接** | https://ojs.aaai.org/index.php/AAAI/article/view/6003 |

**一句话总结**: 设计动态多尺度卷积模块，根据输入特征自适应选择不同感受野的卷积核，增强多尺度表达能力。

**核心贡献**:
- 提出 **Dynamic Multi-scale Convolution (DMConv)**：在同一层使用多个不同扩张率 (dilation rate) 的卷积核，并通过注意力机制动态加权融合
- 引入 **Scale-aware Attention**，让网络根据目标尺寸自动调整不同感受野分支的权重
- 替代传统 FPN 的单一卷积，增强单层特征的多尺度表达

**方法亮点**:
- 多个并行的 dilated convolution (dilation=1,2,3,5) 生成不同尺度的特征
- 全局平均池化 + FC 层生成各分支权重 (channel-wise attention)
- 可直接嵌入 FPN 的每一层，结构改动小

**关键结果**:
- 在 COCO test-dev 上 ResNet-101 backbone 达到 41.7 AP
- AP_S 较 baseline FPN 提升约 2-3 个点
- 计算开销增加约 5%，性价比较高

---

## 6. GLSAN

| 字段 | 内容 |
|------|------|
| **论文** | Global-Local Self-Adaptive Network for Object Detection in Aerial Images |
| **作者** | Yuxuan Bai, Yuan Yuan et al. |
| **会议** | IEEE TGRS 2020 |
| **引用** | ~200+ |
| **论文链接** | https://ieeexplore.ieee.org/document/9206571 |

**一句话总结**: 提出全局-局部自适应网络，通过全局上下文与局部细节的自适应融合来增强航拍小目标检测。

**核心贡献**:
- 设计 **Global-Local Self-Adaptive (GLSAN)** 框架：全局分支捕获场景级上下文，局部分支聚焦目标级细节，两者自适应融合
- 提出 **Self-Adaptive Feature Fusion (SAFF)** 模块，通过可学习的注意力权重自动平衡全局与局部特征的贡献
- 针对航拍图像目标尺度变化剧烈的特点，引入多尺度区域提议策略

**方法亮点**:
- 全局分支使用标准 FPN 提取多尺度特征
- 局部分支对 feature map 进行网格划分，在每个网格内独立提取特征
- SAFF 通过 sigmoid 门控机制逐通道融合全局与局部特征

**关键结果**:
- 在 DOTA 数据集上 mAP 达到 79.4% (2020 年 SOTA 之一)
- 在 VisDrone 数据集上 AP_S 较 baseline 提升约 5 个点
- 对密集小目标场景效果显著

---

## 7. QueryDet

| 字段 | 内容 |
|------|------|
| **论文** | QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection |
| **作者** | Chenhongyi Yang, Zehao Huang, Naiyan Wang |
| **会议** | CVPR 2022 |
| **引用** | ~100+ |
| **论文链接** | https://arxiv.org/abs/2103.09124 |

**一句话总结**: 利用级联稀疏查询策略，在高分辨率特征图上仅对可能存在小目标的位置计算，大幅提升小目标检测效率。

**核心贡献**:
- 提出 **Cascaded Sparse Query** 方法：在低分辨率特征图上先粗检测，将检测到的目标位置作为"查询"传递到高分辨率特征图
- 只在查询位置附近的高分辨率特征上计算检测头，避免全图高分辨率推理的巨大开销
- 实现了"高分辨率检测小目标"与"计算效率"的平衡

**方法亮点**:
- 低层 (P2, 高分辨率) 特征图计算量最大，QueryDet 只在粗检测提示的区域进行密集计算
- 使用 **Query Generation** 模块在 P3 生成初始查询，传播到 P2
- 采用稀疏卷积 (Sparse Convolution) 实现高效的局部特征提取

**关键结果**:
- 在 COCO 上 AP_S 达到 31.8%，与标准 RetinaNet 精度相当
- 推理速度提升约 2x (ResNet-50 backbone, 800-pixel 输入)
- 在高分辨率输入 (1280) 时加速比更显著，可达 3x+

---

## 8. FocusNet

| 字段 | 内容 |
|------|------|
| **论文** | FocusNet: Imbalanced Multi-class Object Detection based on Focus Loss |
| **作者** | Jingchao Liu et al. |
| **会议** | IEEE Access 2020 |
| **引用** | ~100+ |
| **论文链接** | https://ieeexplore.ieee.org/document/9138594 |

**一句话总结**: 通过改进损失函数中的聚焦机制，使模型在训练过程中更关注难以检测的小目标。

**核心贡献**:
- 提出 **Focus Loss**：在 Focal Loss 基础上引入尺度感知因子，对不同尺度的目标赋予不同的聚焦强度
- 设计 **Scale-Aware Feature Enhancement (Safe Module)**：在 FPN 各层加入尺度自适应的特征增强模块
- 解决了小目标在训练中梯度被大目标淹没的问题

**方法亮点**:
- Focus Loss 将 Focal Loss 的聚焦因子 gamma 与目标面积反比关联：面积越小，gamma 越大，聚焦效果越强
- Safe Module 使用可变形卷积 (Deformable Convolution) 增强小目标区域的特征采样
- 整体改动较小，可嵌入各类 anchor-based 检测器

**关键结果**:
- 在 COCO 上 AP_S 较 Focal Loss baseline 提升约 2 个点
- 在 TinyPerson 数据集上 AP 提升约 3 个点
- 对极端小目标 (16x16 以下) 效果尤为明显

---

## 9. AugFPN

| 字段 | 内容 |
|------|------|
| **论文** | AugFPN: Improving Multi-scale Feature Learning for Object Detection |
| **作者** | Chaoxu Guo, Bin Fan et al. |
| **会议** | CVPR 2020 |
| **引用** | ~700+ |
| **论文链接** | https://arxiv.org/abs/2012.06872 |

**一句话总结**: 从三个维度增强 FPN 的多尺度特征融合能力：一致性监督、自适应空间特征融合和特征金字塔残差化。

**核心贡献**:
- 提出 **AugFPN** 架构，包含三个核心改进模块：
  1. **Consistent Supervision**: 对中间层特征施加与最终输出一致的监督信号
  2. **Adaptive Spatial Feature Fusion (ASFF)**: 自适应学习各层特征的空间融合权重
  3. **Residual Feature Augmentation (RFA)**: 利用最大池化下采样残差信息补充丢失的细节

**方法亮点**:
- Consistent Supervision 在 P3-P5 上引入辅助监督，解决 FPN 中间层特征学习不充分的问题
- ASFF 对每个空间位置学习不同尺度特征的权重，比简单相加更灵活
- RFA 将 P5 的残差信息 (max-pool 后的差值) 传递到 P4，弥补下采样信息损失

**关键结果**:
- 在 COCO test-dev 上 ResNet-101 backbone 达到 42.3 AP
- AP_S 较 FPN baseline 提升约 2.5 个点
- 三个模块均有独立增益，可组合使用

---

## 10. DetectoRS

| 字段 | 内容 |
|------|------|
| **论文** | DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution |
| **作者** | Siyuan Qiao, Liang-Chieh Chen, Alan Yuille |
| **会议** | CVPR 2021 |
| **引用** | ~1500+ |
| **论文链接** | https://arxiv.org/abs/2006.02334 |

**一句话总结**: 通过递归特征金字塔 (RFP) 和可切换空洞卷积 (SAC) 两个机制，显著增强检测器的多尺度特征表达能力。

**核心贡献**:
- 提出 **Recursive Feature Pyramid (RFP)**: 将 FPN 的输出反馈回 backbone，形成递归结构，使高层语义信息在多轮迭代中逐步渗透到低层特征
- 设计 **Switchable Atrous Convolution (SAC)**: 在同一卷积层中可切换不同空洞率 (atrous rate)，通过输入依赖的开关动态选择
- RFP 与 SAC 协同工作，RFP 负责全局多尺度融合，SAC 负责局部多尺度适应

**方法亮点**:
- RFP 使用一个额外的卷积层将 FPN 各层输出聚合后反馈到 backbone 的每个 stage
- SAC 的开关机制由一个 1x1 卷积 + sigmoid 门控实现，对不同输入自适应切换
- 递归深度通常设为 2 (两次 FPN 迭代即可获得显著增益)

**关键结果**:
- 在 COCO test-dev 上以 ResNet-101 backbone 达到 49.1 AP (Cascade Mask R-CNN)
- AP_S 较基线提升约 3 个点
- RFP 和 SAC 各自独立贡献约 1.5-2 AP 增益

---

## 11. EfficientDet

| 字段 | 内容 |
|------|------|
| **论文** | EfficientDet: Scalable and Efficient Object Detection |
| **作者** | Mingxing Tan, Ruoming Pang, Quoc V. Le |
| **会议** | CVPR 2020 |
| **引用** | ~8000+ |
| **论文链接** | |https://arxiv.org/abs/1911.09070 |

**一句话总结**: 提出可扩展的高效检测架构，通过 BiFPN 加权双向特征融合和复合缩放策略，在精度与速度之间取得最优平衡。

**核心贡献**:
- 提出 **BiFPN (Bi-directional Feature Pyramid Network)**: 加权双向特征融合，允许所有层级的特征自由双向流动
- 设计 **Compound Scaling** 策略：同时缩放 backbone 分辨率、FPN 层数、检测头维度，类似于 EfficientNet 的复合缩放
- 提出 **EfficientDet** 系列 (D0-D7)，覆盖从移动端到服务器端的全场景

**方法亮点**:
- BiFPN 使用可学习的归一化权重融合不同层特征，替代简单相加
- 双向路径：自底向上 + 自顶向下，多次迭代融合
- 使用 EfficientNet 作为 backbone，以 Depthwise Separable Convolution 降低计算量

**关键结果**:
- EfficientDet-D0: 34.3 AP, 仅 2.8 GFLOPs (YOLOv3 同等精度但计算量仅 1/8)
- EfficientDet-D7: 55.1 AP (test-dev), 与 NAS-FPN 精度相当但参数量少 4 倍
- AP_S 在高分辨率版本 (D5-D7) 上表现尤为突出

---

## 12. FCOS

| 字段 | 内容 |
|------|------|
| **论文** | FCOS: Fully Convolutional One-Stage Object Detection |
| **作者** | Zhiqi Tian, Chunhua Shen et al. |
| **会议** | ICCV 2019 |
| **引用** | ~10000+ |
| **论文链接** | https://arxiv.org/abs/1904.01355 |

**一句话总结**: 提出全卷积单阶段无锚框检测器，通过逐像素预测消除 anchor 设计的复杂性，对小目标检测友好。

**核心贡献**:
- 提出 **FCOS**：完全无 anchor 的单阶段检测器，将检测问题转化为逐像素 (per-pixel) 的分类与回归问题
- 设计 **Center-ness** 分支：预测每个位置到目标中心的归一化距离，用于抑制低质量预测
- 证明无 anchor 检测器可以达到甚至超越 anchor-based 检测器的性能

**方法亮点**:
- 每个特征图位置直接预测到目标框四边的距离 (l, t, r, b)，无需预设 anchor 尺寸和比例
- 多尺度预测中引入 **min/max scale constraints**：P3 负责 [0, 64], P4 负责 [64, 128] 等，避免重叠分配
- Center-ness 作为质量评分，乘入分类得分后再做 NMS

**关键结果**:
- 在 COCO test-dev 上 ResNet-101 backbone 达到 44.7 AP (单阶段 SOTA at the time)
- AP_S 较 RetinaNet 提升约 1 个点，得益于无 anchor 带来的更灵活的小目标匹配
- 推理速度与 RetinaNet 持平，但无需 anchor 超参数调优

---

## 延伸阅读

1. **Multi-scale Detection Survey**: Liu et al., "Deep Learning for Generic Object Detection: A Survey", IJCV 2019
2. **Small Object Detection in Remote Sensing**: Cheng et al., "Object Detection in Optical Remote Sensing Images: A Survey and A New Benchmark", ISPRS 2019
3. **NMS改进**: He et al., "Adaptive NMS: Refining Pedestrian Detection in a Crowd", CVPR 2019
4. **Transformer-based Detection**: Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
5. **数据增强策略**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
6. **YOLO系列小目标优化**: YOLOv8 官方文档中关于小目标训练的 tuning guide

---

### 相关文档

- [01-小目标检测挑战](../03-小目标检测专题/01-小目标检测挑战.md) — 小目标检测挑战
- [05-VisDrone检测综述](../03-小目标检测专题/05-VisDrone检测综述.md) — VisDrone 检测综述
- [03-SAHI切片推理](../05-推理优化专题/03-SAHI切片推理.md) — SAHI 切片推理

## 思考题

<details>
<summary>Q1: FPN 为什么对小目标检测至关重要？其局限性是什么？</summary>

FPN 通过自顶向下的路径将高层语义信息传递到高分辨率的低层特征图 (P2)，使得小目标在高分辨率特征上也能获得丰富的语义表征。局限性在于：
1. 仅单向 (自顶向下) 传递信息，自底向上的细节信息未充分利用；
2. 各层特征简单相加融合，忽略了不同尺度特征之间的重要性差异；
3. 最高分辨率层 (P2) 的感受野有限，可能缺乏足够上下文。

</details>

<details>
<summary>Q2: SNIP 的"尺寸归一化"思想与多尺度测试有何本质区别？</summary>

多尺度测试 (Multi-scale Testing) 在多个分辨率上运行完整检测器，然后融合所有结果；所有尺度的所有目标都会贡献损失/梯度。SNIP 的本质区别在于**选择性训练/推理**：每个尺度分支只处理"尺寸合适"的目标，对过大/过小目标不参与梯度计算。这相当于在训练阶段就隐式地实现了尺度特异性，避免了跨尺度的语义混叠 (semantic aliasing)。

</details>

<details>
<summary>Q3: SAHI 的核心优势是什么？在什么场景下不适用？</summary>

SAHI 的核心优势是**零训练成本**的即插即用方案，任何现有检测器都可以直接使用，无需重新训练。不适用场景包括：
1. 实时视频处理 (每帧切片推理带来延迟翻倍)；
2. 目标均匀分布且无小目标问题的场景 (切片反而引入冗余)；
3. 检测器本身已包含强大的多尺度特征融合机制时，增益递减。

</details>

<details>
<summary>Q4: 比较 ClusDet 和 SAHI 的异同点。</summary>

**相同点**: 都采用"裁剪+检测+合并"的范式来提升小目标检测。**不同点**:
1. ClusDet 使用密度图估计自动确定聚类区域，SAHI 使用固定网格切片；
2. ClusDet 需要修改模型架构并重新训练，SAHI 即插即用不需训练；
3. ClusDet 的裁剪区域由目标密度自适应确定，SAHI 的切片尺寸固定 (可配置 overlap)；
4. ClusDet 对密集目标场景更精准，SAHI 通用性更强。

</details>

<details>
<summary>Q5: FCOS 作为 anchor-free 检测器，为什么对小目标有天然优势？</summary>

1. **无 anchor 尺寸/比例约束**: anchor-based 方法中预设 anchor 与小目标的 IoU 往往很低，导致正样本不足；FCOS 直接在目标内部的每个位置预测，小目标也能获得足够多的正样本；
2. **中心采样策略**: FCOS 以目标中心区域为正样本，避免了 anchor 匹配中尺度不敏感的问题；
3. **Center-ness 质量评分**: 有效抑制了小目标附近低质量预测框的干扰。

</details>

<details>
<summary>Q6: 如果要在遥感图像中检测大量小目标，你会如何组合上述技术？</summary>

推荐技术组合路径:
1. **数据预处理**: SAHI 切片推理，将大图裁剪为 640x640 的重叠切片 (overlap=0.2)；
2. **骨干网络**: 使用 EfficientNet 作为 backbone，在效率和精度间取得平衡；
3. **特征融合**: 采用 BiFPN (来自 EfficientDet) 或 AugFPN 替代标准 FPN；
4. **检测头**: 使用 FCOS anchor-free head，对小目标正样本更友好；
5. **损失函数**: 采用 Focus Loss 对小目标赋予更高聚焦权重；
6. **后处理**: 使用 Soft-NMS 处理密集重叠检测结果。

</details>
