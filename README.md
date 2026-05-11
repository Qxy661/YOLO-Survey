# YOLO 目标检测领域系统调研

> 一份系统性的 YOLO 系列目标检测算法调研文档，覆盖从基础概念到前沿研究的完整知识体系。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-36%20documents-brightgreen.svg)](docs/)

---

## 一句话简介

对 YOLO 系列目标检测算法进行系统性调研，涵盖算法演进、小目标检测、损失函数设计、推理优化等核心专题，附论文导读与实验指南。

---

## 项目结构

```
YOLO-Survey/
├── README.md
├── docs/
│   ├── 00-导读与学习路线.md
│   ├── 01-基础概念/
│   │   ├── 01-目标检测概述.md
│   │   ├── 02-评估指标体系.md
│   │   ├── 03-损失函数.md
│   │   └── 04-数据增强策略.md
│   ├── 02-YOLO演进专题/
│   │   ├── 01-YOLOv1-v3奠基时代.md
│   │   ├── 02-YOLOv4-v5工业时代.md
│   │   ├── 03-YOLOv6-v7竞争时代.md
│   │   ├── 04-YOLOv8统一架构.md
│   │   ├── 05-YOLOv9-v11最新进展.md
│   │   └── 06-YOLO系列全景对比.md
│   ├── 03-小目标检测专题/
│   │   ├── 01-小目标检测挑战.md
│   │   ├── 02-多尺度特征融合.md
│   │   ├── 03-高分辨率策略.md
│   │   ├── 04-注意力机制.md
│   │   └── 05-VisDrone检测综述.md
│   ├── 04-损失函数专题/
│   │   ├── 01-IoU系列损失.md
│   │   ├── 02-分类损失.md
│   │   └── 03-分布式焦点损失DFL.md
│   ├── 05-推理优化专题/
│   │   ├── 01-后处理技术.md
│   │   ├── 02-测试时增强TTA.md
│   │   ├── 03-SAHI切片推理.md
│   │   └── 04-模型部署优化.md
│   ├── 06-论文导读合集/
│   │   ├── yolo-papers.md
│   │   ├── attention-papers.md
│   │   ├── small-object-papers.md
│   │   └── loss-papers.md
│   └── 07-实验与实践/
│       ├── 01-VisDrone数据集详解.md
│       ├── 02-实验复现指南.md
│       ├── 03-消融实验设计.md
│       └── 04-我们的实验总结.md
├── mindmaps/
│   ├── yolo-evolution.md
│   ├── small-object-taxonomy.md
│   └── reading-order.md
└── references/
    └── paper-list.md
```

---

## 文档列表

### 01 - 基础概念

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [01-目标检测概述](docs/01-基础概念/01-目标检测概述.md) | 目标检测的定义、任务类型和经典方法 | Two-stage, One-stage, Anchor-based, Anchor-free, DETR |
| [02-评估指标体系](docs/01-基础概念/02-评估指标体系.md) | mAP、IoU、Precision/Recall 等核心指标 | mAP@0.5, mAP@0.5:0.95, COCOeval |
| [03-损失函数](docs/01-基础概念/03-损失函数.md) | 检测任务中的损失函数设计思路 | 分类损失, 回归损失, IoU系列, Focal Loss |
| [04-数据增强策略](docs/01-基础概念/04-数据增强策略.md) | Mosaic、MixUp、Copy-Paste 等增强方法 | 在线增强, 离线增强, YOLOv8 增强参数 |

### 02 - YOLO 演进专题

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [01-YOLOv1-v3奠基时代](docs/02-YOLO演进专题/01-YOLOv1-v3奠基时代.md) | YOLOv1/v2/v3 的核心创新与演进 | Grid, Anchor, Darknet, 多尺度预测 |
| [02-YOLOv4-v5工业时代](docs/02-YOLO演进专题/02-YOLOv4-v5工业时代.md) | YOLOv4 的 BoF/BoS 和 YOLOv5 的工程化 | CSP, Mosaic, PANet, AutoAnchor |
| [03-YOLOv6-v7竞争时代](docs/02-YOLO演进专题/03-YOLOv6-v7竞争时代.md) | YOLOv6/v7/PP-YOLOE 的技术竞争 | RepVGG, E-ELAN, SIoU, TAL |
| [04-YOLOv8统一架构](docs/02-YOLO演进专题/04-YOLOv8统一架构.md) | YOLOv8 的统一多任务架构详解 | C2f, Decoupled Head, Anchor-Free, DFL |
| [05-YOLOv9-v11最新进展](docs/02-YOLO演进专题/05-YOLOv9-v11最新进展.md) | YOLOv9/v10/v11/v12 最新技术 | PGI, NMS-free, C3k2, C2PSA, Area Attention |
| [06-YOLO系列全景对比](docs/02-YOLO演进专题/06-YOLO系列全景对比.md) | YOLO 全系列横向对比与选型建议 | 性能对比, 技术时间线, 选型指南 |

### 03 - 小目标检测专题

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [01-小目标检测挑战](docs/03-小目标检测专题/01-小目标检测挑战.md) | 小目标检测的核心难点分析 | 特征退化, 样本不均衡, VisDrone |
| [02-多尺度特征融合](docs/03-小目标检测专题/02-多尺度特征融合.md) | FPN/PANet/BiFPN/PAFPN 对比 | 双向融合, 加权融合, P2层 |
| [03-高分辨率策略](docs/03-小目标检测专题/03-高分辨率策略.md) | 输入分辨率与 SAHI 切片推理 | 1280px, 超分辨率, P2检测头 |
| [04-注意力机制](docs/03-小目标检测专题/04-注意力机制.md) | SE/CBAM/CA/ECA/C2PSA 对比分析 | Channel Attention, Spatial Attention, CBAM有害 |
| [05-VisDrone检测综述](docs/03-小目标检测专题/05-VisDrone检测综述.md) | VisDrone 数据集与检测方法梳理 | 无人机视角, ClusDet, QueryDet |

### 04 - 损失函数专题

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [01-IoU系列损失](docs/04-损失函数专题/01-IoU系列损失.md) | IoU→GIoU→DIoU→CIoU→SIoU→WIoU→Inner-IoU | 回归损失演进, 动态聚焦 |
| [02-分类损失](docs/04-损失函数专题/02-分类损失.md) | BCE/Focal/VariFocal/QFL 对比 | 类别不平衡, IoU-aware |
| [03-分布式焦点损失DFL](docs/04-损失函数专题/03-分布式焦点损失DFL.md) | DFL 的原理与对小目标的效果 | 分布式回归, YOLOv8 |

### 05 - 推理优化专题

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [01-后处理技术](docs/05-推理优化专题/01-后处理技术.md) | NMS/Soft-NMS/DIoU-NMS/WBF 对比 | 后处理, NMS-free |
| [02-测试时增强TTA](docs/05-推理优化专题/02-测试时增强TTA.md) | 多尺度翻转等测试时增强策略 | 推理开销, 精度-速度权衡 |
| [03-SAHI切片推理](docs/05-推理优化专题/03-SAHI切片推理.md) | SAHI 切片推理原理与参数调优 | 大图切片, 重叠率, 小目标提升 |
| [04-模型部署优化](docs/05-推理优化专题/04-模型部署优化.md) | ONNX/TensorRT/量化/剪枝/蒸馏 | 导出格式, INT8, 知识蒸馏 |

### 06 - 论文导读合集

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [yolo-papers](docs/06-论文导读合集/yolo-papers.md) | YOLO 系列 11 篇核心论文卡片 | YOLOv1-v11 |
| [attention-papers](docs/06-论文导读合集/attention-papers.md) | 注意力机制 9 篇论文卡片 | SE, CBAM, CA, ECA, GAM, SimAM |
| [small-object-papers](docs/06-论文导读合集/small-object-papers.md) | 小目标检测 12 篇论文卡片 | FPN, SNIP, SAHI, ClusDet, FCOS |
| [loss-papers](docs/06-论文导读合集/loss-papers.md) | 损失函数 9 篇论文卡片 | Focal, IoU系列, DFL, VariFocal |

### 07 - 实验与实践

| 文档 | 主题 | 关键词 |
|------|------|--------|
| [01-VisDrone数据集详解](docs/07-实验与实践/01-VisDrone数据集详解.md) | 数据集下载、格式转换、统计分析 | YOLO格式, 类别分布 |
| [02-实验复现指南](docs/07-实验与实践/02-实验复现指南.md) | 环境配置、训练脚本、评估流程 | PyTorch, Ultralytics, SAHI |
| [03-消融实验设计](docs/07-实验与实践/03-消融实验设计.md) | 如何设计消融实验与归因分析 | 单变量, 增量贡献 |
| [04-我们的实验总结](docs/07-实验与实践/04-我们的实验总结.md) | 5 轮实验复盘与经验教训 | Baseline→SAHI→P2+CBAM |

---

## 快速开始

**推荐起点**：先阅读 [导读与学习路线](docs/00-导读与学习路线.md)，根据你的时间和目标选择合适的学习路线。

| 学习目标 | 路线 | 预计时间 |
|----------|------|:--------:|
| 快速了解 YOLO 全貌 | 入门路线 | 2 天 |
| 深入掌握核心技术 | 进阶路线 | 5 天 |
| 具备独立研究能力 | 研究路线 | 10 天 |

```bash
# 克隆到本地阅读
git clone <repo-url>
cd YOLO-Survey

# 推荐使用 VS Code + Markdown Preview Enhanced 插件
code docs/00-导读与学习路线.md
```

---

## 思维导图

项目附带的思维导图帮助你快速建立知识框架：

| 文件 | 内容 |
|------|------|
| [yolo-evolution.md](mindmaps/yolo-evolution.md) | YOLO 系列版本演进脉络 |
| [small-object-taxonomy.md](mindmaps/small-object-taxonomy.md) | 小目标检测技术分类体系 |
| [reading-order.md](mindmaps/reading-order.md) | 推荐阅读顺序流程图 |

---

## 引用与致谢

### 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{yolo-survey2026,
  title={YOLO: A Systematic Survey of Object Detection},
  author={YOLO-Survey Contributors},
  year={2026},
  howpublished={\url{<repo-url>}}
}
```

### 致谢

- 感谢 YOLO 系列所有原作者的开创性工作
- 感谢开源社区提供的预训练模型和工具链
- 本项目文档基于公开论文和技术博客整理，版权归原始作者所有

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

> :point_right: **开始阅读**：[导读与学习路线](docs/00-导读与学习路线.md)
