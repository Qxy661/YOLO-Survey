# VisDrone 数据集详解

> VisDrone 是目前无人机目标检测领域最具权威性的基准数据集之一，由天津大学 AISKYEYE 团队构建，广泛用于评估 YOLO 系列模型在航拍视角下的检测性能。

## 1. 数据集概述

VisDrone 数据集包含由中国数十个城市上空的无人机拍摄的图像序列，涵盖了多种场景、天气条件和光照变化。

| 属性 | 说明 |
|------|------|
| **采集方式** | 无人机航拍（多旋翼平台，高度 10-800m） |
| **图像分辨率** | 大部分为 2000x1500，部分为 3840x2160 |
| **类别数** | 10 个目标类别 + 1 个 ignored 区域 |
| **任务类型** | 目标检测、目标跟踪、计数、密度估计 |
| **标注规模** | 约 10,209 张图像，超过 500 万个标注框 |

### 10 个目标类别

| 类别 ID | 类别名称 | 说明 |
|---------|---------|------|
| 0 | pedestrian | 行人 |
| 1 | people | 人群（密度较高时的聚合标注） |
| 2 | bicycle | 自行车 |
| 3 | car | 汽车 |
| 4 | van | 面包车/厢式货车 |
| 5 | truck | 卡车 |
| 6 | tricycle | 三轮车 |
| 7 | awning-tricycle | 篷式三轮车 |
| 8 | bus | 公共汽车 |
| 9 | motor | 摩托车 |

> **注意**: ignored regions (category=0) 和 others (category=11) 在评估时通常被忽略。

## 2. 数据下载

### 官方下载地址

| 数据集版本 | 下载地址 |
|-----------|---------|
| VisDrone2019-DET | https://github.com/VisDrone/VisDrone-Dataset |
| VisDrone2021-DET | https://github.com/VisDrone/VisDrone2021-DET |

### 数据集划分

| 子集 | 图像数量 | 用途 |
|------|---------|------|
| train | 6,471 | 模型训练 |
| val | 548 | 超参调优与验证 |
| test-dev | 1,610 | 离线测试（排行榜提交） |
| test-challenge | 1,580 | 挑战赛测试 |

```bash
# 克隆数据集仓库
git clone https://github.com/VisDrone/VisDrone-Dataset.git
cd VisDrone-Dataset

# 目录结构
# VisDrone2019-DET-train/
#   ├── images/
#   │   ├── 0000001_00000_d_0000001.jpg
#   │   └── ...
#   └── annotations/
#       ├── 0000001_00000_d_0000001.txt
#       └── ...
```

## 3. 标注格式

VisDrone 使用文本文件进行标注，每行代表一个目标实例。

### 标注字段

每行格式为逗号分隔的值：

```
<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
```

| 字段 | 类型 | 说明 |
|------|------|------|
| bbox_left | int | 边界框左上角 x 坐标 |
| bbox_top | int | 边界框左上角 y 坐标 |
| bbox_width | int | 边界框宽度（像素） |
| bbox_height | int | 边界框高度（像素） |
| score | float | 置信度（检测任务中使用，标注文件中为 1 或 0） |
| object_category | int | 目标类别编号（0-11） |
| truncation | int | 截断程度：0=未截断，1=部分截断 |
| occlusion | int | 遮挡程度：0=未遮挡，1=部分遮挡，2=大面积遮挡 |

### 标注示例

```
558,356,50,97,1,4,0,0
702,328,72,152,1,4,0,0
810,249,44,65,1,1,0,2
```

## 4. 数据统计分析

### 4.1 各类别数量分布

| 类别 | 训练集 | 验证集 | 占比 |
|------|--------|--------|------|
| pedestrian | 165,736 | 11,846 | 33.5% |
| people | 46,187 | 3,434 | 9.3% |
| bicycle | 12,879 | 913 | 2.6% |
| car | 258,064 | 17,456 | 52.0% |
| van | 20,415 | 1,529 | 4.1% |
| truck | 9,147 | 687 | 1.8% |
| tricycle | 6,324 | 458 | 1.3% |
| awning-tricycle | 4,218 | 312 | 0.8% |
| bus | 3,892 | 291 | 0.8% |
| motor | 35,023 | 2,568 | 7.1% |

> **类别不平衡明显**：car 和 pedestrian 合计占比超过 85%，而 awning-tricycle 和 bus 严重不足。

### 4.2 目标尺寸分布

| 尺寸类别 | 像素面积 | 比例 | 检测难度 |
|---------|---------|------|---------|
| Tiny | < 32x32 | ~40% | 极难 |
| Small | 32x32 ~ 96x96 | ~35% | 较难 |
| Medium | 96x96 ~ 288x288 | ~20% | 中等 |
| Large | > 288x288 | ~5% | 较易 |

> **小目标问题突出**：超过 75% 的目标面积小于 96x96 像素，这是无人机场景的核心挑战。

### 4.3 遮挡程度统计

| 遮挡等级 | 占比 | 说明 |
|---------|------|------|
| 无遮挡 (0) | ~60% | 目标完全可见 |
| 部分遮挡 (1) | ~28% | 目标被其他物体部分遮挡 |
| 大面积遮挡 (2) | ~12% | 目标仅少量可见 |

## 5. YOLO 格式转换

VisDrone 原始标注需要转换为 YOLO 格式（归一化的中心点坐标）。

### 5.1 转换脚本

```python
"""visdrone_to_yolo.py - 将 VisDrone 标注转换为 YOLO 格式"""
import os
from pathlib import Path


def convert_visdrone_to_yolo(
    annotations_dir: str,
    output_dir: str,
    img_width: int = 0,
    img_height: int = 0,
) -> None:
    """将 VisDrone 标注文件转换为 YOLO 格式。

    VisDrone: x, y, w, h (左上角 + 宽高)
    YOLO: cx, cy, w, h (中心点 + 宽高，归一化到 0-1)
    """
    os.makedirs(output_dir, exist_ok=True)
    # VisDrone 类别映射: 原始 ID -> 从 0 开始的连续 ID
    category_map = {i: i for i in range(10)}

    for ann_file in Path(annotations_dir).glob("*.txt"):
        lines = []
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                score = int(parts[4])
                category = int(parts[5])

                # 跳过 ignored regions 和 others
                if category == 0 or category == 11:
                    continue

                # 跳过无效框
                if w <= 0 or h <= 0 or score == 0:
                    continue

                # 类别 ID 映射（从 0 开始）
                yolo_cls = category - 1

                # 转换为 YOLO 格式（归一化中心坐标）
                # 注意：需要知道图像尺寸才能归一化
                # 此处假设使用原始图像尺寸，实际使用时需根据图像调整
                cx = (x + w / 2) / img_width
                cy = (y + h / 2) / img_height
                nw = w / img_width
                nh = h / img_height

                # 裁剪到 [0, 1] 范围
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 写入 YOLO 格式标注
        out_file = Path(output_dir) / ann_file.name
        with open(out_file, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    from PIL import Image

    # 遍历图像获取尺寸并转换
    img_dir = "VisDrone2019-DET-train/images"
    ann_dir = "VisDrone2019-DET-train/annotations"
    out_dir = "VisDrone2019-DET-train/labels_yolo"

    for ann_file in Path(ann_dir).glob("*.txt"):
        img_name = ann_file.stem + ".jpg"
        img_path = Path(img_dir) / img_name
        if img_path.exists():
            img = Image.open(img_path)
            w, h = img.size
            convert_visdrone_to_yolo(
                str(ann_file.parent), out_dir, img_width=w, img_height=h
            )
```

> **关键转换逻辑**：`(x, y, w, h)` -> `(cx, cy, w, h)`，其中 `cx = (x + w/2) / img_width`，`cy = (y + h/2) / img_height`，宽高也需归一化。

## 6. 数据集 YAML 配置

创建 `visdrone.yaml` 供 Ultralytics YOLOv8 使用：

```yaml
# visdrone.yaml - VisDrone 数据集配置文件
path: ./datasets/VisDrone  # 数据集根目录
train: images/train         # 训练集图像路径（相对于 path）
val: images/val             # 验证集图像路径
test: images/test-dev       # 测试集图像路径（可选）

# 类别名称（必须与标签文件中的类别 ID 对应）
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

# 可选：自定义参数
download: https://github.com/VisDrone/VisDrone-Dataset  # 自动下载链接
```

### 目录结构要求

```
datasets/VisDrone/
├── images/
│   ├── train/       # 6471 张训练图像
│   ├── val/         # 548 张验证图像
│   └── test-dev/    # 1610 张测试图像
├── labels/
│   ├── train/       # 训练集 YOLO 格式标注
│   ├── val/         # 验证集 YOLO 格式标注
│   └── test-dev/    # 测试集 YOLO 格式标注
└── visdrone.yaml
```

## 7. 常见问题与应对策略

### 7.1 类别不平衡

| 策略 | 说明 |
|------|------|
| 类别加权损失 | 使用 `class_weights` 参数对稀有类别赋予更大权重 |
| 过采样 | 对少样本类别进行图像复制或数据增强 |
| MixUp / Mosaic | 增强数据多样性，间接缓解不平衡 |
| focal loss | 降低易分类样本的权重，关注困难样本 |

### 7.2 小目标检测困难

| 策略 | 说明 |
|------|------|
| 提高输入分辨率 | 将 `imgsz` 从 640 提升至 1280 |
| 添加小目标检测层 | P2 检测头（160x160 特征图） |
| SAHI 切片推理 | 将大图切成小块分别检测后合并 |
| 多尺度训练 | 训练时随机选择多个输入尺寸 |

### 7.3 数据集划分一致性

> 确保不同版本 YOLO 实验使用相同的 train/val/test 划分，否则实验结果不可比。建议固定随机种子并记录划分方式。

---

## 延伸阅读

- [VisDrone 官方 GitHub](https://github.com/VisDrone/VisDrone-Dataset)
- [Ultralytics VisDrone 教程](https://docs.ultralytics.com/datasets/detect/visdrone/)
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)

## 思考题

<details>
<summary>1. VisDrone 数据集中为什么 people 和 pedestrian 要分开标注？这种设计对检测模型有什么影响？</summary>

people 和 pedestrian 的区分基于场景密度：当行人聚集且难以单独区分时标注为 people，个体清晰可辨时标注为 pedestrian。这种设计反映了实际应用中"个体检测"和"群体检测"的不同需求，但也导致模型在推理时可能混淆这两个类别。训练时可以考虑合并为单一类别以简化任务。

</details>

<details>
<summary>2. 如果你计划在 VisDrone 上训练 YOLOv8，输入分辨率应该选择 640 还是 1280？请分析利弊。</summary>

选择 640：训练速度快、显存占用小，但小目标信息损失严重。选择 1280：保留更多小目标细节、mAP 通常更高，但训练时间约为 640 的 4 倍，显存需求翻倍。考虑到 VisDrone 中 75% 的目标为小目标，建议至少使用 1280 分辨率。如果显存不足，可使用梯度累积或混合精度训练。

</details>

<details>
<summary>3. COCO 格式和 YOLO 格式的标注有什么本质区别？为什么 YOLO 系列使用归一化坐标？</summary>

COCO 使用绝对像素坐标 `(x, y, w, h)`，而 YOLO 使用归一化的中心点坐标 `(cx/width, cy/height, w/width, h/height)`。归一化的优势在于：(1) 与输入分辨率无关，便于统一处理不同尺寸的图像；(2) 数值范围在 0-1 之间，有利于网络训练时的数值稳定性；(3) 简化了数据增强（如 mosaic）时的坐标变换。

</details>
