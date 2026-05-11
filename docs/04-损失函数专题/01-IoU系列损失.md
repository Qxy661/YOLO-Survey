# IoU 系列损失函数

> IoU（Intersection over Union）是目标检测中最核心的评估指标，围绕它衍生出一系列损失函数。
> 本专题从最基础的 IoU Loss 出发，逐步推导 GIoU、DIoU、CIoU、SIoU、WIoU v3 和 Inner-IoU，
> 理解每一代改进解决了什么问题、引入了什么新视角。

---

## 目录

1. [1. IoU Loss -- 起点](#1-iou-loss----起点)
2. [2. GIoU（Generalized IoU）](#2-giougeneralized-iou)
3. [3. DIoU（Distance IoU）](#3-dioudistance-iou)
4. [4. CIoU（Complete IoU）](#4-cioucomplete-iou)
5. [5. SIoU（SCYLLA-IoU）](#5-siouscylla-iou)
6. [6. WIoU v3（Wise-IoU v3）](#6-wiou-v3wise-iou-v3)
7. [7. Inner-IoU](#7-inner-iou)
8. [8. 全系列对比总结](#8-全系列对比总结)
9. [9. 在不同 YOLO 版本中的应用](#9-在不同-yolo-版本中的应用)

---


## 1. IoU Loss -- 起点

### 1.1 定义

IoU 衡量预测框与真实框的重叠程度：

```
IoU = |A ∩ B| / |A ∪ B|
```

IoU Loss 的定义：

```
L_IoU = 1 - IoU
```

### 1.2 局限

```
场景1: 不相交                场景2: 包含
  +-------+                   +-------+
  | pred  |                   | gt    |
  +-------+                   | +---+ |
      +-------+               | |p  | |
      |  gt   |               | +---+ |
      +-------+               +-------+

  IoU = 0                    IoU = 0.5
  L_IoU = 1                  L_IoU = 0.5
  梯度 = 0?                  无法区分"关系"
```

**问题：**
- 当两个框不相交时，IoU = 0，Loss 恒为 1，无法提供梯度方向。
- 无法区分两个框的相对位置关系（距离、方向、长宽比）。

---

## 2. GIoU（Generalized IoU）

> 来源：Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019

### 2.1 核心思想

引入**最小封闭框（Minimum Enclosing Box）**，即能同时包含 pred 和 gt 的最小矩形。

```
+--------------------+
|    C (enclosing)   |
|  +-------+         |
|  | pred  |         |
|  +-------+         |
|      +-------+     |
|      |  gt   |     |
|      +-------+     |
+--------------------+
|C| = 最小封闭框面积
```

### 2.2 公式

```
GIoU = IoU - (|C - (A ∪ B)|) / |C|

L_GIoU = 1 - GIoU
```

### 2.3 改进点

- 当 IoU = 0 时，GIoU 趋近于 -1（当两框相距无穷远），Loss 趋近于 2，产生梯度。
- GIoU <= IoU，范围 [-1, 1]。
- **仍存在的问题：** 当一个框完全包含另一个框时，GIoU 退化为 IoU（因为 C = A ∪ B）。

---

## 3. DIoU（Distance IoU）

> 来源：Zheng et al., "Distance-IoU Loss", AAAI 2020

### 3.1 核心思想

直接用**中心点距离**代替封闭框面积，更高效地惩罚距离。

```
    +-------+
    | pred  |
    |   *c1 |      d = 两中心点的欧氏距离
    +-------+
              d
    +-------+
    |  *c2  |
    |   gt  |
    +-------+

    c = 最小封闭框的对角线长度
```

### 3.2 公式

```
DIoU = IoU - (d^2) / (c^2)

L_DIoU = 1 - DIoU
```

其中：
- `d` = 预测框与真实框中心点的欧氏距离
- `c` = 最小封闭框的对角线长度

### 3.3 改进点

- 收敛速度比 GIoU 更快（直接优化距离，而非间接通过面积）。
- 即使在包含关系下也能产生有效梯度。
- **仍存在的问题：** 没有考虑长宽比（aspect ratio）。

---

## 4. CIoU（Complete IoU）

> 来源：Zheng et al., "Distance-IoU Loss", AAAI 2020（同一篇论文）

### 4.1 核心思想

在 DIoU 基础上，增加**长宽比一致性**惩罚项。

### 4.2 公式

```
CIoU = IoU - (d^2 / c^2) - alpha * v

其中:
  v = (4 / pi^2) * (arctan(w_gt / h_gt) - arctan(w_pred / h_pred))^2
  alpha = v / ((1 - IoU) + v)

L_CIoU = 1 - CIoU
```

### 4.3 参数解读

```
v:  长宽比一致性度量
    w_gt/h_gt  vs  w_pred/h_pred
    当长宽比完全一致时 v = 0

alpha: 自适应权重
    IoU 低时 alpha 小 -> 优先优化距离
    IoU 高时 alpha 大 -> 优先优化长宽比
```

### 4.4 改进点

- 同时优化重叠面积、中心距离、长宽比三个因素。
- **仍存在的问题：** v 的梯度计算中，w 和 h 是耦合的（通过 arctan），优化效率不够高。

---

## 5. SIoU（SCYLLA-IoU）

> 来源：Gevorgyan, "SIoU Loss: More Powerful Learning for Bounding Box Regression", 2022

### 5.1 核心思想

引入**角度感知（Angle-aware）**机制，将方向信息纳入损失计算。

```
角度惩罚示意:

    pred
    +---+
    |   |
    +---+  \  theta = 角度
            \  惩罚方向不对的回归路径
             \
              * gt 中心

传统: 直接优化 dx, dy (水平/垂直)
SIoU: 考虑 pred 中心相对于 gt 中心的角度
```

### 5.2 公式

```
SIoU = IoU - (Delta + Omega)

其中:
  Lambda = 1 - 2 * sin^2(arcsin(d_y / d) - pi/4)
  Delta = (Lambda_x^4 + Lambda_y^4) / 2    # 距离惩罚（角度感知）
  Omega = (1 - e^(-2*w_w))^theta + (1 - e^(-2*w_h))^theta   # 形状惩罚

  w_w = |w_gt - w_pred| / max(w_gt, w_pred)
  w_h = |h_gt - h_pred| / max(h_gt, h_pred)
  theta: 形状惩罚的控制参数（通常 2~4）

L_SIoU = 1 - SIoU
```

### 5.3 四个组成部分

| 组成部分 | 含义 | 作用 |
|---------|------|------|
| Angle Cost | 中心连线的角度 | 引导回归沿正确方向 |
| Distance Cost | 角度感知的距离 | 非对称惩罚 dx, dy |
| Shape Cost | 长宽差异 | 精细化形状对齐 |
| IoU | 重叠度 | 基础度量 |

---

## 6. WIoU v3（Wise-IoU v3）

> 来源：Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism", 2023

### 6.1 核心思想

引入**动态非单调聚焦机制（Dynamic Non-Monotonic Focusing）**，根据样本质量自适应调整梯度。

```
传统损失:                WIoU v3:
                          梯度
  Loss                    |   * (低质量样本)
  |   /                   |  *
  |  /                    | *    (高质量样本 - 降低权重)
  | /                     |*
  |/                      +-------------> IoU
  +------> IoU

  所有样本同等权重         低质量样本获得更多关注
```

### 6.2 公式

```
WIoU v3 = r * L_IoU

其中:
  r = beta * delta^alpha / (beta * delta^alpha + 1 - delta^alpha)    # 聚焦系数
  delta = IoU / IoU_mean   # 归一化 IoU（相对于当前 batch 均值）
  alpha, beta: 超参数（alpha 控制单调性，beta 控制聚焦强度）

L_WIoU_v3 = r * (1 - IoU)
```

### 6.3 关键特性

- **非单调：** 与 Focal Loss 不同，WIoU v3 对"中等质量"样本降低权重，而非简单地降低"高质量"权重。
- **动态：** 聚焦系数随训练进程自适应变化。
- **无需锚框先验：** 纯基于 IoU 的自适应机制。

---

## 7. Inner-IoU

> 来源：Zhang et al., "Inner-IoU: More Efficient Intersection over Union Loss", 2023

### 7.1 核心思想

使用一个**辅助较小框（Inner Box）**来计算 IoU，提高对小目标和高 IoU 阈值场景的敏感性。

```
普通 IoU:                    Inner-IoU:
+------------------+         +------------------+
|    pred          |         |    pred          |
|  +------------+  |         |  +--------+      |
|  |            |  |         |  | inner  |      |
|  +------------+  |         |  +--------+      |
|  +------------+  |         |  +--------+      |
|  |     gt     |  |         |  | inner  |      |
|  +------------+  |         |  +--------+      |
+------------------+         +------------------+
  计算整个框的 IoU            计算缩小后框的 IoU
                             ratio < 1 缩放比例
```

### 7.2 公式

```
Inner-IoU = IoU_inner

其中:
  inner_pred = 缩放 pred 框 (中心不变, 宽高 * ratio)
  inner_gt   = 缩放 gt 框   (中心不变, 宽高 * ratio)
  IoU_inner  = IoU(inner_pred, inner_gt)

  ratio: 缩放比例 (通常 0.5~0.8)

L_Inner-IoU = 1 - Inner-IoU
```

### 7.3 使用方式

Inner-IoU 不是独立的损失，而是一种**策略**，可以与任意 IoU 变体组合：

```
Inner-GIoU, Inner-DIoU, Inner-CIoU, Inner-SIoU ...
```

---

## 8. 全系列对比总结

| 损失函数 | 解决的核心问题 | 新增考量因素 | 收敛速度 | 小目标表现 | YOLO 版本采用 |
|---------|--------------|------------|---------|-----------|-------------|
| IoU | 基础度量 | 重叠面积 | 慢（不相交无梯度） | 差 | YOLOv1~v3 |
| GIoU | 不相交无梯度 | 最小封闭框 | 较慢 | 一般 | YOLOv5 早期 |
| DIoU | 包含关系退化 | 中心距离 | 快 | 较好 | YOLOv5, YOLOv7 |
| CIoU | 忽略长宽比 | 中心距离 + 长宽比 | 快 | 较好 | YOLOv5 (默认), YOLOv7 |
| SIoU | 忽略回归方向 | 角度 + 距离 + 形状 | 更快 | 好 | YOLOv5/v8 可选 |
| WIoU v3 | 样本权重不均 | 动态聚焦 | 快 | 好 | YOLOv8 可选 |
| Inner-IoU | 低 IoU 阈值不敏感 | 辅助缩小框 | 快 | 好 | 可与任意 IoU 组合 |

---

## 9. 在不同 YOLO 版本中的应用

### 9.1 YOLOv5 / YOLOv7

```python
# YOLOv5 hyp.yaml
box: 7.5    # box loss gain
cls: 0.5    # cls loss gain
obj: 1.0    # obj loss gain (CIoU 默认)

# ultralytics/yolo/utils/loss.py 中修改
# ComputeLoss 类里将 CIoU 替换为其他 IoU 变体
```

### 9.2 YOLOv8 / Ultralytics

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
# 通过配置修改损失类型
model.train(data="coco.yaml", box_loss="ciou")
# 可选: "ciou", "siou", "wiou"
```

### 9.3 自定义损失注册

```python
def select_iou_loss(pred_boxes, target_boxes, type="ciou"):
    if type == "ciou":
        return ciou_loss(pred_boxes, target_boxes)
    elif type == "siou":
        return siou_loss(pred_boxes, target_boxes)
    elif type == "wiou":
        return wiou_v3_loss(pred_boxes, target_boxes)
    # ... 其他变体
```

---

## 延伸阅读

### 相关文档

- [损失函数基础](../01-基础概念/03-损失函数.md) — IoU/L1/Smooth L1 基础
- [分类损失](02-分类损失.md) — Focal Loss 等分类损失
- [DFL 分布式焦点损失](03-分布式焦点损失DFL.md) — YOLOv8 的分布式回归
- [YOLOv8统一架构](../02-YOLO演进专题/04-YOLOv8统一架构.md) — YOLOv8 损失函数组合
- [损失论文卡片](../06-论文导读合集/loss-papers.md) — 9 篇损失函数论文

### 论文原文

- [GIoU 原论文](https://arxiv.org/abs/1902.09630)
- [DIoU 和 CIoU 原论文](https://arxiv.org/abs/1911.08287)
- [SIoU 原论文](https://arxiv.org/abs/2205.12740)
- [Wise-IoU 原论文](https://arxiv.org/abs/2301.10051)
- [Inner-IoU 原论文](https://arxiv.org/abs/2311.00141)
- [Ultralytics Loss 源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py)

---

<details>
<summary>思考题</summary>

1. 为什么 CIoU 中的 alpha 权重是 `v / ((1 - IoU) + v)` 这样的设计？如果不加 alpha 会怎样？
2. WIoU v3 的"非单调"聚焦与 Focal Loss 的"单调"聚焦在本质上有何区别？分别适合什么场景？
3. Inner-IoU 的 ratio 参数过大（接近 1）或过小（接近 0）时，分别会出现什么问题？
4. 在你的实际项目中，IoU 损失的选择对最终 mAP 的影响有多大？请设计一个消融实验来验证。
5. SIoU 的角度惩罚在旋转目标检测（如 DOTA 数据集）中是否天然适用？为什么？

</details>
