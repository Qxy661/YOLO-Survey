# YOLO Evolution: From YOLOv1 to YOLO12

```mermaid
mindmap
  root((YOLO Evolution))
    **YOLOv1 (2016)**
      Unified detection as regression
      Single neural network
      7x7 grid prediction
      Real-time speed breakthrough
    **YOLOv2 / YOLO9000 (2017)**
      Batch normalization
      Anchor boxes via k-means
      Multi-scale training
      Darknet-19 backbone
      WordTree for 9000+ categories
    **YOLOv3 (2018)**
      Darknet-53 backbone
      Multi-scale prediction (3 scales)
      Feature Pyramid Network ideas
      Logistic regression for class prediction
    **YOLOv4 (2020)**
      CSPDarknet53 backbone
      SPP + PANet neck
      Mosaic data augmentation
      CIoU loss
      Bag of freebies & specials
    **YOLOv5 (2020)**
      PyTorch re-implementation
      Auto anchor & auto augmentation
      Focus layer
      CSP bottleneck design
      Production-ready deployment
    **YOLOv6 (2022)**
      RepVGG-style backbone
      Efficient decoupled head
      SIoU / TaskAligned assigner
      Quantization-friendly design
    **YOLOv7 (2022)**
      E-ELAN extended ELAN
      Model re-parameterization
      Auxiliary head for training
      Compound model scaling
    **YOLOv8 (2023)**
      Anchor-free design
      C2f module (CSP + split)
      Decoupled head unified
      Ultralytics ecosystem
      Segmentation & pose support
    **YOLOv9 (2024)**
      PGI programmable gradient info
      GELAN architecture
      Reversible functions
      Information bottleneck solution
    **YOLOv10 (2024)**
      NMS-free training
      Consistent dual assignments
      Lightweight classification head
      Rank-guided allocation
    **YOLO11 (2024)**
      C3k2 cross-stage partial
      C2PSA spatial attention
      Improved small object head
      Reduced parameter count
    **YOLO12 (2025)**
      Area attention mechanism
      R-ELAN residual ELAN
      FlashAttention integration
      Efficient large-kernel attention
```
