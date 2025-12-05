# AccFlow (Accumulated Error Flow) 使用指南

## 概述

AccFlow 是一个新的自监督场景流估计方法，主要创新点包括：

1. **未来帧预测**：使用未来帧 (pc0, pc1, pc2, pc3, pc4) 而非历史帧
2. **时间 Embedding**：通过 one-hot 编码指定预测哪一帧对的 flow
3. **累积误差训练**：通过 KNN 插值传递预测误差，实现自监督学习

## 训练

### 自监督训练

```bash
python train.py \
    model=accflow \
    loss_fn=accflowLoss \
    num_frames=5 \
    ssl_label=dufo_label \
    "+add_seloss={chamfer_dis: 1.0, dynamic_chamfer_dis: 1.0, static_flow_loss: 1.0, cluster_based_pc0pc1: 1.0}" \
    train_data=/path/to/train \
    val_data=/path/to/val \
    batch_size=4 \
    epochs=15
```

### 不使用动态标签的简化训练

```bash
# 只用 chamfer distance，不需要 ssl_label
python train.py \
    model=accflow \
    loss_fn=accflowLoss \
    num_frames=5 \
    "+add_seloss={chamfer_dis: 1.0, dynamic_chamfer_dis: 0.0, static_flow_loss: 0.0, cluster_based_pc0pc1: 0.0}" \
    train_data=/path/to/train \
    val_data=/path/to/val
```

## 评估

```bash
python eval.py \
    checkpoint=/path/to/accflow.ckpt \
    data_mode=val \
    dataset_path=/path/to/dataset
```

## 模型参数

在 `conf/model/accflow.yaml` 中配置：

```yaml
name: accflow

target:
  _target_: src.models.AccFlow
  voxel_size: ${voxel_size}
  point_cloud_range: ${point_cloud_range}
  num_frames: ${num_frames}                    # 输入帧数，默认5
  planes: [16, 32, 64, 128, 256, 256, 128, 64, 32, 16]
  num_layer: [2, 2, 2, 2, 2, 2, 2, 2, 2]
  decay_factor: 0.4
  decoder_option: default
  knn_k: 3                                      # 插值的邻居数
  interpolation_method: knn                    # 插值方法
```

## 插值方法

AccFlow 支持多种插值方法用于累积误差训练中的 flow 传递：

| 方法 | 参数 | CUDA加速 | 说明 |
|------|------|----------|------|
| `knn` | `knn_k` | pytorch3d (可选) | K近邻加权平均，默认方法 |
| `three_nn` | - | torch-points-kernels (可选) | PointNet++ 风格的3近邻插值 |
| `rbf` | `sigma`, `max_neighbors` | 否 | 径向基函数(高斯核)插值 |
| `idw` | `power`, `k` | 否 | 反距离加权插值 |

### 使用不同插值方法

```bash
# 使用 KNN (默认)
python train.py model=accflow model.target.interpolation_method=knn model.target.knn_k=3

# 使用 PointNet++ 风格的 three_nn (需要 torch-points-kernels)
python train.py model=accflow model.target.interpolation_method=three_nn

# 使用 RBF 插值
python train.py model=accflow model.target.interpolation_method=rbf

# 使用 IDW 插值
python train.py model=accflow model.target.interpolation_method=idw
```

### 安装可选 CUDA 加速库

```bash
# pytorch3d (用于加速 KNN)
pip install pytorch3d

# torch-points-kernels (用于 three_nn)
pip install torch-points-kernels
```

如果这些库不可用，会自动回退到纯 PyTorch 实现。

## 训练流程详解

### 累积误差训练 (forward_accumulated_error)

1. **预测 F0**: 使用 pc0 和 pc1 预测 flow F0
2. **Warp pc0**: P1' = pc0 + F0 (将 pc0 warp 到 pc1 位置)
3. **预测 F1**: 使用 pc1 和 pc2 预测 flow F1
4. **KNN 插值**: 将 F1 从 pc1 插值到 P1' 的位置
5. **累积**: 继续 warp: P2' = P1' + F1', P3' = P2' + F2', ...
6. **损失计算**: 比较最终 warp 位置与真实 pc_final 的 Chamfer 距离

### 推理流程

推理时只计算 pc0 到 pc1 的 flow (time_idx=0)，与其他模型行为一致。

## 数据集要求

AccFlow 需要使用 `HDF5DatasetFutureFrames` 数据集类，它会自动读取未来帧：

- `num_frames=5` 时：读取 pc0, pc1, pc2, pc3, pc4
- `num_frames=3` 时：读取 pc0, pc1, pc2

## 与 DeltaFlow 的对比

| 特性 | DeltaFlow | AccFlow |
|-----|-----------|---------|
| 帧方向 | 历史帧 (pch3, pch2, pch1, pc0, pc1) | 未来帧 (pc0, pc1, pc2, pc3, pc4) |
| 时间 Embedding | 无 | 有 (one-hot) |
| 训练方式 | 监督 | 自监督（累积误差） |
| 损失函数 | deflowLoss | accflowLoss |
