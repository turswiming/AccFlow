# AccFlow: 基于累积误差传播的多帧自监督场景流估计

## 1. 概述

AccFlow (Accumulated Error Flow) 是一种新颖的自监督场景流估计方法，通过利用多帧点云序列中的累积误差传播来实现无需真值标注的训练。与传统的两帧方法不同，AccFlow 同时处理5帧连续点云，并通过时间嵌入机制灵活预测任意相邻帧对之间的场景流。

### 1.1 核心创新

1. **多帧未来预测架构**: 使用未来帧 (pc0→pc4) 而非历史帧，支持累积误差的前向传播
2. **双层时间嵌入机制**: 在编码器和解码器中同时注入时间信息，增强模型对不同时间步的区分能力
3. **累积误差自监督训练**: 通过误差传播构建跨多帧的自监督信号，无需真值标注

## 2. 网络架构

### 2.1 整体流程

```
输入: 5帧点云 [P₀, P₁, P₂, P₃, P₄] + 时间索引 t
                    ↓
        ┌─────────────────────────┐
        │    AccFlowEncoder       │
        │  (体素化 + 时间嵌入#1)   │
        └─────────────────────────┘
                    ↓
        ┌─────────────────────────┐
        │    MinkUNet Backbone    │
        │    (稀疏3D卷积网络)      │
        └─────────────────────────┘
                    ↓
        ┌─────────────────────────┐
        │  TimeAwarePointHead     │
        │  (解码器 + 时间嵌入#2)   │
        └─────────────────────────┘
                    ↓
输出: 场景流 F_{t→t+1} ∈ ℝ^{N×3}
```

### 2.2 AccFlowEncoder (编码器)

编码器负责将点云体素化并提取差分特征，同时注入第一层时间嵌入。

#### 2.2.1 时间嵌入 #1

采用 one-hot 编码表示目标时间步，通过可学习的线性投影映射到特征空间：

$$\mathbf{e}_t = \text{OneHot}(t) \in \mathbb{R}^{4}$$

$$\mathbf{f}_t^{enc} = W_{enc} \cdot \mathbf{e}_t \in \mathbb{R}^{C}$$

其中 $W_{enc} \in \mathbb{R}^{C \times 4}$ 是可学习的投影矩阵，$C=16$ 是特征通道数。

#### 2.2.2 差分特征计算

根据时间索引 $t$ 选择源帧和目标帧，计算稀疏体素差分：

$$\mathbf{V}_{src} = \text{Voxelize}(P_t)$$
$$\mathbf{V}_{tgt} = \text{Voxelize}(P_{t+1})$$
$$\mathbf{V}_{diff} = \mathbf{V}_{tgt} - \mathbf{V}_{src}$$

#### 2.2.3 时间嵌入融合

通过加法将时间特征融合到每个体素：

$$\mathbf{V}_{out}[i] = \mathbf{V}_{diff}[i] + \mathbf{f}_t^{enc}[b_i]$$

其中 $b_i$ 是第 $i$ 个体素所属的批次索引。

**设计理由**: 加法融合保持特征维度不变，使时间信息作为全局偏置影响所有体素特征。

### 2.3 MinkUNet Backbone

采用 Minkowski 稀疏卷积网络，配置为:
- 通道数: [16, 32, 64, 128, 256, 256, 128, 64, 32, 16]
- 每层卷积块数: [2, 2, 2, 2, 2, 2, 2, 2, 2]
- 总参数量: ~2.5M

### 2.4 TimeAwarePointHead (时间感知解码器)

解码器将体素级特征映射回点级场景流预测，并注入第二层时间嵌入。

#### 2.4.1 时间嵌入 #2

独立于编码器的时间嵌入，使用较小的隐藏维度：

$$\mathbf{e}_t = \text{OneHot}(t) \in \mathbb{R}^{4}$$

$$\mathbf{f}_t^{dec} = \text{ReLU}(W_{dec} \cdot \mathbf{e}_t) \in \mathbb{R}^{8}$$

#### 2.4.2 特征拼接

将体素特征、点特征和时间特征拼接：

$$\mathbf{h}_i = [\mathbf{v}_i \| \mathbf{p}_i \| \mathbf{f}_t^{dec}] \in \mathbb{R}^{40}$$

其中:
- $\mathbf{v}_i \in \mathbb{R}^{16}$: 从 backbone 输出索引的体素特征
- $\mathbf{p}_i \in \mathbb{R}^{16}$: 编码器提取的点特征
- $\mathbf{f}_t^{dec} \in \mathbb{R}^{8}$: 时间嵌入特征

**设计理由**: 拼接融合使MLP能够显式访问时间信息，与体素/点特征形成互补。

#### 2.4.3 流预测 MLP

$$\mathbf{F}_i = W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot \mathbf{h}_i))$$

其中 $W_1 \in \mathbb{R}^{32 \times 40}$, $W_2 \in \mathbb{R}^{3 \times 32}$。

## 3. 累积误差训练

### 3.1 核心思想

利用预测误差在时间维度上的累积，构建跨多帧的自监督信号。假设:
- 单帧预测误差较小
- 累积多帧后误差放大
- 自监督损失能够感知累积误差并反向传播梯度

### 3.2 训练流程

给定5帧点云 $\{P_0, P_1, P_2, P_3, P_4\}$:

**Step 1**: 预测初始流
$$\hat{F}_0 = \text{Model}(P_{0:4}, t=0)$$
$$\hat{P}_1 = P_0 + \hat{F}_0$$

**Step 2**: 迭代累积 (对于 $t = 1, 2, 3$)
$$\hat{F}_t = \text{Model}(P_{0:4}, t)$$
$$\tilde{F}_t = \text{KNN-Interpolate}(\hat{P}_t, P_t, \hat{F}_t)$$
$$\hat{P}_{t+1} = \hat{P}_t + \tilde{F}_t$$

**Step 3**: 计算累积流
$$F_{acc} = \hat{P}_4 - P_0$$

**Step 4**: 应用自监督损失
$$\mathcal{L} = \mathcal{L}_{self}(P_0, P_4, F_{acc})$$

### 3.3 KNN 插值

由于 $\hat{P}_t$ 与 $P_t$ 位置不完全对应，使用 KNN 插值获取累积位置处的流值：

$$\tilde{F}_t(\mathbf{x}) = \frac{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j \hat{F}_t(\mathbf{p}_j)}{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j}$$

其中 $\mathcal{N}_k(\mathbf{x})$ 是 $\mathbf{x}$ 在 $P_t$ 中的 $k$ 近邻，$w_j = 1/\|\mathbf{x} - \mathbf{p}_j\|$。

### 3.4 渐进式累积采样

为稳定训练，采用概率采样累积步数：

| 累积步数 | 含义 | 采样概率 |
|---------|------|---------|
| 1 | 仅 $F_0$ | 50% |
| 2 | $F_0 + F_1$ | 25% |
| 3 | $F_0 + F_1 + F_2$ | 12.5% |
| 4 | $F_0 + F_1 + F_2 + F_3$ | 12.5% |

## 4. 张量形状汇总

| 模块 | 变量 | 形状 | 说明 |
|------|------|------|------|
| **编码器** | `time_embed` | $[B, 4]$ | one-hot 编码 |
| | `time_feat_enc` | $[B, 16]$ | 投影后时间特征 |
| | `voxel_diff` | $[N_v, 16]$ | 差分体素特征 |
| | `voxel_out` | $[N_v, 16]$ | 融合后特征 |
| **Backbone** | `sparse_tensor` | $[B, 16, D, H, W]$ | 稀疏卷积输出 |
| **解码器** | `time_embed` | $[B, 4]$ | one-hot 编码 |
| | `time_feat_dec` | $[B, 8]$ | 投影后时间特征 |
| | `concat_feat` | $[N_p, 40]$ | 拼接特征 |
| | `flow` | $[N_p, 3]$ | 输出场景流 |

## 5. 实验验证

### 5.1 时间嵌入有效性

通过混淆矩阵实验验证时间嵌入的区分能力:

|  | GT $F_{0→1}$ | GT $F_{1→2}$ | GT $F_{2→3}$ | GT $F_{3→4}$ |
|--|-------------|-------------|-------------|-------------|
| Pred $t=0$ | **0.089** | 0.097 | 0.096 | 0.094 |
| Pred $t=1$ | 0.104 | **0.097** | 0.103 | 0.102 |
| Pred $t=2$ | 0.106 | 0.106 | **0.099** | 0.104 |
| Pred $t=3$ | 0.098 | 0.098 | 0.097 | **0.088** |

对角线元素均为该行最小值，证明模型能够正确区分不同时间步。

## 6. 使用方法

### 6.1 配置文件

```yaml
# conf/model/accflow.yaml
name: accflow
target:
  _target_: src.models.AccFlow
  num_frames: 5
  decoder_option: time_aware  # 启用时间感知解码器
  knn_k: 3
  accumulate_probs: [0.5, 0.25, 0.125, 0.125]
```

### 6.2 推理示例

```python
from src.models import AccFlow

model = AccFlow(decoder_option='time_aware')
model.load_from_checkpoint('checkpoint.ckpt')

# 预测 pc0 → pc1 的场景流
result = model(batch, time_idx=0)
flow_0_to_1 = result['flow']

# 预测 pc2 → pc3 的场景流
result = model(batch, time_idx=2)
flow_2_to_3 = result['flow']
```

## 7. 参考文献

- DeltaFlow: Multi-Frame Scene Flow Estimation with Temporal Changes
- SeFlow: Self-Supervised Scene Flow Estimation
- MinkowskiEngine: Sparse Tensor Networks
