# AccFlow: Multi-Frame Self-Supervised Scene Flow Estimation via Accumulated Error Propagation

## 1. Overview

AccFlow (Accumulated Error Flow) is a novel self-supervised scene flow estimation method that leverages accumulated error propagation across multi-frame point cloud sequences for training without ground truth annotations. Unlike conventional two-frame approaches, AccFlow processes five consecutive point cloud frames and employs a temporal embedding mechanism to flexibly predict scene flow between arbitrary adjacent frame pairs.

### 1.1 Key Contributions

1. **Multi-Frame Future Prediction Architecture**: Utilizes future frames (pc0→pc4) rather than historical frames, enabling forward propagation of accumulated errors
2. **Dual-Level Temporal Embedding**: Injects temporal information at both encoder and decoder stages, enhancing the model's discriminative capability across different time steps
3. **Accumulated Error Self-Supervision**: Constructs cross-frame self-supervised signals through error propagation, eliminating the need for ground truth annotations

## 2. Network Architecture

### 2.1 Overall Pipeline

```
Input: 5-frame point clouds [P₀, P₁, P₂, P₃, P₄] + time index t
                    ↓
        ┌─────────────────────────┐
        │    AccFlowEncoder       │
        │ (Voxelization + Time    │
        │   Embedding #1)         │
        └─────────────────────────┘
                    ↓
        ┌─────────────────────────┐
        │    MinkUNet Backbone    │
        │  (Sparse 3D ConvNet)    │
        └─────────────────────────┘
                    ↓
        ┌─────────────────────────┐
        │  TimeAwarePointHead     │
        │ (Decoder + Time         │
        │   Embedding #2)         │
        └─────────────────────────┘
                    ↓
Output: Scene Flow F_{t→t+1} ∈ ℝ^{N×3}
```

### 2.2 AccFlowEncoder

The encoder is responsible for voxelizing point clouds, extracting differential features, and injecting the first-level temporal embedding.

#### 2.2.1 Temporal Embedding #1

We adopt one-hot encoding to represent the target time step, mapping it to the feature space via a learnable linear projection:

$$\mathbf{e}_t = \text{OneHot}(t) \in \mathbb{R}^{4}$$

$$\mathbf{f}_t^{enc} = W_{enc} \cdot \mathbf{e}_t \in \mathbb{R}^{C}$$

where $W_{enc} \in \mathbb{R}^{C \times 4}$ is a learnable projection matrix and $C=16$ denotes the feature channel dimension.

#### 2.2.2 Differential Feature Computation

Based on the time index $t$, the source and target frames are selected to compute sparse voxel differences:

$$\mathbf{V}_{src} = \text{Voxelize}(P_t)$$
$$\mathbf{V}_{tgt} = \text{Voxelize}(P_{t+1})$$
$$\mathbf{V}_{diff} = \mathbf{V}_{tgt} - \mathbf{V}_{src}$$

#### 2.2.3 Temporal Embedding Fusion

The temporal features are fused into each voxel via element-wise addition:

$$\mathbf{V}_{out}[i] = \mathbf{V}_{diff}[i] + \mathbf{f}_t^{enc}[b_i]$$

where $b_i$ denotes the batch index of the $i$-th voxel.

**Design Rationale**: Additive fusion preserves the feature dimensionality, allowing temporal information to serve as a global bias that influences all voxel features uniformly.

### 2.3 MinkUNet Backbone

We employ a Minkowski sparse convolutional network with the following configuration:
- Channel dimensions: [16, 32, 64, 128, 256, 256, 128, 64, 32, 16]
- Convolutional blocks per layer: [2, 2, 2, 2, 2, 2, 2, 2, 2]
- Total parameters: ~2.5M

### 2.4 TimeAwarePointHead

The decoder maps voxel-level features back to point-level scene flow predictions while injecting the second-level temporal embedding.

#### 2.4.1 Temporal Embedding #2

Independent from the encoder's temporal embedding, utilizing a smaller hidden dimension:

$$\mathbf{e}_t = \text{OneHot}(t) \in \mathbb{R}^{4}$$

$$\mathbf{f}_t^{dec} = \text{ReLU}(W_{dec} \cdot \mathbf{e}_t) \in \mathbb{R}^{8}$$

#### 2.4.2 Feature Concatenation

Voxel features, point features, and temporal features are concatenated:

$$\mathbf{h}_i = [\mathbf{v}_i \| \mathbf{p}_i \| \mathbf{f}_t^{dec}] \in \mathbb{R}^{40}$$

where:
- $\mathbf{v}_i \in \mathbb{R}^{16}$: Voxel features indexed from backbone output
- $\mathbf{p}_i \in \mathbb{R}^{16}$: Point features extracted by the encoder
- $\mathbf{f}_t^{dec} \in \mathbb{R}^{8}$: Temporal embedding features

**Design Rationale**: Concatenation fusion enables the MLP to explicitly access temporal information, providing complementary signals alongside voxel and point features.

#### 2.4.3 Flow Prediction MLP

$$\mathbf{F}_i = W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot \mathbf{h}_i))$$

where $W_1 \in \mathbb{R}^{32 \times 40}$ and $W_2 \in \mathbb{R}^{3 \times 32}$.

## 3. Accumulated Error Training

### 3.1 Core Concept

The method leverages the accumulation of prediction errors along the temporal dimension to construct cross-frame self-supervised signals. The underlying assumptions are:
- Single-frame prediction errors are relatively small
- Errors amplify when accumulated across multiple frames
- Self-supervised losses can perceive accumulated errors and backpropagate gradients accordingly

### 3.2 Training Procedure

Given five point cloud frames $\{P_0, P_1, P_2, P_3, P_4\}$:

**Step 1**: Predict initial flow
$$\hat{F}_0 = \text{Model}(P_{0:4}, t=0)$$
$$\hat{P}_1 = P_0 + \hat{F}_0$$

**Step 2**: Iterative accumulation (for $t = 1, 2, 3$)
$$\hat{F}_t = \text{Model}(P_{0:4}, t)$$
$$\tilde{F}_t = \text{KNN-Interpolate}(\hat{P}_t, P_t, \hat{F}_t)$$
$$\hat{P}_{t+1} = \hat{P}_t + \tilde{F}_t$$

**Step 3**: Compute accumulated flow
$$F_{acc} = \hat{P}_4 - P_0$$

**Step 4**: Apply self-supervised loss
$$\mathcal{L} = \mathcal{L}_{self}(P_0, P_4, F_{acc})$$

### 3.3 KNN Interpolation

Since $\hat{P}_t$ and $P_t$ do not have exact point correspondences, KNN interpolation is employed to obtain flow values at accumulated positions:

$$\tilde{F}_t(\mathbf{x}) = \frac{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j \hat{F}_t(\mathbf{p}_j)}{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j}$$

where $\mathcal{N}_k(\mathbf{x})$ denotes the $k$-nearest neighbors of $\mathbf{x}$ in $P_t$, and $w_j = 1/\|\mathbf{x} - \mathbf{p}_j\|$.

### 3.4 Progressive Accumulation Sampling

To stabilize training, we employ probabilistic sampling of accumulation steps:

| Accumulation Steps | Semantics | Sampling Probability |
|-------------------|-----------|---------------------|
| 1 | $F_0$ only | 50% |
| 2 | $F_0 + F_1$ | 25% |
| 3 | $F_0 + F_1 + F_2$ | 12.5% |
| 4 | $F_0 + F_1 + F_2 + F_3$ | 12.5% |

## 4. Tensor Shape Summary

| Module | Variable | Shape | Description |
|--------|----------|-------|-------------|
| **Encoder** | `time_embed` | $[B, 4]$ | One-hot encoding |
| | `time_feat_enc` | $[B, 16]$ | Projected temporal features |
| | `voxel_diff` | $[N_v, 16]$ | Differential voxel features |
| | `voxel_out` | $[N_v, 16]$ | Fused features |
| **Backbone** | `sparse_tensor` | $[B, 16, D, H, W]$ | Sparse convolution output |
| **Decoder** | `time_embed` | $[B, 4]$ | One-hot encoding |
| | `time_feat_dec` | $[B, 8]$ | Projected temporal features |
| | `concat_feat` | $[N_p, 40]$ | Concatenated features |
| | `flow` | $[N_p, 3]$ | Output scene flow |

## 5. Experimental Validation

### 5.1 Temporal Embedding Effectiveness

The discriminative capability of temporal embedding is validated through a confusion matrix experiment:

|  | GT $F_{0→1}$ | GT $F_{1→2}$ | GT $F_{2→3}$ | GT $F_{3→4}$ |
|--|-------------|-------------|-------------|-------------|
| Pred $t=0$ | **0.089** | 0.097 | 0.096 | 0.094 |
| Pred $t=1$ | 0.104 | **0.097** | 0.103 | 0.102 |
| Pred $t=2$ | 0.106 | 0.106 | **0.099** | 0.104 |
| Pred $t=3$ | 0.098 | 0.098 | 0.097 | **0.088** |

Diagonal elements consistently exhibit the minimum EPE values within their respective rows, demonstrating that the model effectively distinguishes between different time steps.

## 6. Usage

### 6.1 Configuration

```yaml
# conf/model/accflow.yaml
name: accflow
target:
  _target_: src.models.AccFlow
  num_frames: 5
  decoder_option: time_aware  # Enable time-aware decoder
  knn_k: 3
  accumulate_probs: [0.5, 0.25, 0.125, 0.125]
```

### 6.2 Inference Example

```python
from src.models import AccFlow

model = AccFlow(decoder_option='time_aware')
model.load_from_checkpoint('checkpoint.ckpt')

# Predict scene flow from pc0 to pc1
result = model(batch, time_idx=0)
flow_0_to_1 = result['flow']

# Predict scene flow from pc2 to pc3
result = model(batch, time_idx=2)
flow_2_to_3 = result['flow']
```

## 7. References

- DeltaFlow: Multi-Frame Scene Flow Estimation with Temporal Changes
- SeFlow: Self-Supervised Scene Flow Estimation
- MinkowskiEngine: Sparse Tensor Networks
