"""
# Created: 2025-12-05
# AccumulateErrorFlow - A new self-supervised scene flow method
# Based on DeltaFlow architecture with accumulated error training

# Key innovations:
# 1. Uses future frames (pc0, pc1, pc2, pc3, pc4) instead of history frames
# 2. Adds time embedding (one-hot) to specify which frame pair to predict flow for
# 3. Accumulates prediction errors through KNN interpolation for self-supervised training
"""

import torch
import torch.nn as nn
import dztimer
import spconv.pytorch as spconv
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

from .basic import cal_pose0to1
from .basic.encoder import DynamicVoxelizer, DynamicPillarFeatureNet
from .basic.sparse_encoder import MinkUNet
from .basic.decoder import SparseGRUHead
from .basic.flow4d_module import Point_head

# Try to import CUDA-accelerated interpolation methods
try:
    from pytorch3d.ops import knn_points, knn_gather
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    print("[AccFlow] pytorch3d not available, using pure PyTorch KNN")

try:
    # PointNet++ style three_interpolate from Open3D or torch-points-kernels
    from torch_points_kernels import three_interpolate, three_nn
    HAS_POINTNET_OPS = True
except ImportError:
    HAS_POINTNET_OPS = False


def knn_interpolate_flow(query_points, ref_points, ref_flow, k=3):
    """
    Interpolate flow from reference points to query points using KNN.
    Uses pytorch3d CUDA implementation if available, otherwise pure PyTorch.

    Args:
        query_points: [N, 3] points to interpolate flow to
        ref_points: [M, 3] reference points with known flow
        ref_flow: [M, 3] flow vectors at reference points
        k: number of nearest neighbors

    Returns:
        interpolated_flow: [N, 3] interpolated flow at query points
    """
    if HAS_PYTORCH3D and query_points.shape[0] > 100:
        # Use pytorch3d's CUDA KNN (faster for large point clouds)
        # pytorch3d expects [B, N, 3] format
        query_pts = query_points.unsqueeze(0)  # [1, N, 3]
        ref_pts = ref_points.unsqueeze(0)  # [1, M, 3]
        ref_f = ref_flow.unsqueeze(0)  # [1, M, 3]

        # KNN query
        knn_result = knn_points(query_pts, ref_pts, K=k, return_sorted=False)
        knn_dist = knn_result.dists  # [1, N, k]
        knn_idx = knn_result.idx  # [1, N, k]

        # Gather flow values
        knn_flow = knn_gather(ref_f, knn_idx)  # [1, N, k, 3]

        # Inverse distance weighting
        weights = 1.0 / (knn_dist.sqrt() + 1e-8)  # [1, N, k]
        weights = weights / weights.sum(dim=2, keepdim=True)

        # Weighted sum
        interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=2)  # [1, N, 3]
        return interpolated_flow.squeeze(0)  # [N, 3]
    else:
        # Pure PyTorch fallback
        dist = torch.cdist(query_points, ref_points)
        knn_dist, knn_idx = dist.topk(k, dim=1, largest=False)
        weights = 1.0 / (knn_dist + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        knn_flow = ref_flow[knn_idx]
        interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=1)
        return interpolated_flow


def three_nn_interpolate_flow(query_points, ref_points, ref_flow):
    """
    PointNet++ style three nearest neighbor interpolation with CUDA.
    This is the interpolation used in PointNet++ feature propagation.

    Uses inverse distance weighting with 3 nearest neighbors.
    CUDA implementation from torch-points-kernels.

    Args:
        query_points: [N, 3] points to interpolate flow to
        ref_points: [M, 3] reference points with known flow
        ref_flow: [M, 3] flow vectors at reference points

    Returns:
        interpolated_flow: [N, 3] interpolated flow at query points
    """
    if HAS_POINTNET_OPS:
        # torch-points-kernels expects [B, N, 3] format
        query_pts = query_points.unsqueeze(0).contiguous()  # [1, N, 3]
        ref_pts = ref_points.unsqueeze(0).contiguous()  # [1, M, 3]
        ref_f = ref_flow.unsqueeze(0).transpose(1, 2).contiguous()  # [1, 3, M]

        # Three nearest neighbors
        dist, idx = three_nn(query_pts, ref_pts)  # dist: [1, N, 3], idx: [1, N, 3]

        # Compute weights (inverse distance)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = dist_recip.sum(dim=2, keepdim=True)
        weight = dist_recip / norm  # [1, N, 3]

        # Interpolate
        interpolated = three_interpolate(ref_f, idx, weight)  # [1, 3, N]
        return interpolated.squeeze(0).transpose(0, 1)  # [N, 3]
    else:
        # Fallback to KNN with k=3
        return knn_interpolate_flow(query_points, ref_points, ref_flow, k=3)


def rbf_interpolate_flow(query_points, ref_points, ref_flow, sigma=0.5, max_neighbors=16):
    """
    Radial Basis Function (RBF) interpolation for flow.
    Uses Gaussian kernel weighting.

    Args:
        query_points: [N, 3] points to interpolate flow to
        ref_points: [M, 3] reference points with known flow
        ref_flow: [M, 3] flow vectors at reference points
        sigma: RBF kernel width
        max_neighbors: maximum number of neighbors to consider (for efficiency)

    Returns:
        interpolated_flow: [N, 3] interpolated flow at query points
    """
    # Find nearest neighbors first (for efficiency)
    dist = torch.cdist(query_points, ref_points)

    if max_neighbors < ref_points.shape[0]:
        # Only use top-k neighbors
        knn_dist, knn_idx = dist.topk(max_neighbors, dim=1, largest=False)

        # Gaussian RBF weights
        weights = torch.exp(-knn_dist ** 2 / (2 * sigma ** 2))  # [N, k]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Gather and weight
        knn_flow = ref_flow[knn_idx]  # [N, k, 3]
        interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=1)
    else:
        # Use all points
        weights = torch.exp(-dist ** 2 / (2 * sigma ** 2))  # [N, M]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        interpolated_flow = weights @ ref_flow  # [N, 3]

    return interpolated_flow


def idw_interpolate_flow(query_points, ref_points, ref_flow, power=2, k=8):
    """
    Inverse Distance Weighting (IDW) interpolation.
    Weight = 1 / distance^power

    Args:
        query_points: [N, 3] points to interpolate flow to
        ref_points: [M, 3] reference points with known flow
        ref_flow: [M, 3] flow vectors at reference points
        power: distance power (higher = more local)
        k: number of nearest neighbors

    Returns:
        interpolated_flow: [N, 3] interpolated flow at query points
    """
    dist = torch.cdist(query_points, ref_points)
    knn_dist, knn_idx = dist.topk(k, dim=1, largest=False)

    # IDW weights
    weights = 1.0 / (knn_dist ** power + 1e-8)  # [N, k]
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Gather and weight
    knn_flow = ref_flow[knn_idx]  # [N, k, 3]
    interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=1)

    return interpolated_flow


# Interpolation method registry
INTERPOLATION_METHODS = {
    'knn': knn_interpolate_flow,
    'three_nn': three_nn_interpolate_flow,
    'rbf': rbf_interpolate_flow,
    'idw': idw_interpolate_flow,
}


def interpolate_flow(query_points, ref_points, ref_flow, method='knn', **kwargs):
    """
    Unified interface for flow interpolation.

    Args:
        query_points: [N, 3] points to interpolate flow to
        ref_points: [M, 3] reference points with known flow
        ref_flow: [M, 3] flow vectors at reference points
        method: interpolation method ('knn', 'three_nn', 'rbf', 'idw')
        **kwargs: method-specific parameters

    Returns:
        interpolated_flow: [N, 3] interpolated flow at query points
    """
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"Unknown interpolation method: {method}. Available: {list(INTERPOLATION_METHODS.keys())}")

    return INTERPOLATION_METHODS[method](query_points, ref_points, ref_flow, **kwargs)


def wrap_batch_pcs_future(batch, num_frames=5):
    """
    Wrap batch point clouds for AccFlow model - uses FUTURE frames instead of history.

    Frame layout: pc0, pc1, pc2, pc3, pc4 (t, t+1, t+2, t+3, t+4)

    All point clouds are transformed to pc0's coordinate system for consistency.
    """
    batch_sizes = len(batch["pose0"])

    # Initialize result dictionary
    pcs_dict = {f'pc{i}s': [] for i in range(num_frames)}
    pcs_dict['pose_flows'] = {i: [] for i in range(num_frames - 1)}  # pose_flow for each transition

    for batch_id in range(batch_sizes):
        # Reference pose is pc0's pose - all other frames transform to pc0's coordinate
        ref_pose = batch["pose0"][batch_id]

        with torch.no_grad():
            # pc0 stays in its own coordinate (no transformation needed for the base)
            pcs_dict['pc0s'].append(batch["pc0"][batch_id])

            # Process each future frame
            for i in range(1, num_frames):
                pc_key = f'pc{i}'
                pose_key = f'pose{i}'

                if pc_key in batch and pose_key in batch:
                    # Transform pc_i to pc0's coordinate system
                    pose_i_to_0 = cal_pose0to1(batch[pose_key][batch_id], ref_pose)
                    pc_i = batch[pc_key][batch_id]
                    pc_i_transformed = pc_i @ pose_i_to_0[:3, :3].T + pose_i_to_0[:3, 3]
                    pcs_dict[f'pc{i}s'].append(pc_i_transformed)

            # Compute pose flows for each consecutive pair (in pc0's coordinate system)
            # pose_flow_i represents ego motion from frame i to frame i+1, applied to points in pc0's coords
            for i in range(num_frames - 1):
                pose_key_i = f'pose{i}'
                pose_key_i1 = f'pose{i+1}'

                if pose_key_i in batch and pose_key_i1 in batch:
                    # Pose from frame i to frame i+1
                    pose_i_to_i1 = cal_pose0to1(batch[pose_key_i][batch_id], batch[pose_key_i1][batch_id])

                    # For pc0, compute the pose flow (displacement due to ego motion)
                    pc_i_key = f'pc{i}'
                    if pc_i_key in batch:
                        pc_i = batch[pc_i_key][batch_id]
                        pc_i_transformed = pc_i @ pose_i_to_i1[:3, :3].T + pose_i_to_i1[:3, 3]
                        pose_flow = pc_i_transformed - pc_i
                        pcs_dict['pose_flows'][i].append(pose_flow)

    # Stack all lists into tensors
    for i in range(num_frames):
        key = f'pc{i}s'
        if pcs_dict[key]:
            pcs_dict[key] = torch.stack(pcs_dict[key], dim=0)

    # Stack pose flows
    for i in range(num_frames - 1):
        if pcs_dict['pose_flows'][i]:
            pcs_dict['pose_flows'][i] = pcs_dict['pose_flows'][i]  # Keep as list for per-point handling

    return pcs_dict


class AccFlowEncoder(nn.Module):
    """Modified SparseVoxelNet that accepts time embedding"""

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, num_frames: int = 5, decay_factor=1.0, timer=None) -> None:
        super().__init__()

        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels,),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')

        self.voxel_spatial_shape = pseudo_image_dims
        self.num_feature = feat_channels
        self.num_frames = num_frames
        self.decay_factor = decay_factor

        # Time embedding: one-hot encoding for which frame pair to predict
        # For 5 frames, we have 4 possible flow predictions: 0->1, 1->2, 2->3, 3->4
        self.time_embed_dim = num_frames - 1  # 4 for 5 frames
        # Project time embedding to feature dimension
        self.time_proj = nn.Linear(self.time_embed_dim, feat_channels)

        if timer is None:
            self.timer = dztimer.Timing()
            self.timer.start("Total")
        else:
            self.timer = timer

    def process_batch(self, voxel_info_list, if_return_point_feats=False):
        voxel_feats_list_batch = []
        voxel_coors_list_batch = []
        point_feats_lst = []

        for batch_index, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)
            if if_return_point_feats:
                point_feats_lst.append(point_feats)
            batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
            voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)
            voxel_feats_list_batch.append(voxel_feats)
            voxel_coors_list_batch.append(voxel_coors_batch)

        voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
        coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32)

        if if_return_point_feats:
            return voxel_feats_sp, coors_batch_sp, point_feats_lst

        return voxel_feats_sp, coors_batch_sp

    def forward(self, pcs_dict, time_idx: int = 0) -> torch.Tensor:
        """
        Forward pass with time embedding.

        Args:
            pcs_dict: Dictionary containing point clouds pc0s, pc1s, pc2s, ...
            time_idx: Which frame pair to predict (0 means pc0->pc1, 1 means pc1->pc2, etc.)

        Returns:
            Dictionary with sparse features and voxel info
        """
        bz_ = pcs_dict['pc0s'].shape[0]
        device = pcs_dict['pc0s'].device

        # Create one-hot time embedding
        time_embed = torch.zeros(bz_, self.time_embed_dim, device=device)
        time_embed[:, time_idx] = 1.0
        time_feat = self.time_proj(time_embed)  # [B, feat_channels]

        # Get source and target frame based on time_idx
        src_key = f'pc{time_idx}s'
        tgt_key = f'pc{time_idx + 1}s'

        src_pc = pcs_dict[src_key]
        tgt_pc = pcs_dict[tgt_key]

        # Process target frame
        tgt_voxel_info_list = self.voxelizer(tgt_pc)
        tgt_voxel_feats_sp, tgt_coors_batch_sp = self.process_batch(tgt_voxel_info_list)
        tgt_num_voxels = tgt_voxel_feats_sp.shape[0]

        sparse_max_size = [bz_, *self.voxel_spatial_shape, self.num_feature]
        sparse_tgt = torch.sparse_coo_tensor(tgt_coors_batch_sp.t(), tgt_voxel_feats_sp, size=sparse_max_size)

        # Process source frame
        src_voxel_info_list = self.voxelizer(src_pc)
        src_voxel_feats_sp, src_coors_batch_sp, src_point_feats_lst = self.process_batch(
            src_voxel_info_list, if_return_point_feats=True
        )
        src_num_voxels = src_voxel_feats_sp.shape[0]

        sparse_src = torch.sparse_coo_tensor(src_coors_batch_sp.t(), src_voxel_feats_sp, size=sparse_max_size)

        # Compute delta (difference) features
        sparse_diff = sparse_tgt - sparse_src

        # Add time embedding to the features
        self.timer[2].start("D_Delta_Sparse")
        features = sparse_diff.coalesce().values()
        indices = sparse_diff.coalesce().indices().t().to(dtype=torch.int32)

        # Add time embedding: for each voxel, add the corresponding batch's time feature
        batch_indices_for_time = indices[:, 0].long()
        time_feat_expanded = time_feat[batch_indices_for_time]  # [num_voxels, feat_channels]
        features = features + time_feat_expanded

        all_pcdiff_sparse = spconv.SparseConvTensor(
            features.contiguous(), indices.contiguous(),
            self.voxel_spatial_shape, bz_
        )
        self.timer[2].stop()

        output = {
            'delta_sparse': all_pcdiff_sparse,
            'src_3dvoxel_infos_lst': src_voxel_info_list,
            'src_point_feats_lst': src_point_feats_lst,
            'src_num_voxels': src_num_voxels,
            'tgt_3dvoxel_infos_lst': tgt_voxel_info_list,
            'tgt_num_voxels': tgt_num_voxels,
            'd_num_voxels': indices.shape[0],
            'time_idx': time_idx,
        }
        return output


class AccFlow(nn.Module):
    """
    AccumulateErrorFlow model.

    Key differences from DeltaFlow:
    1. Uses future frames instead of history frames
    2. Has time embedding to specify which frame pair to predict
    3. Training uses accumulated error propagation for self-supervised learning
    """

    def __init__(self, voxel_size=[0.2, 0.2, 0.2],
                 point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size=[512, 512, 32],
                 num_frames=5,
                 planes=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
                 num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 decay_factor=1.0,
                 decoder_option="default",
                 knn_k=3,  # k for KNN interpolation
                 interpolation_method="knn",  # interpolation method: knn, three_nn, rbf, idw
                 accumulate_probs=None,  # Probability for each accumulation step [1, 2, 3, 4], None = always full
                 ):
        super().__init__()
        point_output_ch = planes[0]
        voxel_output_ch = planes[-1]
        self.timer = dztimer.Timing()
        self.num_frames = num_frames
        self.knn_k = knn_k
        self.interpolation_method = interpolation_method

        # Accumulation step probability sampling
        # accumulate_probs[i] = probability of accumulating exactly (i+1) steps
        # e.g., [0.5, 0.25, 0.125, 0.125] means:
        #   50% chance: 1 step (only pc0->pc1)
        #   25% chance: 2 steps (pc0->pc1->pc2)
        #   12.5% chance: 3 steps (pc0->pc1->pc2->pc3)
        #   12.5% chance: 4 steps (pc0->pc1->pc2->pc3->pc4)
        if accumulate_probs is not None:
            self.accumulate_probs = accumulate_probs
        else:
            self.accumulate_probs = None  # None means always use full accumulation

        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print('[LOG] AccFlow Param detail: voxel_size = {}, pseudo_dims = {}, num_frames={}'.format(
                voxel_size, grid_feature_size, num_frames))
            print('[LOG] Model detail: planes = {}, time decay = {}, decoder = {}, knn_k = {}, interp = {}'.format(
                planes, decay_factor, decoder_option, knn_k, interpolation_method))
            print('[LOG] Accumulate probs: {}'.format(self.accumulate_probs))

        self.pc2voxel = AccFlowEncoder(
            voxel_size=voxel_size,
            pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2]],
            point_cloud_range=point_cloud_range,
            feat_channels=point_output_ch,
            num_frames=num_frames,
            decay_factor=decay_factor,
            timer=self.timer[1]
        )
        self.backbone = MinkUNet(planes, num_layer)

        if decoder_option == "deflow":
            self.flowdecoder = SparseGRUHead(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch, num_iters=1)
        else:
            self.flowdecoder = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)

        self.voxel_spatial_shape = grid_feature_size
        self.cnt = 0
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model."):]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward_single(self, pcs_dict, time_idx: int = 0):
        """
        Forward pass for a single time index.

        Args:
            pcs_dict: Dictionary with transformed point clouds
            time_idx: Which frame pair to predict (0=pc0->pc1, 1=pc1->pc2, etc.)

        Returns:
            Dictionary with flow predictions and auxiliary info
        """
        self.timer[1].start("3D Sparse Voxel")
        sparse_dict = self.pc2voxel(pcs_dict, time_idx=time_idx)
        self.timer[1].stop()

        self.timer[3].start("3D Network")
        backbone_res = self.backbone(sparse_dict['delta_sparse'])
        src_3dvoxel_infos_lst = sparse_dict['src_3dvoxel_infos_lst']
        self.timer[3].stop()

        self.timer[4].start("Flow Decoder")
        flows = self.flowdecoder(backbone_res, src_3dvoxel_infos_lst, sparse_dict['src_point_feats_lst'])
        self.timer[4].stop()

        return {
            "flow": flows,
            "src_valid_point_idxes": [e["point_idxes"] for e in src_3dvoxel_infos_lst],
            "src_points_lst": [e["points"] for e in src_3dvoxel_infos_lst],
            "tgt_valid_point_idxes": [e["point_idxes"] for e in sparse_dict['tgt_3dvoxel_infos_lst']],
            "tgt_points_lst": [e["points"] for e in sparse_dict['tgt_3dvoxel_infos_lst']],
            "d_num_voxels": sparse_dict['d_num_voxels'],
            "time_idx": time_idx,
        }

    def forward(self, batch, time_idx: int = 0, training_mode: bool = False):
        """
        Main forward pass.

        Args:
            batch: Batch from dataloader with pc0, pc1, pc2, ... and poses
            time_idx: Which frame pair to predict (default 0 for pc0->pc1)
            training_mode: If True, use accumulated error training flow

        Returns:
            Model output dictionary
        """
        self.cnt += 1
        self.timer[0].start("Data Preprocess")
        pcs_dict = wrap_batch_pcs_future(batch, num_frames=self.num_frames)
        self.timer[0].stop()

        if training_mode and self.num_frames > 2:
            # Training with accumulated error
            return self.forward_accumulated_error(batch, pcs_dict)
        else:
            # Normal inference: predict pc0 -> pc1 flow
            result = self.forward_single(pcs_dict, time_idx=time_idx)

            # Add pose flow for the specified time index
            result['pose_flow'] = pcs_dict['pose_flows'][time_idx]

            # For compatibility, also add pc0/pc1 specific keys
            result["pc0_valid_point_idxes"] = result["src_valid_point_idxes"]
            result["pc0_points_lst"] = result["src_points_lst"]
            result["pc1_valid_point_idxes"] = result["tgt_valid_point_idxes"]
            result["pc1_points_lst"] = result["tgt_points_lst"]
            result["d_num_voxels"] = [result["d_num_voxels"]]

            return result

    def _sample_num_accumulate_steps(self):
        """
        Sample the number of accumulation steps based on configured probabilities.

        Returns:
            int: Number of accumulation steps to use (1 to num_frames-1)
        """
        if self.accumulate_probs is None:
            # No probability configured, use full accumulation
            return self.num_frames - 1

        import random
        # accumulate_probs[i] = probability of using (i+1) steps
        # e.g., [0.5, 0.25, 0.125, 0.125] for 1, 2, 3, 4 steps
        r = random.random()
        cumsum = 0.0
        for i, prob in enumerate(self.accumulate_probs):
            cumsum += prob
            if r < cumsum:
                return i + 1
        # Fallback to last option
        return len(self.accumulate_probs)

    def forward_accumulated_error(self, batch, pcs_dict):
        """
        Training forward pass with accumulated error propagation.

        This implements the self-supervised training procedure:
        1. Predict flow F0 from pc0 -> pc1
        2. Create warped point cloud P1' = P0 + F0
        3. Use KNN to interpolate F1 from P1 to P1'
        4. Continue accumulating: P2' = P1' + F1', P3' = P2' + F2', etc.
        5. Final accumulated flow is used for self-supervised loss

        With accumulate_probs configured, the number of accumulation steps is sampled:
        - 1 step: only F0 (pc0->pc1)
        - 2 steps: F0 + F1 (pc0->pc1->pc2)
        - etc.

        Returns:
            Dictionary with accumulated flow for self-supervised training
        """
        batch_size = pcs_dict['pc0s'].shape[0]
        device = pcs_dict['pc0s'].device

        # Sample how many accumulation steps to use for this forward pass
        num_acc_steps = self._sample_num_accumulate_steps()
        # Clamp to valid range
        max_steps = self.num_frames - 1
        num_acc_steps = min(num_acc_steps, max_steps)

        # Results for each batch
        all_accumulated_flows = []
        all_valid_idxes = []
        all_pc0_points = []

        for batch_id in range(batch_size):
            # Get individual point clouds for this batch
            pc0 = pcs_dict['pc0s'][batch_id]  # [N0, 3]

            # Remove NaN padding
            valid_mask = ~torch.isnan(pc0[:, 0])
            pc0 = pc0[valid_mask]

            # Predict F0: pc0 -> pc1
            single_pcs_dict = {
                f'pc{i}s': pcs_dict[f'pc{i}s'][batch_id:batch_id+1]
                for i in range(self.num_frames)
                if f'pc{i}s' in pcs_dict
            }

            result_0 = self.forward_single(single_pcs_dict, time_idx=0)
            flow_0 = result_0['flow'][0]  # [N0_valid, 3]
            valid_idx_0 = result_0['src_valid_point_idxes'][0]

            # Initialize accumulated position and valid indices
            # P1' = P0 + F0 (warped position)
            pc0_valid = pc0[valid_idx_0]  # Points that went through the network
            accumulated_pos = pc0_valid + flow_0  # P1'
            accumulated_flow = flow_0.clone()  # Total displacement from pc0

            # Iterate through remaining frame pairs (up to num_acc_steps)
            # num_acc_steps=1 means only F0, so loop doesn't run
            # num_acc_steps=2 means F0+F1, so t=1 only
            for t in range(1, num_acc_steps):
                # Get real pc at time t+1
                pc_t1 = pcs_dict[f'pc{t+1}s'][batch_id]
                valid_mask_t1 = ~torch.isnan(pc_t1[:, 0])
                pc_t1 = pc_t1[valid_mask_t1]

                # Get real pc at time t
                pc_t = pcs_dict[f'pc{t}s'][batch_id]
                valid_mask_t = ~torch.isnan(pc_t[:, 0])
                pc_t = pc_t[valid_mask_t]

                # Predict F_t: pc_t -> pc_{t+1}
                result_t = self.forward_single(single_pcs_dict, time_idx=t)
                flow_t = result_t['flow'][0]  # [N_t_valid, 3]
                valid_idx_t = result_t['src_valid_point_idxes'][0]

                # Get valid points from pc_t
                pc_t_valid = pc_t[valid_idx_t]

                # Use interpolation to get F_t from pc_t to accumulated_pos (P_t')
                # accumulated_pos is where our points ended up after previous accumulation
                # We need to find corresponding flow values at these positions
                if accumulated_pos.shape[0] > 0 and pc_t_valid.shape[0] > 0:
                    # Use configured interpolation method
                    if self.interpolation_method == 'knn':
                        interpolated_flow = interpolate_flow(
                            accumulated_pos, pc_t_valid, flow_t,
                            method='knn', k=self.knn_k
                        )
                    elif self.interpolation_method == 'three_nn':
                        interpolated_flow = interpolate_flow(
                            accumulated_pos, pc_t_valid, flow_t,
                            method='three_nn'
                        )
                    elif self.interpolation_method == 'rbf':
                        interpolated_flow = interpolate_flow(
                            accumulated_pos, pc_t_valid, flow_t,
                            method='rbf', sigma=0.5, max_neighbors=self.knn_k * 4
                        )
                    elif self.interpolation_method == 'idw':
                        interpolated_flow = interpolate_flow(
                            accumulated_pos, pc_t_valid, flow_t,
                            method='idw', power=2, k=self.knn_k
                        )
                    else:
                        # Default to KNN
                        interpolated_flow = interpolate_flow(
                            accumulated_pos, pc_t_valid, flow_t,
                            method='knn', k=self.knn_k
                        )

                    # Update accumulated position and flow
                    accumulated_pos = accumulated_pos + interpolated_flow  # P_{t+1}'
                    accumulated_flow = accumulated_flow + interpolated_flow

            # Store results for this batch
            all_accumulated_flows.append(accumulated_flow)
            all_valid_idxes.append(valid_idx_0)
            all_pc0_points.append(pc0_valid)

        # Package results
        # The accumulated flow represents the total displacement from pc0 to the final warped position
        # This can be compared against the real pc_final using chamfer distance

        # Get pose flow for pc0 -> pc1 (used for reference)
        pose_flows = pcs_dict['pose_flows'][0]

        result = {
            "flow": all_accumulated_flows,  # Accumulated flow from pc0
            "pose_flow": pose_flows,
            "pc0_valid_point_idxes": all_valid_idxes,
            "pc0_points_lst": all_pc0_points,
            "accumulated_target_frame": num_acc_steps,  # Index of the target frame (1=pc1, 2=pc2, etc.)
            "d_num_voxels": [0],  # Placeholder
        }

        return result

    def forward_turbo(self, batch, num_windows=None):
        """
        Turbo inference mode: average predictions from multiple sliding windows.

        核心思想：
        - 普通推理：用 [pc0, pc1, pc2, pc3, pc4] 预测 pc0→pc1 的 flow
        - Turbo 推理：滑动窗口，多次预测同一个 pc0→pc1，然后取平均

        示例 (5帧模型，数据集提供8帧)：
        - 窗口1: [pc-3, pc-2, pc-1, pc0, pc1] → time_idx=3 预测 pc0→pc1
        - 窗口2: [pc-2, pc-1, pc0, pc1, pc2] → time_idx=2 预测 pc0→pc1
        - 窗口3: [pc-1, pc0, pc1, pc2, pc3] → time_idx=1 预测 pc0→pc1
        - 窗口4: [pc0, pc1, pc2, pc3, pc4] → time_idx=0 预测 pc0→pc1

        最终 flow = mean(4个预测)

        要求：数据集需要提供额外的历史帧 (pch1, pch2, pch3, ...)

        Args:
            batch: Batch from dataloader, 需要包含 pch1, pch2, pch3 等历史帧
            num_windows: 滑动窗口数量 (默认 = num_frames - 1)

        Returns:
            Averaged flow prediction
        """
        if num_windows is None:
            num_windows = self.num_frames - 1

        batch_size = len(batch["pose0"])
        _device = batch["pc0"].device  # Reserved for future use

        all_flow_predictions = []

        # 检查是否有足够的历史帧来做 turbo 推理
        has_history = 'pch1' in batch

        if not has_history:
            # 没有历史帧，退化为普通推理
            print("[Turbo] Warning: No history frames available, falling back to normal inference")
            return self.forward(batch, time_idx=0)

        # 窗口滑动：从最早的窗口到最新的窗口
        for window_offset in range(num_windows):
            # window_offset=0: 使用 [pc0, pc1, pc2, pc3, pc4], time_idx=0
            # window_offset=1: 使用 [pch1, pc0, pc1, pc2, pc3], time_idx=1
            # window_offset=2: 使用 [pch2, pch1, pc0, pc1, pc2], time_idx=2
            # ...

            if window_offset == 0:
                # 最新窗口：直接用未来帧
                pcs_dict = wrap_batch_pcs_future(batch, num_frames=self.num_frames)
                result = self.forward_single(pcs_dict, time_idx=0)
                all_flow_predictions.append(result['flow'])
            else:
                # 需要构建包含历史帧的 batch
                # 检查是否有足够的历史帧
                history_key = f'pch{window_offset}'
                if history_key not in batch:
                    continue

                # 构建滑动窗口的 batch
                shifted_batch = self._build_shifted_batch(batch, window_offset)
                if shifted_batch is None:
                    continue

                pcs_dict = wrap_batch_pcs_future(shifted_batch, num_frames=self.num_frames)
                result = self.forward_single(pcs_dict, time_idx=window_offset)

                # 需要将预测的 flow 变换回 pc0 的坐标系
                # 因为不同窗口的参考坐标系不同
                transformed_flow = self._transform_flow_to_pc0(
                    result['flow'], batch, window_offset
                )
                if transformed_flow is not None:
                    all_flow_predictions.append(transformed_flow)

        # 平均所有预测
        if len(all_flow_predictions) == 0:
            return self.forward(batch, time_idx=0)
        elif len(all_flow_predictions) == 1:
            final_flow = all_flow_predictions[0]
        else:
            # 对每个 batch 分别平均
            final_flow = []
            for b in range(batch_size):
                flows_b = [pred[b] for pred in all_flow_predictions if b < len(pred)]
                if len(flows_b) > 0:
                    # 需要处理不同预测可能有不同数量的点
                    # 简单起见，取第一个预测的点数
                    avg_flow = torch.stack(flows_b, dim=0).mean(dim=0)
                    final_flow.append(avg_flow)

        # 构建返回结果
        pcs_dict = wrap_batch_pcs_future(batch, num_frames=self.num_frames)
        result = self.forward_single(pcs_dict, time_idx=0)
        result['flow'] = final_flow if isinstance(final_flow, list) else [final_flow]
        result['pose_flow'] = pcs_dict['pose_flows'][0]
        result["pc0_valid_point_idxes"] = result["src_valid_point_idxes"]
        result["pc0_points_lst"] = result["src_points_lst"]
        result["pc1_valid_point_idxes"] = result["tgt_valid_point_idxes"]
        result["pc1_points_lst"] = result["tgt_points_lst"]
        result["d_num_voxels"] = [result["d_num_voxels"]]
        result["turbo_mode"] = True
        result["num_windows_used"] = len(all_flow_predictions)

        return result

    def _build_shifted_batch(self, batch, window_offset):
        """
        构建滑动窗口的 batch。

        window_offset=1 时：
        - 新的 pc0 = 原来的 pch1
        - 新的 pc1 = 原来的 pc0
        - 新的 pc2 = 原来的 pc1
        - ...
        """
        try:
            shifted_batch = {}

            # 基本信息保持不变
            shifted_batch['scene_id'] = batch['scene_id']
            if 'timestamp' in batch:
                shifted_batch['timestamp'] = batch['timestamp']

            # 滑动帧
            for i in range(self.num_frames):
                original_idx = i - window_offset

                if original_idx < 0:
                    # 需要历史帧
                    history_idx = -original_idx
                    pc_key = f'pch{history_idx}'
                    pose_key = f'poseh{history_idx}'
                    gm_key = f'gmh{history_idx}'
                else:
                    pc_key = f'pc{original_idx}'
                    pose_key = f'pose{original_idx}'
                    gm_key = f'gm{original_idx}'

                if pc_key in batch:
                    shifted_batch[f'pc{i}'] = batch[pc_key]
                if pose_key in batch:
                    shifted_batch[f'pose{i}'] = batch[pose_key]
                if gm_key in batch:
                    shifted_batch[f'gm{i}'] = batch[gm_key]

            # 检查是否有足够的帧
            if 'pc0' not in shifted_batch or 'pc1' not in shifted_batch:
                return None

            return shifted_batch
        except Exception as e:
            print(f"[Turbo] Error building shifted batch: {e}")
            return None

    def _transform_flow_to_pc0(self, flow, _batch, _window_offset):
        """
        将预测的 flow 变换到 pc0 的坐标系。

        由于不同窗口的参考帧不同，预测的 flow 在不同坐标系下。
        需要变换到统一的 pc0 坐标系才能平均。

        简化处理：假设 ego-motion 已经被补偿，直接返回 flow。
        完整实现需要考虑坐标变换。
        """
        # TODO: 实现完整的坐标变换
        # 当前简化：假设所有帧已经在同一坐标系下（通过 wrap_batch_pcs_future）
        return flow
