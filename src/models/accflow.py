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

from .basic import cal_pose0to1,wrap_batch_pcs
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

    All point clouds are transformed to pc1's coordinate system for consistency.
    This matches the convention used by wrap_batch_pcs() in DeFlow/DeltaFlow.
    """
    batch_sizes = len(batch["pose0"])

    # Initialize result dictionary
    pcs_dict = {f'pc{i}s': [] for i in range(num_frames)}
    pcs_dict['pose_flows'] = []  # pose_flow for pc0 -> pc1 transition (matches wrap_batch_pcs output format)

    for batch_id in range(batch_sizes):
        # Reference pose is pc1's pose - all frames transform to pc1's coordinate
        ref_pose = batch["pose1"][batch_id]

        with torch.no_grad():
            # Transform pc0 to pc1's coordinate system
            pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], ref_pose)
            pc0 = batch["pc0"][batch_id]
            pc0_transformed = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            pcs_dict['pc0s'].append(pc0_transformed)
            
            # Compute pose_flow for pc0 (matches wrap_batch_pcs output format)
            pose_flow = pc0_transformed - pc0
            pcs_dict['pose_flows'].append(pose_flow)

            # pc1 stays in its own coordinate (it IS the reference)
            pcs_dict['pc1s'].append(batch["pc1"][batch_id])

            # Process future frames (pc2, pc3, pc4, ...) - transform to pc1's coordinate
            for i in range(2, num_frames):
                pc_key = f'pc{i}'
                pose_key = f'pose{i}'

                if pc_key in batch and pose_key in batch:
                    # Transform pc_i to pc1's coordinate system
                    pose_i_to_1 = cal_pose0to1(batch[pose_key][batch_id], ref_pose)
                    pc_i = batch[pc_key][batch_id]
                    pc_i_transformed = pc_i @ pose_i_to_1[:3, :3].T + pose_i_to_1[:3, 3]
                    pcs_dict[f'pc{i}s'].append(pc_i_transformed)

    # Stack all lists into tensors
    for i in range(num_frames):
        key = f'pc{i}s'
        if pcs_dict[key]:
            pcs_dict[key] = torch.stack(pcs_dict[key], dim=0)

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

    def voxelize_all_frames(self, pcs_dict, num_frames):
        """
        Pre-voxelize all frames and cache the results.

        Args:
            pcs_dict: Dictionary containing point clouds pc0s, pc1s, pc2s, ...
            num_frames: Number of frames to voxelize

        Returns:
            cache: Dictionary with voxelization results for each frame
        """
        cache = {}
        for i in range(num_frames):
            pc_key = f'pc{i}s'
            if pc_key not in pcs_dict:
                continue
            pc = pcs_dict[pc_key]
            voxel_info_list = self.voxelizer(pc)
            voxel_feats_sp, coors_batch_sp, point_feats_lst = self.process_batch(
                voxel_info_list, if_return_point_feats=True
            )
            cache[i] = {
                'voxel_info_list': voxel_info_list,
                'voxel_feats_sp': voxel_feats_sp,
                'coors_batch_sp': coors_batch_sp,
                'point_feats_lst': point_feats_lst,
            }
        return cache

    def forward_with_cache(self, pcs_dict, voxel_cache, time_idx: int = 0) -> torch.Tensor:
        """
        Forward pass using pre-cached voxelization results.

        Args:
            pcs_dict: Dictionary containing point clouds (for batch size info)
            voxel_cache: Pre-computed voxelization cache from voxelize_all_frames
            time_idx: Which frame pair to predict (0 means pc0->pc1, etc.)

        Returns:
            Dictionary with sparse features and voxel info
        """
        bz_ = pcs_dict['pc0s'].shape[0]
        device = pcs_dict['pc0s'].device

        # Create one-hot time embedding
        time_embed = torch.zeros(bz_, self.time_embed_dim, device=device)
        time_embed[:, time_idx] = 1.0
        time_feat = self.time_proj(time_embed)  # [B, feat_channels]

        # Get cached voxelization for source and target frames
        src_cache = voxel_cache[time_idx]
        tgt_cache = voxel_cache[time_idx + 1]

        src_voxel_feats_sp = src_cache['voxel_feats_sp']
        src_coors_batch_sp = src_cache['coors_batch_sp']
        src_voxel_info_list = src_cache['voxel_info_list']
        src_point_feats_lst = src_cache['point_feats_lst']

        tgt_voxel_feats_sp = tgt_cache['voxel_feats_sp']
        tgt_coors_batch_sp = tgt_cache['coors_batch_sp']
        tgt_voxel_info_list = tgt_cache['voxel_info_list']

        sparse_max_size = [bz_, *self.voxel_spatial_shape, self.num_feature]
        sparse_tgt = torch.sparse_coo_tensor(tgt_coors_batch_sp.t(), tgt_voxel_feats_sp, size=sparse_max_size)
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
            'src_num_voxels': src_voxel_feats_sp.shape[0],
            'tgt_3dvoxel_infos_lst': tgt_voxel_info_list,
            'tgt_num_voxels': tgt_voxel_feats_sp.shape[0],
            'd_num_voxels': indices.shape[0],
            'time_idx': time_idx,
        }
        return output

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


class TimeAwarePointHead(nn.Module):
    """
    Point head with time embedding for AccFlow.

    Adds time embedding to the decoder to reinforce time-awareness throughout the network.
    The time embedding is concatenated with point features before the final MLP.

    Architecture:
        Input: voxel_feat (from backbone) + point_feat (from encoder) + time_embed
        → MLP → flow [N, 3]
    """

    def __init__(self, voxel_feat_dim: int = 16, point_feat_dim: int = 16,
                 time_embed_dim: int = 4, time_hidden_dim: int = 8):
        """
        Args:
            voxel_feat_dim: Dimension of voxel features from backbone
            point_feat_dim: Dimension of point features from encoder
            time_embed_dim: Dimension of one-hot time embedding (num_frames - 1)
            time_hidden_dim: Hidden dimension for time embedding projection
        """
        super().__init__()

        self.time_embed_dim = time_embed_dim

        # Project time embedding to a smaller hidden dimension
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_hidden_dim),
            nn.ReLU(),
        )

        # Total input dim = voxel_feat + point_feat + time_hidden
        self.input_dim = voxel_feat_dim + point_feat_dim + time_hidden_dim

        self.PPmodel_flow = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward_single(self, voxel_feat, voxel_coords, point_feat, time_feat_expanded):
        """
        Forward for a single batch item.

        Args:
            voxel_feat: [C, D, H, W] dense voxel features
            voxel_coords: [N, 3] voxel coordinates for each point
            point_feat: [N, point_feat_dim] point features
            time_feat_expanded: [N, time_hidden_dim] time embedding expanded to each point

        Returns:
            flow: [N, 3] predicted flow
        """
        # Get voxel features at point locations
        voxel_to_point_feat = voxel_feat[:, voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]].T

        # Concatenate all features: voxel + point + time
        concated_point_feat = torch.cat([voxel_to_point_feat, point_feat, time_feat_expanded], dim=-1)

        flow = self.PPmodel_flow(concated_point_feat)
        return flow

    def forward(self, sparse_tensor, voxelizer_infos, pc0_point_feats_lst, time_idx: int = 0):
        """
        Forward pass with time embedding.

        Args:
            sparse_tensor: Sparse tensor from backbone
            voxelizer_infos: List of voxelizer info dicts
            pc0_point_feats_lst: List of point features
            time_idx: Which frame pair (0, 1, 2, or 3)

        Returns:
            List of flow tensors, one per batch item
        """
        voxel_feats = sparse_tensor.dense()
        batch_size = len(voxelizer_infos)
        device = voxel_feats.device

        # Create one-hot time embedding and project
        time_embed = torch.zeros(batch_size, self.time_embed_dim, device=device)
        time_embed[:, time_idx] = 1.0
        time_feat = self.time_proj(time_embed)  # [B, time_hidden_dim]

        flow_outputs = []
        for batch_idx, voxelizer_info in enumerate(voxelizer_infos):
            voxel_coords = voxelizer_info["voxel_coords"]
            point_feat = pc0_point_feats_lst[batch_idx]
            voxel_feat = voxel_feats[batch_idx, :]

            # Expand time features to each point
            num_points = point_feat.shape[0]
            time_feat_expanded = time_feat[batch_idx:batch_idx+1].expand(num_points, -1)

            flow = self.forward_single(voxel_feat, voxel_coords, point_feat, time_feat_expanded)
            flow_outputs.append(flow)

        return flow_outputs


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

        # Decoder selection
        self.decoder_option = decoder_option
        if decoder_option == "deflow":
            self.flowdecoder = SparseGRUHead(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch, num_iters=1)
        elif decoder_option == "time_aware":
            # TimeAwarePointHead with time embedding in decoder
            self.flowdecoder = TimeAwarePointHead(
                voxel_feat_dim=voxel_output_ch,
                point_feat_dim=point_output_ch,
                time_embed_dim=num_frames - 1,  # 4 for 5 frames
                time_hidden_dim=8
            )
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
        # Pass time_idx to decoder if it's TimeAwarePointHead
        if self.decoder_option == "time_aware":
            flows = self.flowdecoder(backbone_res, src_3dvoxel_infos_lst, sparse_dict['src_point_feats_lst'], time_idx=time_idx)
        else:
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

    def forward_single_with_cache(self, pcs_dict, voxel_cache, time_idx: int = 0):
        """
        Forward pass for a single time index using cached voxelization.

        Args:
            pcs_dict: Dictionary with transformed point clouds
            voxel_cache: Pre-computed voxelization cache
            time_idx: Which frame pair to predict (0=pc0->pc1, 1=pc1->pc2, etc.)

        Returns:
            Dictionary with flow predictions and auxiliary info
        """
        self.timer[1].start("3D Sparse Voxel (cached)")
        sparse_dict = self.pc2voxel.forward_with_cache(pcs_dict, voxel_cache, time_idx=time_idx)
        self.timer[1].stop()

        self.timer[3].start("3D Network")
        backbone_res = self.backbone(sparse_dict['delta_sparse'])
        src_3dvoxel_infos_lst = sparse_dict['src_3dvoxel_infos_lst']
        self.timer[3].stop()

        self.timer[4].start("Flow Decoder")
        # Pass time_idx to decoder if it's TimeAwarePointHead
        if self.decoder_option == "time_aware":
            flows = self.flowdecoder(backbone_res, src_3dvoxel_infos_lst, sparse_dict['src_point_feats_lst'], time_idx=time_idx)
        else:
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
            # Normal inference: predict flow for frame pair specified by time_idx
            result = self.forward_single(pcs_dict, time_idx=time_idx)

            # Determine source and target frame based on time_idx
            src_frame = time_idx
            tgt_frame = time_idx + 1
            src_pose_key = f"pose{src_frame}"
            tgt_pose_key = f"pose{tgt_frame}"
            src_pc_key = f"pc{src_frame}"

            # Transform flow from source coordinate to target coordinate for evaluation
            # The network predicts flow in source frame's coordinate system, but evaluation
            # expects flow in target frame's coordinate system (like SeFlow/DeltaFlow)
            # Flow is a vector, so only rotation is needed (no translation)
            batch_size = len(batch["pose0"])
            transformed_flows = []
            for b in range(batch_size):
                flow_src = result['flow'][b]  # [N, 3] in source coordinate
                # Get rotation from source to target
                pose_src_to_tgt = cal_pose0to1(batch[src_pose_key][b], batch[tgt_pose_key][b])
                R_src_to_tgt = pose_src_to_tgt[:3, :3]  # [3, 3]
                # Rotate flow vector to target coordinate
                flow_tgt = flow_src @ R_src_to_tgt.T
                transformed_flows.append(flow_tgt)
            result['flow'] = transformed_flows

            # Recompute pose_flow in target coordinate system (matching SeFlow/DeltaFlow convention)
            # pose_flow = transform_src - src, where transform_src is source transformed to target coord
            pose_flows = []
            for b in range(batch_size):
                pc_src = batch[src_pc_key][b]
                pose_src_to_tgt = cal_pose0to1(batch[src_pose_key][b], batch[tgt_pose_key][b])
                transform_pc_src = pc_src @ pose_src_to_tgt[:3, :3].T + pose_src_to_tgt[:3, 3]
                pose_flow = transform_pc_src - pc_src
                pose_flows.append(pose_flow)
            result['pose_flow'] = pose_flows

            # For compatibility, also add pc0/pc1 specific keys
            # Note: these indices are relative to the source frame (pc{time_idx})
            result["pc0_valid_point_idxes"] = result["src_valid_point_idxes"]
            result["pc0_points_lst"] = result["src_points_lst"]
            result["pc1_valid_point_idxes"] = result["tgt_valid_point_idxes"]
            result["pc1_points_lst"] = result["tgt_points_lst"]
            result["d_num_voxels"] = [result["d_num_voxels"]]
            result["time_idx"] = time_idx

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

        OPTIMIZED VERSION: Uses batched forward and caches voxelization results.

        This implements the self-supervised training procedure:
        1. Pre-voxelize all frames once (cached)
        2. Batch forward for each time_idx (all samples in batch together)
        3. KNN interpolation per sample (cannot be batched due to varying point counts)
        4. Accumulate flow predictions

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

        # ====== OPTIMIZATION 1: Pre-voxelize all frames once ======
        self.timer[1].start("Voxelize All Frames")
        voxel_cache = self.pc2voxel.voxelize_all_frames(pcs_dict, num_frames=num_acc_steps + 1)
        self.timer[1].stop()

        # ====== OPTIMIZATION 2: Batched forward for time_idx=0 ======
        # All samples in batch are processed together
        result_0 = self.forward_single_with_cache(pcs_dict, voxel_cache, time_idx=0)

        # Extract per-sample results
        flows_t0 = result_0['flow']  # List of [N_i, 3] for each sample
        valid_idxes_t0 = result_0['src_valid_point_idxes']  # List of valid indices

        # Initialize accumulated positions and flows for each sample
        accumulated_positions = []  # List of [N_i, 3]
        accumulated_flows = []  # List of [N_i, 3]
        all_valid_idxes = []
        all_pc0_points = []

        for batch_id in range(batch_size):
            pc0 = pcs_dict['pc0s'][batch_id]
            valid_mask = ~torch.isnan(pc0[:, 0])
            pc0 = pc0[valid_mask]

            flow_0 = flows_t0[batch_id]
            valid_idx_0 = valid_idxes_t0[batch_id]

            pc0_valid = pc0[valid_idx_0]
            accumulated_pos = pc0_valid + flow_0
            accumulated_flow = flow_0.clone()

            accumulated_positions.append(accumulated_pos)
            accumulated_flows.append(accumulated_flow)
            all_valid_idxes.append(valid_idx_0)
            all_pc0_points.append(pc0_valid)

        # ====== Iterate through remaining time steps ======
        for t in range(1, num_acc_steps):
            # ====== OPTIMIZATION 2: Batched forward for time_idx=t ======
            result_t = self.forward_single_with_cache(pcs_dict, voxel_cache, time_idx=t)
            flows_t = result_t['flow']
            valid_idxes_t = result_t['src_valid_point_idxes']

            # KNN interpolation must be done per-sample (varying point counts)
            for batch_id in range(batch_size):
                accumulated_pos = accumulated_positions[batch_id]
                accumulated_flow = accumulated_flows[batch_id]

                # Get pc_t valid points
                pc_t = pcs_dict[f'pc{t}s'][batch_id]
                valid_mask_t = ~torch.isnan(pc_t[:, 0])
                pc_t = pc_t[valid_mask_t]

                flow_t = flows_t[batch_id]
                valid_idx_t = valid_idxes_t[batch_id]
                pc_t_valid = pc_t[valid_idx_t]

                if accumulated_pos.shape[0] > 0 and pc_t_valid.shape[0] > 0:
                    # Interpolate flow from pc_t to accumulated_pos
                    interpolated_flow = interpolate_flow(
                        accumulated_pos, pc_t_valid, flow_t,
                        method=self.interpolation_method,
                        k=self.knn_k
                    )

                    # Update accumulated position and flow
                    accumulated_positions[batch_id] = accumulated_pos + interpolated_flow
                    accumulated_flows[batch_id] = accumulated_flow + interpolated_flow

        # Package results
        pose_flows = pcs_dict['pose_flows'][0]

        result = {
            "flow": accumulated_flows,
            "pose_flow": pose_flows,
            "pc0_valid_point_idxes": all_valid_idxes,
            "pc0_points_lst": all_pc0_points,
            "accumulated_target_frame": num_acc_steps,
            "d_num_voxels": [0],
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


class AccFlow2Frame(nn.Module):
    """
    2-Frame AccFlow with Sliding Window Accumulated Error Training.

    Key features:
    1. Network only sees 2 frames at a time (pc_t, pc_{t+1})
    2. No time embedding - simpler architecture like DeFlow
    3. Training uses sliding window on 5-frame sequences:
       - Predict F0: pc0 -> pc1
       - Warp pc0 to pc1', use KNN to get F1 from pc1
       - Predict F1: pc1 -> pc2 (with real pc1, pc2)
       - Accumulate: pc0 + F0 + interpolated_F1 + ...
    4. Loss computed on accumulated trajectory vs final frame

    This allows the model to learn consistent flow prediction that
    accumulates well over multiple frames, without needing time embedding.
    """

    def __init__(self, voxel_size=[0.2, 0.2, 0.2],
                 point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size=[512, 512, 32],
                 num_frames=5,  # Number of frames in dataset for accumulation
                 planes=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
                 num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 decay_factor=1.0,
                 decoder_option="default",
                 knn_k=3,
                 interpolation_method="knn",
                 accumulate_probs=None,
                 ):
        super().__init__()
        point_output_ch = planes[0]
        voxel_output_ch = planes[-1]
        self.timer = dztimer.Timing()
        self.num_frames = num_frames  # For accumulation training
        self.knn_k = knn_k
        self.interpolation_method = interpolation_method
        self.accumulate_probs = accumulate_probs

        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print('[LOG] AccFlow2Frame Param detail: voxel_size = {}, pseudo_dims = {}, num_frames={}'.format(
                voxel_size, grid_feature_size, num_frames))
            print('[LOG] Model detail: planes = {}, time decay = {}, decoder = {}, knn_k = {}, interp = {}'.format(
                planes, decay_factor, decoder_option, knn_k, interpolation_method))
            print('[LOG] Accumulate probs: {}'.format(self.accumulate_probs))

        # Simple 2-frame encoder without time embedding
        self.pc2voxel = AccFlow2FrameEncoder(
            voxel_size=voxel_size,
            pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2]],
            point_cloud_range=point_cloud_range,
            feat_channels=point_output_ch,
            decay_factor=decay_factor,
            timer=self.timer[1]
        )
        self.backbone = MinkUNet(planes, num_layer)

        # Decoder - no time embedding needed
        self.decoder_option = decoder_option
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

    def forward_pair(self, pc0, pc1, pose0, pose1):
        """
        Forward pass for a single pair of point clouds.

        Args:
            pc0: [B, N, 3] source point cloud
            pc1: [B, M, 3] target point cloud
            pose0: [B, 4, 4] pose of pc0
            pose1: [B, 4, 4] pose of pc1

        Returns:
            Dictionary with flow predictions
        """
        # Use wrap_batch_pcs for consistent coordinate system handling (pc1 coordinate)
        batch_dict = {
            'pc0': pc0,
            'pc1': pc1,
            'pose0': pose0,
            'pose1': pose1
        }
        
        # Transform pc0 to pc1's coordinate system using standard function
        pcs_dict = wrap_batch_pcs_future(batch_dict, num_frames=2)

        # Forward through network
        self.timer[1].start("3D Sparse Voxel")
        sparse_dict = self.pc2voxel(pcs_dict)
        self.timer[1].stop()

        self.timer[3].start("3D Network")
        backbone_res = self.backbone(sparse_dict['delta_sparse'])
        pc0_3dvoxel_infos_lst = sparse_dict['pc0_3dvoxel_infos_lst']
        pc1_3dvoxel_infos_lst = sparse_dict['pc1_3dvoxel_infos_lst']
        self.timer[3].stop()

        self.timer[4].start("Flow Decoder")
        flows = self.flowdecoder(backbone_res, pc0_3dvoxel_infos_lst, sparse_dict['pc0_point_feats_lst'])
        self.timer[4].stop()

        return {
            "flow": flows,
            "pose_flow": pcs_dict['pose_flows'],
            "pc0_valid_point_idxes": [e["point_idxes"] for e in pc0_3dvoxel_infos_lst],
            "pc0_points_lst": [e["points"] for e in pc0_3dvoxel_infos_lst],
            "pc1_valid_point_idxes": [e["point_idxes"] for e in pc1_3dvoxel_infos_lst],
            "pc1_points_lst": [e["points"] for e in pc1_3dvoxel_infos_lst],
            "d_num_voxels": [sparse_dict['d_num_voxels']],
        }

    def _sample_num_accumulate_steps(self):
        """Sample number of accumulation steps based on probability."""
        if self.accumulate_probs is None:
            return self.num_frames - 1

        import random
        r = random.random()
        cumsum = 0.0
        for i, prob in enumerate(self.accumulate_probs):
            cumsum += prob
            if r < cumsum:
                return i + 1
        return len(self.accumulate_probs)

    def forward_accumulated_error(self, batch):
        """
        Training forward with accumulated error using sliding window.

        Process:
        1. Predict F0: forward(pc0, pc1) -> flow_0
        2. Warp: pc0' = pc0 + flow_0
        3. For t = 1 to num_acc_steps-1:
           a. Predict F_t: forward(pc_t, pc_{t+1}) -> flow_t
           b. Interpolate flow_t to pc0' positions using KNN
           c. Accumulate: pc0' = pc0' + interpolated_flow_t
        4. Compare pc0' with pc_{num_acc_steps} using chamfer distance
        """
        batch_size = len(batch['pose0'])
        device = batch['pc0'].device

        num_acc_steps = self._sample_num_accumulate_steps()
        max_steps = self.num_frames - 1
        num_acc_steps = min(num_acc_steps, max_steps)

        # Transform ALL frames to pc1 coordinate system ONCE
        # This ensures all operations happen in the same coordinate system
        pcs_dict_all = wrap_batch_pcs_future(batch, num_frames=num_acc_steps + 1)

        # Step 1: Predict flow for pc0 -> pc1
        # Build pcs_dict for encoder (needs pc0s and pc1s)
        pcs_dict_0 = {
            'pc0s': pcs_dict_all['pc0s'],
            'pc1s': pcs_dict_all['pc1s'],
            'pose_flows': pcs_dict_all['pose_flows']
        }
        
        # Forward through network
        self.timer[1].start("3D Sparse Voxel")
        sparse_dict_0 = self.pc2voxel(pcs_dict_0)
        self.timer[1].stop()

        self.timer[3].start("3D Network")
        backbone_res_0 = self.backbone(sparse_dict_0['delta_sparse'])
        pc0_3dvoxel_infos_lst = sparse_dict_0['pc0_3dvoxel_infos_lst']
        self.timer[3].stop()

        self.timer[4].start("Flow Decoder")
        flows_0 = self.flowdecoder(backbone_res_0, pc0_3dvoxel_infos_lst, sparse_dict_0['pc0_point_feats_lst'])
        self.timer[4].stop()
        
        valid_idxes_0 = [e["point_idxes"] for e in pc0_3dvoxel_infos_lst]

        # Initialize accumulated positions and flows
        accumulated_positions = []
        accumulated_flows = []
        all_valid_idxes = []
        all_pc0_points = []

        for b in range(batch_size):
            flow_0 = flows_0[b]
            valid_idx_0 = valid_idxes_0[b]
            
            # pc0 is already in pc1 coordinate from wrap_batch_pcs_future
            pc0_in_pc1_coord = pcs_dict_all['pc0s'][b]
            valid_mask = ~torch.isnan(pc0_in_pc1_coord[:, 0])
            pc0_valid = pc0_in_pc1_coord[valid_mask]

            # Only track points that went through network
            pc0_tracked = pc0_valid[valid_idx_0]
            accumulated_pos = pc0_tracked + flow_0
            accumulated_flow = flow_0.clone()

            accumulated_positions.append(accumulated_pos)
            accumulated_flows.append(accumulated_flow)
            all_valid_idxes.append(valid_idx_0)
            all_pc0_points.append(pc0_tracked)

        # Step 2-3: Iterate through remaining frame pairs
        # All frames are already in pc1 coordinate system from pcs_dict_all
        for t in range(1, num_acc_steps):
            pc_t_key = f'pc{t}s'
            pc_t1_key = f'pc{t+1}s'

            if pc_t_key not in pcs_dict_all or pc_t1_key not in pcs_dict_all:
                break

            # Build pcs_dict for this frame pair (both already in pc1 coordinate)
            pcs_dict_t = {
                'pc0s': pcs_dict_all[pc_t_key],  # pc_t in pc1 coord
                'pc1s': pcs_dict_all[pc_t1_key],  # pc_{t+1} in pc1 coord
            }
            
            # Forward through network
            sparse_dict_t = self.pc2voxel(pcs_dict_t)
            backbone_res_t = self.backbone(sparse_dict_t['delta_sparse'])
            pc_t_3dvoxel_infos_lst = sparse_dict_t['pc0_3dvoxel_infos_lst']
            flows_t = self.flowdecoder(backbone_res_t, pc_t_3dvoxel_infos_lst, sparse_dict_t['pc0_point_feats_lst'])
            
            valid_idxes_t = [e["point_idxes"] for e in pc_t_3dvoxel_infos_lst]

            # KNN interpolation for each sample
            # All points are in pc1 coordinate system, so we can directly compute distances
            for b in range(batch_size):
                accumulated_pos = accumulated_positions[b]
                accumulated_flow = accumulated_flows[b]

                # Get pc_t in pc1 coordinate (already transformed)
                pc_t_in_pc1_coord = pcs_dict_all[pc_t_key][b]
                valid_mask_t = ~torch.isnan(pc_t_in_pc1_coord[:, 0])
                pc_t_valid = pc_t_in_pc1_coord[valid_mask_t]

                flow_t = flows_t[b]
                valid_idx_t = valid_idxes_t[b]
                pc_t_tracked = pc_t_valid[valid_idx_t]

                # Both accumulated_pos and pc_t_tracked are in pc1 coordinate
                if accumulated_pos.shape[0] > 0 and pc_t_tracked.shape[0] > 0:
                    interpolated_flow = interpolate_flow(
                        accumulated_pos, pc_t_tracked, flow_t,
                        method=self.interpolation_method,
                        k=self.knn_k
                    )

                    # Update position and flow (all in pc1 coordinate)
                    accumulated_positions[b] = accumulated_pos + interpolated_flow
                    accumulated_flows[b] = accumulated_flow + interpolated_flow

        # Package results
        final_accumulated_flows = []
        for b in range(batch_size):
            final_accumulated_flows.append(accumulated_flows[b])

        result = {
            "flow": final_accumulated_flows,
            "pose_flow": pcs_dict_all['pose_flows'],
            "pc0_valid_point_idxes": all_valid_idxes,
            "pc0_points_lst": all_pc0_points,
            "accumulated_target_frame": num_acc_steps,
            "d_num_voxels": [0],
        }

        return result

    def forward(self, batch, training_mode: bool = False):
        """
        Main forward pass.

        Args:
            batch: Batch from dataloader
            training_mode: If True, use accumulated error training

        Returns:
            Model output dictionary
        """
        self.cnt += 1

        if training_mode and self.num_frames > 2:
            return self.forward_accumulated_error(batch)
        else:
            # Standard 2-frame inference using wrap_batch_pcs
            pcs_dict = wrap_batch_pcs(batch, num_frames=2)
            
            # Forward through network
            self.timer[1].start("3D Sparse Voxel")
            sparse_dict = self.pc2voxel(pcs_dict)
            self.timer[1].stop()

            self.timer[3].start("3D Network")
            backbone_res = self.backbone(sparse_dict['delta_sparse'])
            pc0_3dvoxel_infos_lst = sparse_dict['pc0_3dvoxel_infos_lst']
            pc1_3dvoxel_infos_lst = sparse_dict['pc1_3dvoxel_infos_lst']
            self.timer[3].stop()

            self.timer[4].start("Flow Decoder")
            flows = self.flowdecoder(backbone_res, pc0_3dvoxel_infos_lst, sparse_dict['pc0_point_feats_lst'])
            self.timer[4].stop()

            return {
                "flow": flows,
                "pose_flow": pcs_dict['pose_flows'],
                "pc0_valid_point_idxes": [e["point_idxes"] for e in pc0_3dvoxel_infos_lst],
                "pc0_points_lst": [e["points"] for e in pc0_3dvoxel_infos_lst],
                "pc1_valid_point_idxes": [e["point_idxes"] for e in pc1_3dvoxel_infos_lst],
                "pc1_points_lst": [e["points"] for e in pc1_3dvoxel_infos_lst],
                "d_num_voxels": [sparse_dict['d_num_voxels']],
            }


class AccFlow2FrameEncoder(nn.Module):
    """
    Simple 2-frame encoder without time embedding.
    Similar to DeltaFlow's SparseVoxelNet but optimized for AccFlow2Frame.
    """

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, decay_factor=1.0, timer=None) -> None:
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
        self.decay_factor = decay_factor

        if timer is None:
            self.timer = dztimer.Timing()
            self.timer.start("Total")
        else:
            self.timer = timer

    def process_batch(self, voxel_info_list, if_return_point_feats=False):
        """Process voxel info list to get features."""
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

    def forward(self, pcs_dict) -> torch.Tensor:
        """
        Forward pass for 2-frame input.

        Args:
            pcs_dict: Dictionary with 'pc0s' and 'pc1s'

        Returns:
            Dictionary with sparse features
        """
        self.timer[0].start("A_Voxelization")
        pc0_voxel_info_list = self.voxelizer(pcs_dict['pc0s'])
        pc1_voxel_info_list = self.voxelizer(pcs_dict['pc1s'])
        self.timer[0].stop()

        self.timer[1].start("B_Pillar_Feature")
        pc0_voxel_feats_sp, pc0_coors_batch_sp, pc0_point_feats_lst = self.process_batch(
            pc0_voxel_info_list, if_return_point_feats=True)
        pc1_voxel_feats_sp, pc1_coors_batch_sp = self.process_batch(pc1_voxel_info_list)
        self.timer[1].stop()

        bz_ = pcs_dict['pc0s'].shape[0]
        sparse_max_size = [bz_, *self.voxel_spatial_shape, self.num_feature]
        sparse_pc0 = torch.sparse_coo_tensor(pc0_coors_batch_sp.t(), pc0_voxel_feats_sp, size=sparse_max_size)
        sparse_pc1 = torch.sparse_coo_tensor(pc1_coors_batch_sp.t(), pc1_voxel_feats_sp, size=sparse_max_size)

        # Delta features (pc1 - pc0)
        self.timer[2].start("D_Delta_Sparse")
        sparse_diff = sparse_pc1 - sparse_pc0
        features = sparse_diff.coalesce().values()
        indices = sparse_diff.coalesce().indices().t().to(dtype=torch.int32)

        all_pcdiff_sparse = spconv.SparseConvTensor(
            features.contiguous(), indices.contiguous(),
            self.voxel_spatial_shape, bz_
        )
        self.timer[2].stop()

        output = {
            'delta_sparse': all_pcdiff_sparse,
            'pc0_3dvoxel_infos_lst': pc0_voxel_info_list,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_num_voxels': pc0_voxel_feats_sp.shape[0],
            'pc1_3dvoxel_infos_lst': pc1_voxel_info_list,
            'pc1_num_voxels': pc1_voxel_feats_sp.shape[0],
            'd_num_voxels': indices.shape[0],
        }
        return output
