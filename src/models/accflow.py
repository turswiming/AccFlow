"""
# Created: 2025-12-05
# AccumulateErrorFlow - A self-supervised scene flow method
# Based on DeltaFlow architecture with accumulated error training

# Key features:
# 1. Uses history frames (pch1, pch2, ...) like DeltaFlow for richer temporal context
# 2. Training uses sliding window accumulated error for self-supervised learning
# 3. Inference only requires history frames + pc0 + pc1 (same as DeltaFlow)
"""

import torch
import torch.nn as nn
import dztimer
import spconv.pytorch as spconv
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

from .basic import cal_pose0to1, wrap_batch_pcs
from .basic.encoder import DynamicVoxelizer, DynamicPillarFeatureNet
from .basic.sparse_encoder import MinkUNet, SparseVoxelNet
from .basic.decoder import SparseGRUHead
from .basic.flow4d_module import Point_head

# Try to import CUDA-accelerated interpolation methods
try:
    from pytorch3d.ops import knn_points, knn_gather
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False

try:
    from torch_points_kernels import three_interpolate, three_nn
    HAS_POINTNET_OPS = True
except ImportError:
    HAS_POINTNET_OPS = False


def knn_interpolate_flow(query_points, ref_points, ref_flow, k=3):
    """Interpolate flow from reference points to query points using KNN."""
    if HAS_PYTORCH3D and query_points.shape[0] > 100:
        query_pts = query_points.unsqueeze(0)
        ref_pts = ref_points.unsqueeze(0)
        ref_f = ref_flow.unsqueeze(0)

        knn_result = knn_points(query_pts, ref_pts, K=k, return_sorted=False)
        knn_dist = knn_result.dists
        knn_idx = knn_result.idx

        knn_flow = knn_gather(ref_f, knn_idx)

        weights = 1.0 / (knn_dist.sqrt() + 1e-8)
        weights = weights / weights.sum(dim=2, keepdim=True)

        interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=2)
        return interpolated_flow.squeeze(0)
    else:
        dist = torch.cdist(query_points, ref_points)
        knn_dist, knn_idx = dist.topk(k, dim=1, largest=False)
        weights = 1.0 / (knn_dist + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        knn_flow = ref_flow[knn_idx]
        interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=1)
        return interpolated_flow


def three_nn_interpolate_flow(query_points, ref_points, ref_flow):
    """PointNet++ style three nearest neighbor interpolation."""
    if HAS_POINTNET_OPS:
        query_pts = query_points.unsqueeze(0).contiguous()
        ref_pts = ref_points.unsqueeze(0).contiguous()
        ref_f = ref_flow.unsqueeze(0).transpose(1, 2).contiguous()

        dist, idx = three_nn(query_pts, ref_pts)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = dist_recip.sum(dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated = three_interpolate(ref_f, idx, weight)
        return interpolated.squeeze(0).transpose(0, 1)
    else:
        return knn_interpolate_flow(query_points, ref_points, ref_flow, k=3)


def rbf_interpolate_flow(query_points, ref_points, ref_flow, sigma=0.5, max_neighbors=16):
    """Radial Basis Function (RBF) interpolation for flow."""
    dist = torch.cdist(query_points, ref_points)

    if max_neighbors < ref_points.shape[0]:
        knn_dist, knn_idx = dist.topk(max_neighbors, dim=1, largest=False)
        weights = torch.exp(-knn_dist ** 2 / (2 * sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        knn_flow = ref_flow[knn_idx]
        interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=1)
    else:
        weights = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        interpolated_flow = weights @ ref_flow

    return interpolated_flow


def idw_interpolate_flow(query_points, ref_points, ref_flow, power=2, k=8):
    """Inverse Distance Weighting (IDW) interpolation."""
    dist = torch.cdist(query_points, ref_points)
    knn_dist, knn_idx = dist.topk(k, dim=1, largest=False)

    weights = 1.0 / (knn_dist ** power + 1e-8)
    weights = weights / weights.sum(dim=1, keepdim=True)

    knn_flow = ref_flow[knn_idx]
    interpolated_flow = (weights.unsqueeze(-1) * knn_flow).sum(dim=1)

    return interpolated_flow


INTERPOLATION_METHODS = {
    'knn': knn_interpolate_flow,
    'three_nn': three_nn_interpolate_flow,
    'rbf': rbf_interpolate_flow,
    'idw': idw_interpolate_flow,
}


def interpolate_flow(query_points, ref_points, ref_flow, method='knn', **kwargs):
    """Unified interface for flow interpolation."""
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"Unknown interpolation method: {method}. Available: {list(INTERPOLATION_METHODS.keys())}")
    return INTERPOLATION_METHODS[method](query_points, ref_points, ref_flow, **kwargs)


class AccFlow(nn.Module):
    """
    AccumulateErrorFlow with history frames support.

    Architecture identical to DeltaFlow:
    - Uses SparseVoxelNet encoder with history frames (pch1, pch2, ...) 
    - MinkUNet backbone
    - Point_head or SparseGRUHead decoder

    Training mode (accumulated error):
    - All frames are first transformed to pc1 coordinate system
    - Sliding window predicts flow for consecutive frame pairs with history context
    - Errors are accumulated through KNN interpolation
    - Loss computed on accumulated trajectory vs target frame

    Inference mode:
    - Standard 2-frame prediction with history context (same as DeltaFlow)

    Args:
        voxel_size: Voxel size for point cloud voxelization
        point_cloud_range: Range of point cloud [x_min, y_min, z_min, x_max, y_max, z_max]
        grid_feature_size: Size of the voxel grid [X, Y, Z]
        num_frames: Total number of frames (2 + num_history_frames)
                    e.g., num_frames=5 means pc0, pc1, pch1, pch2, pch3
        planes: Channel dimensions for MinkUNet
        num_layer: Number of layers for each stage in MinkUNet
        decay_factor: Decay factor for history frame features
        decoder_option: 'default' (Point_head) or 'deflow' (SparseGRUHead)
        knn_k: Number of neighbors for KNN interpolation in accumulated training
        interpolation_method: Method for flow interpolation ('knn', 'three_nn', 'rbf', 'idw')
        accumulate_probs: Probability distribution for number of accumulation steps
    """

    def __init__(self, voxel_size=[0.2, 0.2, 0.2],
                 point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size=[512, 512, 32],
                 num_frames=2,
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
        self.num_frames = num_frames
        self.knn_k = knn_k
        self.interpolation_method = interpolation_method
        self.accumulate_probs = accumulate_probs

        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print('[LOG] AccFlow Param detail: voxel_size = {}, pseudo_dims = {}, num_frames={}'.format(
                voxel_size, grid_feature_size, num_frames))
            print('[LOG] Model detail: planes = {}, decay = {}, decoder = {}, knn_k = {}, interp = {}'.format(
                planes, decay_factor, decoder_option, knn_k, interpolation_method))
            if accumulate_probs is not None:
                print('[LOG] Accumulate probs: {}'.format(self.accumulate_probs))

        # Use DeltaFlow's SparseVoxelNet encoder (supports history frames with decay)
        self.pc2voxel = SparseVoxelNet(
            voxel_size=voxel_size,
            pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2]],
            point_cloud_range=point_cloud_range,
            feat_channels=point_output_ch,
            decay_factor=decay_factor,
            timer=self.timer[1]
        )
        self.backbone = MinkUNet(planes, num_layer)

        # Decoder
        self.decoder_option = decoder_option
        if decoder_option == "deflow":
            self.flowdecoder = SparseGRUHead(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch, num_iters=1)
        else:
            self.flowdecoder = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)

        self.voxel_spatial_shape = grid_feature_size
        self.timer.start("Total")

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model."):]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

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
                return i
        return len(self.accumulate_probs) - 1

    def _forward_network(self, pcs_dict):
        """
        Forward through encoder, backbone, and decoder.
        
        Args:
            pcs_dict: Dictionary containing:
                - pc0s, pc1s: current frame pair [B, N, 3]
                - pch1s, pch2s, ...: history frames (optional)
                - pose_flows: ego-motion displacement for pc0
        
        Returns:
            Dictionary with flow predictions and auxiliary info
        """
        sparse_dict = self.pc2voxel(pcs_dict)
        backbone_res = self.backbone(sparse_dict['delta_sparse'])
        pc0_3dvoxel_infos_lst = sparse_dict['pc0_3dvoxel_infos_lst']
        pc1_3dvoxel_infos_lst = sparse_dict['pc1_3dvoxel_infos_lst']
        flows = self.flowdecoder(backbone_res, pc0_3dvoxel_infos_lst, sparse_dict['pc0_point_feats_lst'])

        return {
            "flow": flows,
            "pose_flow": pcs_dict['pose_flows'],
            "pc0_3dvoxel_infos_lst": pc0_3dvoxel_infos_lst,
            "pc1_3dvoxel_infos_lst": pc1_3dvoxel_infos_lst,
            "pc0_point_feats_lst": sparse_dict['pc0_point_feats_lst'],
            "d_num_voxels": sparse_dict['d_num_voxels'],
            "pch1_3dvoxel_infos_lst": sparse_dict.get('pch1_3dvoxel_infos_lst', None),
        }

    def _transform_all_to_pc1_coord(self, batch, num_acc_steps):
        """
        Transform all frames (pc0, pc1, pc2, ..., pch1, pch2, ...) to pc1 coordinate system.
        
        Args:
            batch: Input batch containing all point clouds and poses
            num_acc_steps: Number of accumulation steps (determines how many future frames needed)
        
        Returns:
            Dictionary with transformed point clouds:
                - 'pc0', 'pc1', 'pc2', ... : transformed point clouds [B, N, 3]
                - 'pch1', 'pch2', ... : transformed history frames [B, N, 3]
                - 'pose_flow_0': pose flow for pc0 (list of [N, 3])
        """
        batch_size = len(batch['pose0'])
        
        transformed = {}
        pose_flow_0 = []
        
        for b in range(batch_size):
            ref_pose = batch['pose1'][b]  # pc1's pose as reference
            
            with torch.no_grad():
                # pc0 → pc1 coordinate
                pose_0to1 = cal_pose0to1(batch['pose0'][b], ref_pose)
                pc0_in_pc1 = batch['pc0'][b] @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
                
                if b == 0:
                    transformed['pc0'] = [pc0_in_pc1]
                    transformed['pc1'] = [batch['pc1'][b]]  # pc1 already in its own coord
                else:
                    transformed['pc0'].append(pc0_in_pc1)
                    transformed['pc1'].append(batch['pc1'][b])
                
                pose_flow_0.append(pc0_in_pc1 - batch['pc0'][b])
                
                # Future frames pc2, pc3, ... → pc1 coordinate
                for i in range(2, num_acc_steps + 1):
                    pc_key = f'pc{i}'
                    pose_key = f'pose{i}'
                    if pc_key in batch and pose_key in batch:
                        pose_i_to_1 = cal_pose0to1(batch[pose_key][b], ref_pose)
                        pc_i_in_pc1 = batch[pc_key][b] @ pose_i_to_1[:3, :3].T + pose_i_to_1[:3, 3]
                        
                        if b == 0:
                            transformed[pc_key] = [pc_i_in_pc1]
                        else:
                            transformed[pc_key].append(pc_i_in_pc1)
                
                # History frames pch1, pch2, ... → pc1 coordinate
                for h in range(1, self.num_frames - 1):
                    pch_key = f'pch{h}'
                    poseh_key = f'poseh{h}'
                    if pch_key in batch and poseh_key in batch:
                        pose_h_to_1 = cal_pose0to1(batch[poseh_key][b], ref_pose)
                        pch_in_pc1 = batch[pch_key][b] @ pose_h_to_1[:3, :3].T + pose_h_to_1[:3, 3]
                        
                        if b == 0:
                            transformed[pch_key] = [pch_in_pc1]
                        else:
                            transformed[pch_key].append(pch_in_pc1)
        
        # Stack into tensors
        for key in transformed:
            transformed[key] = torch.stack(transformed[key], dim=0)
        transformed['pose_flow_0'] = pose_flow_0
        
        return transformed

    def forward_accumulated_error(self, batch):
        """
        Training forward with accumulated error using sliding window.

        All frames are first transformed to pc1 coordinate system, then:
        1. Predict flow F0: pc0 -> pc1 (with original history frames pch1, pch2, ...)
        2. Warp pc0 to get pc0' = pc0 + F0
        3. For each subsequent step t:
           a. Build pcs_dict with pc_t as "pc0", pc_{t+1} as "pc1"
           b. Use sliding history: pc_{t-1} as pch1, pc_{t-2} as pch2, etc.
           c. Since all in pc1 coord, pose is identity (pose_flow = 0)
           d. Predict flow F_t, interpolate to accumulated positions
           e. Accumulate: pc0' = pc0' + interpolated_F_t
        4. Compare final pc0' with target frame
        """
        batch_size = len(batch['pose0'])
        device = batch['pc0'].device

        sampled_acc_steps = self._sample_num_accumulate_steps()
        # Convert 0-based index to target frame index (0 -> pc1, 1 -> pc2, etc.)
        target_frame_idx = sampled_acc_steps + 1
        
        # Check how many future frames we actually have
        max_future = 1
        while f'pc{max_future + 1}' in batch and f'pose{max_future + 1}' in batch:
            max_future += 1
        num_acc_steps = min(target_frame_idx, max_future)

        # Step 1: Transform all frames to pc1 coordinate system
        all_pcs = self._transform_all_to_pc1_coord(batch, num_acc_steps)

        # Step 2: First step pc0 → pc1 with original history frames
        # All frames already in pc1 coord, so pose is identity → pose_flow = 0
        pcs_dict_0 = {
            'pc0s': all_pcs['pc0'],
            'pc1s': all_pcs['pc1'],
            'pose_flows': all_pcs['pose_flow_0'],
        }
        # Add history frames (pch1, pch2, ...)
        for h in range(1, self.num_frames - 1):
            pch_key = f'pch{h}'
            if pch_key in all_pcs:
                pcs_dict_0[f'{pch_key}s'] = all_pcs[pch_key]

        self.timer[1].start("3D Sparse Voxel")
        result_0 = self._forward_network(pcs_dict_0)
        self.timer[1].stop()
        
        flows_0 = result_0['flow']
        pc0_3dvoxel_infos_lst = result_0['pc0_3dvoxel_infos_lst']
        valid_idxes_0 = [e["point_idxes"] for e in pc0_3dvoxel_infos_lst]

        # Initialize accumulated positions and flows
        accumulated_positions = []
        accumulated_flows = []
        all_valid_idxes = []
        all_pc0_points = []

        for b in range(batch_size):
            flow_0 = flows_0[b]
            valid_idx_0 = valid_idxes_0[b]
            
            # pc0 is in pc1 coordinate after transformation
            pc0_in_pc1 = all_pcs['pc0'][b]
            valid_mask = ~torch.isnan(pc0_in_pc1[:, 0])
            pc0_valid = pc0_in_pc1[valid_mask]

            # Only track points that went through network
            pc0_tracked = pc0_valid[valid_idx_0]
            accumulated_pos = pc0_tracked + flow_0
            accumulated_flow = flow_0.clone()

            accumulated_positions.append(accumulated_pos)
            accumulated_flows.append(accumulated_flow)
            all_valid_idxes.append(valid_idx_0)
            all_pc0_points.append(pc0_tracked)

        # Step 3: Subsequent steps with sliding history window
        for t in range(1, num_acc_steps):
            pc_t_key = f'pc{t}'
            pc_t1_key = f'pc{t + 1}'

            if pc_t_key not in all_pcs or pc_t1_key not in all_pcs:
                break

            # DEBUG LOG - show which frames are used for this step
            num_history_slots = self.num_frames - 2
            history_used = []
            for h in range(1, num_history_slots + 1):
                hist_frame_idx = t - h
                if hist_frame_idx >= 0:
                    history_used.append(f'pc{hist_frame_idx}')
                else:
                    history_used.append(f'pch{-hist_frame_idx}')

            # Build pcs_dict for this step
            # All frames already in pc1 coord, so pose is identity → pose_flow = 0
            pcs_dict_t = {
                'pc0s': all_pcs[pc_t_key],
                'pc1s': all_pcs[pc_t1_key],
                'pose_flows': [torch.zeros(all_pcs[pc_t_key].shape[1], 3, device=device) 
                              for _ in range(batch_size)],
            }
            
            # Add sliding history frames relative to pc_t
            # For t=1: pc0 becomes pch1, original pch1 becomes pch2, etc.
            # For t=2: pc1 becomes pch1, pc0 becomes pch2, original pch1 becomes pch3, etc.
            num_history_slots = self.num_frames - 2  # how many pch slots we have
            for h in range(1, num_history_slots + 1):
                hist_frame_idx = t - h  # which frame index to use
                
                if hist_frame_idx >= 0:
                    # Use pc{hist_frame_idx} as pch{h}
                    hist_key = f'pc{hist_frame_idx}'
                else:
                    # Use original pch{-hist_frame_idx} as pch{h}
                    orig_h = -hist_frame_idx
                    hist_key = f'pch{orig_h}'

                if hist_key in all_pcs:
                    pcs_dict_t[f'pch{h}s'] = all_pcs[hist_key]

            # Forward through network
            result_t = self._forward_network(pcs_dict_t)
            flows_t = result_t['flow']
            pc_t_3dvoxel_infos_lst = result_t['pc0_3dvoxel_infos_lst']
            valid_idxes_t = [e["point_idxes"] for e in pc_t_3dvoxel_infos_lst]

            # KNN interpolation for each sample in batch
            for b in range(batch_size):
                accumulated_pos = accumulated_positions[b]
                accumulated_flow = accumulated_flows[b]

                # Get pc_t's valid points (already in pc1 coord)
                pc_t_in_pc1 = all_pcs[pc_t_key][b]
                valid_mask_t = ~torch.isnan(pc_t_in_pc1[:, 0])
                pc_t_valid = pc_t_in_pc1[valid_mask_t]

                flow_t = flows_t[b]
                valid_idx_t = valid_idxes_t[b]
                pc_t_tracked = pc_t_valid[valid_idx_t]

                # Interpolate flow from pc_t's tracked points to accumulated positions
                if accumulated_pos.shape[0] > 0 and pc_t_tracked.shape[0] > 0:
                    interpolated_flow = interpolate_flow(
                        accumulated_pos, pc_t_tracked, flow_t,
                        method=self.interpolation_method,
                        k=self.knn_k
                    )

                    # Update accumulated position and flow
                    accumulated_positions[b] = accumulated_pos + interpolated_flow
                    accumulated_flows[b] = accumulated_flow + interpolated_flow

        # Package results
        result = {
            "flow": accumulated_flows,
            "pose_flow": all_pcs['pose_flow_0'],
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
            batch: Batch from dataloader containing:
                - pc0, pc1: current frame pair
                - pose0, pose1: poses for current frames
                - pch1, pch2, ...: history frames (for num_frames > 2)
                - poseh1, poseh2, ...: poses for history frames
                - For accumulated training: pc2, pc3, ... with poses
            training_mode: If True and accumulated training data available, 
                          use accumulated error training

        Returns:
            Model output dictionary with flow predictions
        """

        # Check if accumulated training should be used
        # Requires future frames (pc2, pc3, ...) in batch
        has_future_frames = 'pc2' in batch and 'pose2' in batch
        
        if training_mode and has_future_frames and self.accumulate_probs is not None:
            return self.forward_accumulated_error(batch)
        else:
            # Standard inference: use wrap_batch_pcs with history frames
            self.timer[0].start("Data Preprocess")
            pcs_dict = wrap_batch_pcs(batch, num_frames=self.num_frames)
            self.timer[0].stop()

            self.timer[1].start("3D Sparse Voxel")
            result = self._forward_network(pcs_dict)
            self.timer[1].stop()

            model_res = {
                "d_num_voxels": [result['d_num_voxels']],
                "flow": result['flow'],
                'pose_flow': result['pose_flow'],
                "pc0_valid_point_idxes": [e["point_idxes"] for e in result['pc0_3dvoxel_infos_lst']],
                "pc0_points_lst": [e["points"] for e in result['pc0_3dvoxel_infos_lst']],
                "pc1_valid_point_idxes": [e["point_idxes"] for e in result['pc1_3dvoxel_infos_lst']],
                "pc1_points_lst": [e["points"] for e in result['pc1_3dvoxel_infos_lst']],
            }
            
            if result['pch1_3dvoxel_infos_lst'] is not None:
                model_res["pch1_valid_point_idxes"] = [e["point_idxes"] for e in result['pch1_3dvoxel_infos_lst']]
                model_res["pch1_points_lst"] = [e["points"] for e in result['pch1_3dvoxel_infos_lst']]
            else:
                model_res["pch1_valid_point_idxes"] = None

            return model_res
