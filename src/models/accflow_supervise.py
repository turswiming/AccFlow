"""
# Created: 2025-12-XX
# AccFlowSupervise - Supervised version of AccumulateErrorFlow
# Based on DeltaFlow architecture with accumulated error training using GT flow

# Key features:
# 1. Uses history frames (pch1, pch2, ...) like DeltaFlow for richer temporal context
# 2. Training uses accumulated error computation with GT flow to maximize accumulated error
# 3. Inference only requires history frames + pc0 + pc1 (same as DeltaFlow), minimizes error
"""

import torch
import torch.nn as nn
import dztimer

from .basic import cal_pose0to1, wrap_batch_pcs
from .basic.sparse_encoder import MinkUNet, SparseVoxelNet
from .basic.decoder import SparseGRUHead
from .basic.flow4d_module import Point_head
from .accflow import interpolate_flow


class AccFlowSupervise(nn.Module):
    """
    Supervised AccumulateErrorFlow based on DeltaFlow architecture.
    
    Training mode (accumulated error):
    - All frames are first transformed to pc1 coordinate system
    - Sliding window predicts flow for consecutive frame pairs with history context
    - Errors are accumulated through KNN interpolation
    - GT flow is also accumulated to compute accumulated error
    - Loss maximizes accumulated error (to learn error accumulation patterns)
    
    Inference mode:
    - Standard 2-frame prediction with history context (same as DeltaFlow)
    - Minimizes single-frame error
    
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
            print('[LOG] AccFlowSupervise Param detail: voxel_size = {}, pseudo_dims = {}, num_frames={}'.format(
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
                return i + 1
        return len(self.accumulate_probs)

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
        Also transform GT flows to pc1 coordinate system.
        
        Args:
            batch: Input batch containing all point clouds, poses, and GT flows
            num_acc_steps: Number of accumulation steps (determines how many future frames needed)
        
        Returns:
            Dictionary with transformed point clouds and GT flows:
                - 'pc0', 'pc1', 'pc2', ... : transformed point clouds [B, N, 3]
                - 'pch1', 'pch2', ... : transformed history frames [B, N, 3]
                - 'gt_flow_0', 'gt_flow_1', ... : transformed GT flows [B, N, 3]
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
                
                # Transform GT flow for pc0 → pc1
                # Note: batch['flow'] contains GT flow for pc0->pc1 (already in pc1 coordinate after pose transformation)
                if 'flow' in batch and batch['flow'][b] is not None:
                    gt_flow_0 = batch['flow'][b]  # GT flow for pc0->pc1
                    # GT flow is already relative to pc1 coordinate, but we need to account for pose transformation
                    # The flow in batch is typically: pc1 - pc0 (in original coordinates)
                    # After transforming pc0 to pc1 coord, the flow should be: pc1 - pc0_transformed
                    # But typically batch['flow'] is already the relative flow, so we just rotate it
                    gt_flow_0_transformed = gt_flow_0 @ pose_0to1[:3, :3].T
                    if b == 0:
                        transformed['gt_flow_0'] = [gt_flow_0_transformed]
                    else:
                        transformed['gt_flow_0'].append(gt_flow_0_transformed)
                
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
                        
                        # Try to get GT flow for pc{i-1} → pc{i}
                        # Check multiple possible keys
                        flow_key_variants = [
                            f'flow_{i-1}_{i}',  # flow_1_2, flow_2_3, etc.
                            f'flow{i-1}',  # flow1, flow2, etc.
                        ]
                        gt_flow_i = None
                        for flow_key in flow_key_variants:
                            if flow_key in batch and batch[flow_key][b] is not None:
                                gt_flow_i = batch[flow_key][b]
                                break
                        
                        # If no direct GT flow, try to compute from point cloud difference
                        if gt_flow_i is None and i > 1:
                            # Approximate GT flow as: pc_i - pc_{i-1} (in pc1 coordinate)
                            pc_prev_key = f'pc{i-1}'
                            if pc_prev_key in transformed:
                                if isinstance(transformed[pc_prev_key], list):
                                    pc_prev = transformed[pc_prev_key][b]
                                else:
                                    pc_prev = transformed[pc_prev_key][b]
                                # This is approximate - actual GT flow would be better
                                # For now, we'll skip this and only use available GT flows
                                gt_flow_i = None
                        
                        if gt_flow_i is not None:
                            gt_flow_i_transformed = gt_flow_i @ pose_i_to_1[:3, :3].T
                            gt_flow_key = f'gt_flow_{i-1}'
                            if b == 0:
                                transformed[gt_flow_key] = [gt_flow_i_transformed]
                            else:
                                transformed[gt_flow_key].append(gt_flow_i_transformed)
                
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
        
        # Stack all lists into tensors
        for key in transformed:
            if isinstance(transformed[key], list):
                transformed[key] = torch.stack(transformed[key], dim=0)
        
        transformed['pose_flow_0'] = pose_flow_0
        return transformed

    def forward_accumulated_error_supervised(self, batch):
        """
        Training forward with accumulated error using GT flow.
        
        Process:
        1. Transform all frames to pc1 coordinate system
        2. Predict F0: forward(pc0, pc1) -> flow_0
        3. Accumulate predicted flow: pred_flow_acc = flow_0
        4. Accumulate GT flow: gt_flow_acc = gt_flow_0
        5. For t = 1 to num_acc_steps-1:
           a. Predict F_t: forward(pc_t, pc_{t+1}) -> flow_t
           b. Interpolate flow_t to accumulated positions using KNN
           c. Accumulate predicted flow: pred_flow_acc += interpolated_F_t
           d. Accumulate GT flow: gt_flow_acc += interpolated_GT_flow_t
        6. Compute accumulated error: ||pred_flow_acc - gt_flow_acc||
        7. Maximize accumulated error (negative loss)
        
        Returns:
            Dictionary with accumulated flows and error information
        """
        batch_size = len(batch['pose0'])
        device = batch['pc0'].device

        sampled_acc_steps = self._sample_num_accumulate_steps()
        target_frame_idx = sampled_acc_steps + 1
        
        # Check how many future frames we actually have
        max_future = 1
        while f'pc{max_future + 1}' in batch and f'pose{max_future + 1}' in batch:
            max_future += 1
        num_acc_steps = min(target_frame_idx, max_future)

        # Step 1: Transform all frames and GT flows to pc1 coordinate system
        all_pcs = self._transform_all_to_pc1_coord(batch, num_acc_steps)

        # Step 2: First step pc0 → pc1 with original history frames
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

        # Initialize accumulated positions, predicted flows, and GT flows
        accumulated_positions = []
        accumulated_pred_flows = []
        accumulated_gt_flows = []
        accumulated_gt_positions = []
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
            accumulated_pred_flow = flow_0.clone()
            
            # Initialize GT flow accumulation
            if 'gt_flow_0' in all_pcs:
                gt_flow_0 = all_pcs['gt_flow_0'][b]
                gt_flow_0_valid = gt_flow_0[valid_mask]
                accumulated_gt_flow = gt_flow_0_valid[valid_idx_0].clone()
            else:
                # If no GT flow, use zeros
                accumulated_gt_flow = torch.zeros_like(accumulated_pred_flow)

            # Also track GT flow accumulated position (should match predicted if GT is perfect)
            if 'gt_flow_0' in all_pcs:
                accumulated_gt_pos = pc0_tracked + accumulated_gt_flow
            else:
                accumulated_gt_pos = accumulated_pos.clone()
            
            accumulated_positions.append(accumulated_pos)
            accumulated_pred_flows.append(accumulated_pred_flow)
            accumulated_gt_flows.append(accumulated_gt_flow)
            accumulated_gt_positions.append(accumulated_gt_pos)
            all_valid_idxes.append(valid_idx_0)
            all_pc0_points.append(pc0_tracked)

        # Step 3: Subsequent steps with sliding history window
        for t in range(1, num_acc_steps):
            pc_t_key = f'pc{t}'
            pc_t1_key = f'pc{t + 1}'

            if pc_t_key not in all_pcs or pc_t1_key not in all_pcs:
                break

            # Build pcs_dict for this step
            pcs_dict_t = {
                'pc0s': all_pcs[pc_t_key],
                'pc1s': all_pcs[pc_t1_key],
                'pose_flows': [torch.zeros(all_pcs[pc_t_key].shape[1], 3, device=device) 
                              for _ in range(batch_size)],
            }
            
            # Add sliding history frames relative to pc_t
            num_history_slots = self.num_frames - 2
            for h in range(1, num_history_slots + 1):
                hist_frame_idx = t - h
                
                if hist_frame_idx >= 0:
                    hist_key = f'pc{hist_frame_idx}'
                else:
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
                accumulated_pred_flow = accumulated_pred_flows[b]
                accumulated_gt_flow = accumulated_gt_flows[b]
                accumulated_gt_pos = accumulated_gt_positions[b]

                # Get pc_t's valid points (already in pc1 coord)
                pc_t_in_pc1 = all_pcs[pc_t_key][b]
                valid_mask_t = ~torch.isnan(pc_t_in_pc1[:, 0])
                pc_t_valid = pc_t_in_pc1[valid_mask_t]

                flow_t = flows_t[b]
                valid_idx_t = valid_idxes_t[b]
                pc_t_tracked = pc_t_valid[valid_idx_t]

                # Interpolate predicted flow from pc_t's tracked points to accumulated positions
                interpolated_pred_flow = interpolate_flow(
                    accumulated_pos, pc_t_tracked, flow_t,
                    method=self.interpolation_method,
                    k=self.knn_k
                )

                # Update accumulated position and predicted flow
                accumulated_positions[b] = accumulated_pos + interpolated_pred_flow
                accumulated_pred_flows[b] = accumulated_pred_flow + interpolated_pred_flow
                
                # Interpolate GT flow if available
                gt_flow_key = f'gt_flow_{t}'
                if gt_flow_key in all_pcs:
                    gt_flow_t = all_pcs[gt_flow_key][b]
                    gt_flow_t_valid = gt_flow_t[valid_mask_t]
                    gt_flow_t_tracked = gt_flow_t_valid[valid_idx_t]
                    
                    # Interpolate GT flow to GT accumulated positions (not predicted accumulated positions)
                    # This ensures GT flow accumulation is consistent with GT trajectory
                    interpolated_gt_flow = interpolate_flow(
                        accumulated_gt_pos, pc_t_tracked, gt_flow_t_tracked,
                        method=self.interpolation_method,
                        k=self.knn_k
                    )
                    accumulated_gt_flows[b] = accumulated_gt_flow + interpolated_gt_flow
                    accumulated_gt_positions[b] = accumulated_gt_pos + interpolated_gt_flow

        # Package results
        result = {
            "flow": accumulated_pred_flows,  # Accumulated predicted flow
            "gt_flow_accumulated": accumulated_gt_flows,  # Accumulated GT flow
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
            batch: Batch from dataloader
            training_mode: If True, use accumulated error training with GT flow
        
        Returns:
            Model output dictionary
        """
        self.timer[0].start("Data Preprocess")
        
        if training_mode and self.num_frames > 2:
            # Training mode: accumulated error with GT flow
            return self.forward_accumulated_error_supervised(batch)
        else:
            # Inference mode: standard 2-frame prediction (same as DeltaFlow)
            pcs_dict = wrap_batch_pcs(batch, num_frames=self.num_frames)
            self.timer[0].stop()

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

