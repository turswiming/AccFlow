"""
推理脚本：生成sceneflow和long term trajectories
Inference script: Generate sceneflow and long term trajectories

直接读取h5文件，使用AccFlow2Frame模型进行推理，生成sceneflow和未来4帧的long term trajectories
Directly read h5 files, use AccFlow2Frame model for inference, generate sceneflow and long term trajectories for 4 future frames
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from src.models.accflow import AccFlow2Frame, interpolate_flow, wrap_batch_pcs_future


def load_model(checkpoint_path, device):
    """加载AccFlow2Frame模型 / Load AccFlow2Frame model"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint to get hyperparameters
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get config from checkpoint
    if 'hyper_parameters' in ckpt and 'cfg' in ckpt['hyper_parameters']:
        cfg = ckpt['hyper_parameters']['cfg']
        model_cfg = cfg['model']['target']
        voxel_size = model_cfg.get('voxel_size', [0.2, 0.2, 0.2])
        point_cloud_range = model_cfg.get('point_cloud_range', [-51.2, -51.2, -2.4, 51.2, 51.2, 4.0])
        grid_feature_size = model_cfg.get('grid_feature_size', [512, 512, 32])
        num_frames = model_cfg.get('num_frames', 5)
        planes = model_cfg.get('planes', [16, 32, 64, 128, 256, 256, 128, 64, 32, 16])
        num_layer = model_cfg.get('num_layer', [2, 2, 2, 2, 2, 2, 2, 2, 2])
        decay_factor = model_cfg.get('decay_factor', 1.0)
        decoder_option = model_cfg.get('decoder_option', 'default')
        knn_k = model_cfg.get('knn_k', 3)
        interpolation_method = model_cfg.get('interpolation_method', 'knn')
        print(f"Loaded config from checkpoint")
    else:
        # Default config for AccFlow2Frame
        voxel_size = [0.2, 0.2, 0.2]
        point_cloud_range = [-51.2, -51.2, -2.4, 51.2, 51.2, 4.0]
        grid_feature_size = [512, 512, 32]
        num_frames = 5
        planes = [16, 32, 64, 128, 256, 256, 128, 64, 32, 16]
        num_layer = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        decay_factor = 1.0
        decoder_option = 'default'
        knn_k = 3
        interpolation_method = 'knn'
        print("Using default config")
    
    model = AccFlow2Frame(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        grid_feature_size=grid_feature_size,
        num_frames=num_frames,
        planes=planes,
        num_layer=num_layer,
        decay_factor=decay_factor,
        decoder_option=decoder_option,
        knn_k=knn_k,
        interpolation_method=interpolation_method,
    )
    model.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model


def read_frames_from_h5(h5_path, timestamps, start_idx, num_frames=5):
    """
    从h5文件中读取连续num_frames帧的数据
    Read consecutive num_frames from h5 file
    
    Args:
        h5_path: path to h5 file
        timestamps: sorted list of timestamps
        start_idx: starting index in timestamps
        num_frames: number of frames to read
    
    Returns:
        frames: list of dicts, each dict contains 'pc', 'pose', 'ground_mask', 'timestamp'
    """
    frames = []
    with h5py.File(h5_path, 'r') as f:
        for i in range(num_frames):
            if start_idx + i >= len(timestamps):
                break
            
            ts = str(timestamps[start_idx + i])
            if ts not in f:
                break
            
            group = f[ts]
            frame_data = {
                'pc': group['lidar'][:][:, :3].astype(np.float32),
                'pose': group['pose'][:].astype(np.float32),
                'ground_mask': group['ground_mask'][:] if 'ground_mask' in group else np.zeros(group['lidar'].shape[0], dtype=np.bool_),
                'timestamp': ts
            }
            frames.append(frame_data)
    
    return frames


def predict_sceneflows(model, frames, device):
    """
    预测4个sceneflow: F0, F1, F2, F3
    先过滤掉地面点，然后转换到pc1坐标系下，再输入模型
    Predict 4 sceneflows: F0, F1, F2, F3
    First filter ground points, then transform to pc1 coordinate system, then input to model
    
    Returns:
        flows: list of [N, 3] flow tensors (non-ground points only)
        valid_indices_list: list of valid point indices for each flow (relative to non-ground points)
        pcs_dict_all: all point clouds transformed to pc1 coordinate (with ground points, for mapping back)
        non_ground_indices: list of non-ground point indices for each frame (relative to original point cloud)
    """
    model.eval()
    
    # 先过滤地面点，然后再转换坐标系
    # First filter ground points, then transform coordinate system
    frames_no_ground = []
    non_ground_indices = []
    
    for i in range(len(frames)):
        pc = frames[i]['pc']  # [N, 3]
        gm = frames[i]['ground_mask']  # [N]
        
        # 过滤掉地面点
        # Filter ground points
        non_ground_mask = ~gm
        non_ground_idx = np.where(non_ground_mask)[0]
        pc_no_ground = pc[non_ground_mask]  # [N_no_ground, 3]
        
        # 保存非地面点索引（相对于原始点云）
        # Save non-ground point indices (relative to original point cloud)
        non_ground_indices.append(non_ground_idx)
        
        # 创建新的frame数据（只包含非地面点）
        # Create new frame data (only non-ground points)
        frame_no_ground = {
            'pc': pc_no_ground,
            'pose': frames[i]['pose'],
            'ground_mask': np.zeros(len(pc_no_ground), dtype=bool),  # 已经没有地面点了
        }
        frames_no_ground.append(frame_no_ground)
    
    # 准备batch数据（只包含非地面点）
    # Prepare batch data (only non-ground points)
    batch = {
        'pc0': torch.from_numpy(frames_no_ground[0]['pc']).float().unsqueeze(0),  # [1, N_no_ground, 3]
        'pc1': torch.from_numpy(frames_no_ground[1]['pc']).float().unsqueeze(0),
        'pose0': torch.from_numpy(frames_no_ground[0]['pose']).float().unsqueeze(0),
        'pose1': torch.from_numpy(frames_no_ground[1]['pose']).float().unsqueeze(0),
    }
    
    # 添加未来帧
    # Add future frames
    for i in range(2, len(frames_no_ground)):
        batch[f'pc{i}'] = torch.from_numpy(frames_no_ground[i]['pc']).float().unsqueeze(0)
        batch[f'pose{i}'] = torch.from_numpy(frames_no_ground[i]['pose']).float().unsqueeze(0)
    
    # 将所有帧转换到pc1坐标系下（只包含非地面点）
    # Transform all frames to pc1 coordinate system (only non-ground points)
    pcs_dict_no_ground = wrap_batch_pcs_future(batch, num_frames=len(frames_no_ground))
    
    # 同时需要保存包含地面点的pcs_dict_all，用于后续映射回原始大小
    # Also need to save pcs_dict_all with ground points for mapping back to original size
    batch_all = {
        'pc0': torch.from_numpy(frames[0]['pc']).float().unsqueeze(0),
        'pc1': torch.from_numpy(frames[1]['pc']).float().unsqueeze(0),
        'pose0': torch.from_numpy(frames[0]['pose']).float().unsqueeze(0),
        'pose1': torch.from_numpy(frames[1]['pose']).float().unsqueeze(0),
    }
    for i in range(2, len(frames)):
        batch_all[f'pc{i}'] = torch.from_numpy(frames[i]['pc']).float().unsqueeze(0)
        batch_all[f'pose{i}'] = torch.from_numpy(frames[i]['pose']).float().unsqueeze(0)
    pcs_dict_all = wrap_batch_pcs_future(batch_all, num_frames=len(frames))
    
    # 创建identity pose（因为点云已经在pc1坐标系下）
    # Create identity pose (since point clouds are already in pc1 coordinate system)
    identity_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 4, 4]
    
    flows = []
    valid_indices_list = []
    
    with torch.no_grad():
        # 预测F0: pc0 -> pc1 (使用非地面点，已经在pc1坐标系下)
        # Predict F0: pc0 -> pc1 (using non-ground points, already in pc1 coordinate system)
        pc0 = pcs_dict_no_ground['pc0s']  # [1, N_no_ground, 3] already in pc1 coord
        pc1 = pcs_dict_no_ground['pc1s']  # [1, M_no_ground, 3] already in pc1 coord
        
        result0 = model.forward_pair(pc0.to(device), pc1.to(device), identity_pose, identity_pose)
        flow0 = result0['flow'][0]  # [N_valid, 3]
        valid_idx0 = result0['pc0_valid_point_idxes'][0]  # [N_valid] relative to non-ground points
        real_flow0 = torch.zeros_like(pc0).squeeze(0).to(device)
        real_flow0[valid_idx0,:] = flow0
        flows.append(real_flow0.cpu().numpy())
        valid_indices_list.append(valid_idx0.cpu().numpy())
        
        # 预测F1, F2, F3: pc1->pc2, pc2->pc3, pc3->pc4 (使用非地面点，已经在pc1坐标系下)
        # Predict F1, F2, F3: pc1->pc2, pc2->pc3, pc3->pc4 (using non-ground points, already in pc1 coord)
        for t in range(1, min(4, len(frames_no_ground) - 1)):
            pc_t_key = f'pc{t}s'
            pc_t1_key = f'pc{t+1}s'
            
            if pc_t_key not in pcs_dict_no_ground or pc_t1_key not in pcs_dict_no_ground:
                break
            
            pc_t = pcs_dict_no_ground[pc_t_key]  # [1, N_no_ground, 3] already in pc1 coord
            pc_t1 = pcs_dict_no_ground[pc_t1_key]  # [1, M_no_ground, 3] already in pc1 coord
            
            result_t = model.forward_pair(pc_t.to(device), pc_t1.to(device), identity_pose, identity_pose)
            flow_t = result_t['flow'][0]
            valid_idx_t = result_t['pc0_valid_point_idxes'][0]
            flow_real = torch.zeros_like(pc_t).squeeze(0).to(device)
            flow_real[valid_idx_t,:] = flow_t
            flows.append(flow_real.cpu().numpy())
    # for i in range(len(flows)):
    #     print(f"flow {i} shape: {flows[i].shape}")
    # for key in pcs_dict_all:
    #     print(f"pc {key} shape: {pcs_dict_all[key][0].shape}")
    # for i in range(len(non_ground_indices)):
    #     print(f"non_ground_indices {i} shape: {non_ground_indices[i].shape}")
    # exit()
    return flows, pcs_dict_all, non_ground_indices

def compute_long_term_trajectories(pcs_dict_all, flows, non_ground_indices, device, knn_k=3):
    """
    计算long term trajectories
    所有点云都在pc1坐标系下，使用非地面点
    Compute long term trajectories
    All point clouds are in pc1 coordinate system, using non-ground points
    
    Returns:
        final_trajectory: [N, 3] 最终位置Pt+4' (non-ground points only)
        trajectory_steps: list of [N, 3] 每一步的位置 (non-ground points only)
    """
    # 获取pc0的有效点（经过网络处理的点），已经在pc1坐标系下，且是非地面点
    # Get valid points from pc0 (points processed by network), already in pc1 coord, non-ground points only
    pc0 = pcs_dict_all['pc0s'][0]  # [N, 3] in pc1 coord (all points)
    
    # 获取非地面点的索引（确保与pc0_valid在同一设备上）
    # Get non-ground point indices (ensure on same device as pc0_valid)
    non_ground_idx0 = torch.from_numpy(non_ground_indices[0]).long().to(pc0.device)
    pc0_no_ground = pc0[non_ground_idx0]  # [N_no_ground, 3]
    
    
    pt = pc0_no_ground# [N, 3] 初始点云，在pc1坐标系下，非地面点
    
    # 确保pt在device上，以便与flow0相加
    # Ensure pt is on device to add with flow0
    pt = pt.to(device)
    
    trajectory_steps = [pt.cpu().numpy()]  # Pt
    
    # 步骤1: Pt+1' = Pt + F0
    # Step 1: Pt+1' = Pt + F0
    flow0 = torch.from_numpy(flows[0]).float().to(device)
    pt1_pred = pt + flow0  # [N, 3] in pc1 coord
    trajectory_steps.append(pt1_pred.cpu().numpy())
    
    # 步骤2-4: 迭代插值和累积
    # Steps 2-4: Iterative interpolation and accumulation
    current_pos = pt1_pred.clone()
    
    for t in range(1, min(len(flows), 4)):
        # flows[t]是pc{t} -> pc{t+1}的flow，valid_indices_list[t]是相对于pc{t}的非地面点的索引
        # flows[t] is flow from pc{t} -> pc{t+1}, valid_indices_list[t] is relative to pc{t}'s non-ground points
        # 根据任务描述：在Pt+1中查找knn，使用Ft+1作为feature
        # According to task: find knn in Pt+1, use Ft+1 as feature
        # 所以我们需要使用pc{t+1}的非地面点来查找knn，但flow是在pc{t}的点上的
        # So we need to use pc{t+1}'s non-ground points for knn, but flow is on pc{t}'s points
        
        # 获取pc{t+1}用于KNN查找
        # Get pc{t+1} for KNN search
        pc_t1_key = f'pc{t+1}s'
        if pc_t1_key not in pcs_dict_all:
            break
            
        pc_t1 = pcs_dict_all[pc_t1_key][0]  # [M, 3] in pc1 coord (all points)
        valid_mask_t1 = ~torch.isnan(pc_t1[:, 0])
        pc_t1_valid = pc_t1[valid_mask_t1]  # [M_valid, 3]
        
        # 获取pc{t+1}的非地面点（用于KNN查找）
        # Get non-ground points of pc{t+1} (for KNN search)
        non_ground_idx_t1 = torch.from_numpy(non_ground_indices[t+1]).long().to(pc_t1_valid.device)
        pc_t1_no_ground = pc_t1_valid[non_ground_idx_t1]  # [M_no_ground, 3]
        
        # 获取pc{t}的非地面点和flow（flow是在pc{t}的点上的）
        # Get non-ground points and flow of pc{t} (flow is on pc{t}'s points)
        pc_t_key = f'pc{t}s'
        pc_t = pcs_dict_all[pc_t_key][0]  # [N, 3] in pc1 coord (all points)
        valid_mask_t = ~torch.isnan(pc_t[:, 0])
        pc_t_valid = pc_t[valid_mask_t]  # [N_valid, 3]
        non_ground_idx_t = torch.from_numpy(non_ground_indices[t]).long().to(pc_t_valid.device)
        pc_t_no_ground = pc_t_valid[non_ground_idx_t]  # [N_no_ground, 3]
        
        flow_real = torch.from_numpy(flows[t]).float().to(device)  # [M_valid, 3] relative to pc{t}'s non-ground points
        pc_t_tracked = pc_t_no_ground  # [M_valid, 3] pc{t}上经过网络的点
        
        # 确保在device上
        # Ensure on device
        pc_t1_no_ground = pc_t1_no_ground.to(device)
        pc_t_tracked = pc_t_tracked.to(device)
        
        # 但根据任务描述，flow应该是在pc{t+1}的点上的，所以我们需要先找到current_pos在pc{t+1}中的对应点
        # But according to task, flow should be on pc{t+1}'s points, so we need to find corresponding points first
        # 实际上，flows[t]是pc{t}->pc{t+1}的flow，所以flow是在pc{t}的点上的
        # Actually, flows[t] is flow from pc{t}->pc{t+1}, so flow is on pc{t}'s points
        # 我们需要将pc{t}的flow插值到current_pos（current_pos是预测的pc{t+1}的位置）
        # We need to interpolate pc{t}'s flow to current_pos (current_pos is predicted pc{t+1} position)
        if current_pos.shape[0] > 0 and pc_t_tracked.shape[0] > 0:
            interpolated_flow = interpolate_flow(
                current_pos, pc_t_tracked, flow_real,
                method='knn', k=knn_k
            )
            # 累积位置 / Accumulate position
            current_pos = current_pos + interpolated_flow
            trajectory_steps.append(current_pos.cpu().numpy())
        else:
            trajectory_steps.append(current_pos.cpu().numpy())
    
    return trajectory_steps




def process_frames(model, frames, device, knn_k=3):
    """
    处理5帧数据
    Process 5 frames of data
    """
    if len(frames) < 5:
        return None
    
    # 预测sceneflow / Predict sceneflow (without ground points)
    flows, pcs_dict_all, non_ground_indices = predict_sceneflows(model, frames, device)
    
    if len(flows) < 4:
        return None
    
    # 计算long term trajectories / Compute long term trajectories (without ground points)
    trajectory_steps = compute_long_term_trajectories(
        pcs_dict_all, flows, non_ground_indices, device, knn_k=knn_k
    )
    
    # 将flows映射回原始大小（包括地面点）
    # Map flows back to original size (including ground points)
    flows_mapped = []
    for i, flow in enumerate(flows):
        flows_mapped.append(flow)
    
    # 映射trajectory_steps
    origin_pc0_size = pcs_dict_all['pc0s'][0].shape[0]
    origin_pc0 = pcs_dict_all['pc0s'][0].cpu().numpy()  # [N, 3] in pc1 coord (all points)
    trajectory_steps_mapped = []
    for i, step in enumerate(trajectory_steps):
        # 第一步是pc0的非地面点，需要映射
        full_trajectory_step = origin_pc0.copy()
        full_trajectory_step[non_ground_indices[0]] = step
        trajectory_steps_mapped.append(full_trajectory_step)
    final_trajectory_mapped = trajectory_steps_mapped[-1]

    
    return {
        'flows': flows_mapped,  # Mapped to original size
        'final_trajectory': final_trajectory_mapped,  # Mapped to original size
        'trajectory_steps': trajectory_steps_mapped,  # Mapped to original size
        'timestamp': frames[0]['timestamp'],
    }


def process_single_file(args):
    """
    处理单个h5文件的worker函数
    Process single h5 file worker function
    
    Args:
        args: tuple of (h5_path, checkpoint_path, output_dir, knn_k, step_size, gpu_id)
    
    Returns:
        (h5_path, success, error_msg)
    """
    h5_path, checkpoint_path, output_dir, knn_k, step_size, gpu_id = args
    
    try:
        import sys
        # 确保错误信息能输出到stderr
        # Ensure error messages can be output to stderr
        sys.stderr.flush()
        sys.stdout.flush()
        # 设置设备（如果有多个GPU，分配不同的GPU）
        # Set device (if multiple GPUs, assign different GPU)
        if torch.cuda.is_available():
            if gpu_id is not None and gpu_id < torch.cuda.device_count():
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        # 每个进程加载自己的模型
        # Each process loads its own model
        try:
            model = load_model(checkpoint_path, device)
        except Exception as e:
            import traceback
            error_msg = f"Failed to load model: {e}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}", flush=True)
            return (h5_path, False, error_msg)
        
        # 读取所有timestamps
        # Read all timestamps
        try:
            with h5py.File(h5_path, 'r') as f:
                timestamps = sorted([int(k) for k in f.keys()])
        except Exception as e:
            import traceback
            error_msg = f"Failed to read timestamps from {h5_path}: {e}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}", flush=True)
            return (h5_path, False, error_msg)
        
        scene_id = Path(h5_path).stem
        
        # 每隔step_size帧推理一次，这样可以覆盖全部信息
        # Process every step_size frames to cover all information
        start_indices = list(range(0, len(timestamps) - 4, step_size))
        
        # 存储该场景的所有结果
        # Store all results for this scene
        scene_results = {}
        
        for start_idx in start_indices:
            # 读取5帧数据
            # Read 5 frames
            frames = read_frames_from_h5(h5_path, timestamps, start_idx, num_frames=5)
            
            if len(frames) < 5:
                continue
            
            # 处理并推理
            # Process and infer
            try:
                result = process_frames(model, frames, device, knn_k=knn_k)
            except Exception as e:
                import traceback
                error_msg = f"Failed to process frames at start_idx={start_idx} in {h5_path}: {e}\n{traceback.format_exc()}"
                print(f"[ERROR] {error_msg}", flush=True)
                # 继续处理下一个start_idx，不中断整个文件
                # Continue processing next start_idx, don't interrupt entire file
                continue
            
            if result is None:
                continue
            
            # 保存结果，key为起始timestamp
            # Save result with starting timestamp as key
            timestamp_key = frames[0]['timestamp']
            scene_results[timestamp_key] = result
        
        # 保存该场景的所有结果到新的h5文件
        # Save all results for this scene to a new h5 file
        if scene_results:
            output_path = Path(output_dir) / f"{scene_id}_trajectories.h5"
            
            try:
                with h5py.File(output_path, 'w') as f:
                    for timestamp_key, result in scene_results.items():
                        group = f.create_group(timestamp_key)
                        
                        # 保存sceneflow / Save sceneflow
                        for i, flow in enumerate(result['flows']):
                            if flow is not None:
                                group.create_dataset(f'SSL_sceneflow_{i}', data=flow, compression='gzip', compression_opts=4)
                        
                        # 保存long term trajectories / Save long term trajectories
                        group.create_dataset('SSL_long_term_trajectory', data=result['final_trajectory'], compression='gzip', compression_opts=4)
                        
                        for i, step in enumerate(result['trajectory_steps']):
                            group.create_dataset(f'SSL_trajectory_step_{i}', data=step, compression='gzip', compression_opts=4)
                
                return (h5_path, True, f"Saved {len(scene_results)} results to {output_path}")
            except Exception as e:
                import traceback
                error_msg = f"Failed to save results to {output_path}: {e}\n{traceback.format_exc()}"
                print(f"[ERROR] {error_msg}", flush=True)
                return (h5_path, False, error_msg)
        else:
            return (h5_path, False, "No results to save")
    
    except Exception as e:
        import traceback
        import sys
        error_msg = f"Error processing {h5_path}: {e}\n{traceback.format_exc()}"
        # 实时打印错误到stderr，确保能看到
        # Print error to stderr in real-time to ensure visibility
        print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
        sys.stderr.flush()
        return (h5_path, False, error_msg)


def main():
    checkpoint_path = "/workspace/OpenSceneFlow/logs/jobs/accflow-accflow2frame-0/12-09-20-59/checkpoints/04_accflow-accflow2frame.ckpt"
    data_dir = "/workspace/preprocess_step1/preprocess_step1/sensor/train"
    output_dir = "/workspace/seg_train"
    knn_k = 3  # KNN的k值，可以设置为3或更高 / KNN k value, can be set to 3 or higher
    step_size = 5  # 每隔5帧推理一次 / Process every 5 frames
    num_workers = None  # None表示使用所有CPU核心 / None means use all CPU cores
    
    # 设置多进程启动方法为spawn（CUDA需要spawn方式）
    # Set multiprocessing start method to spawn (CUDA requires spawn method)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        # If already set, ignore error
        pass
    
    # 创建输出目录 / Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有h5文件 / Get all h5 files
    data_dir = Path(data_dir)
    h5_files = sorted(data_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} h5 files")
    
    # 确定GPU数量和工作进程数
    # Determine number of GPUs and worker processes
    num_gpus = 1
    num_workers = 5
    
    print(f"Using {num_workers} worker process(es)")
    
    # 准备参数列表
    # Prepare argument list
    if num_gpus > 0:
        # 如果有GPU，循环分配GPU ID
        # If GPUs available, cycle assign GPU IDs
        gpu_ids = [0 for i in range(len(h5_files))]
    else:
        # 如果没有GPU，所有进程使用CPU
        # If no GPU, all processes use CPU
        gpu_ids = [None] * len(h5_files)
    
    args_list = [
        (str(h5_path), checkpoint_path, str(output_dir), knn_k, step_size, gpu_id)
        for h5_path, gpu_id in zip(h5_files, gpu_ids)
    ]
    
    # 使用多进程处理
    # Process using multiprocessing
    results = []
    failed_results = []
    
    with mp.Pool(processes=num_workers) as pool:
        # 使用imap_unordered可以更快地获取结果，并且能更好地处理错误
        # Use imap_unordered to get results faster and handle errors better
        try:
            for result in tqdm(
                pool.imap(process_single_file, args_list),
                total=len(h5_files),
                desc="Processing files"
            ):
                if result is not None:
                    h5_path, success, msg = result
                    results.append(result)
                    if not success:
                        # 实时打印失败信息
                        # Print failure info in real-time
                        print(f"\n[FAILED] {h5_path}: {msg}", flush=True)
                        failed_results.append(result)
                    else:
                        # 打印成功信息（可选）
                        # Print success info (optional)
                        print(f"\n[SUCCESS] {h5_path}: {msg}", flush=True)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user, shutting down pool...")
            pool.terminate()
            pool.join()
            raise
        except Exception as e:
            import traceback
            print(f"\n[FATAL ERROR] Pool error: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            pool.terminate()
            pool.join()
            raise
    
    # 打印结果摘要
    # Print result summary
    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*60}")
    
    # 打印失败的文件
    # Print failed files
    if failed_count > 0:
        print("\nFailed files:")
        for h5_path, success, error_msg in results:
            if not success:
                print(f"  {h5_path}: {error_msg}")
    
    print("Done!")


if __name__ == '__main__':
    main()
