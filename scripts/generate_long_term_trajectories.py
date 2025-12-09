"""
Generate long-term trajectories using AccFlow model.

This script:
1. Loads a trained AccFlow checkpoint
2. Reads training data with 5 consecutive frames
3. Performs inference to predict flow for frames 0->1, 1->2, 2->3, 3->4
4. Accumulates flows to generate long-term trajectories
5. Saves results in compressed format (.npz)

Usage:
    python scripts/generate_long_term_trajectories.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_dir /path/to/train_data \
        --output_dir /path/to/output \
        --num_samples 100
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from src.models.accflow import AccFlow, wrap_batch_pcs_future
from src.dataset import HDF5DatasetFutureFrames


def collate_fn_pad(batch):
    """Collate function with padding for variable-length point clouds."""
    data_dict = {}

    # Find max points in this batch for each frame
    num_frames = 5
    max_points = {}
    for i in range(num_frames):
        pc_key = f'pc{i}'
        if pc_key in batch[0]:
            max_points[pc_key] = max(b[pc_key].shape[0] for b in batch)

    for key in batch[0].keys():
        if key.startswith('pc') and not key.endswith('dynamic'):
            # Pad point clouds with NaN
            pc_key = key
            if pc_key in max_points:
                max_len = max_points[pc_key]
                padded = []
                for b in batch:
                    pc = b[pc_key]
                    if len(pc) < max_len:
                        pad = np.full((max_len - len(pc), 3), np.nan, dtype=np.float32)
                        pc = np.concatenate([pc, pad], axis=0)
                    padded.append(torch.from_numpy(pc).float())
                data_dict[key] = torch.stack(padded, dim=0)
        elif key.startswith('pose'):
            data_dict[key] = torch.stack([torch.from_numpy(b[key]).float() for b in batch], dim=0)
        elif key.startswith('gm'):
            # Ground mask - pad with False
            max_len = max_points.get(f'pc{key[2:]}', max(b[key].shape[0] for b in batch))
            padded = []
            for b in batch:
                gm = b[key]
                if len(gm) < max_len:
                    pad = np.zeros(max_len - len(gm), dtype=np.bool_)
                    gm = np.concatenate([gm, pad], axis=0)
                padded.append(torch.from_numpy(gm))
            data_dict[key] = torch.stack(padded, dim=0)
        elif key in ['scene_id', 'timestamp', 'eval_flag']:
            data_dict[key] = [b[key] for b in batch]
        elif key == 'flow':
            # Pad flow with NaN
            max_len = max_points.get('pc0', max(b[key].shape[0] for b in batch))
            padded = []
            for b in batch:
                flow = b[key]
                if len(flow) < max_len:
                    pad = np.full((max_len - len(flow), 3), np.nan, dtype=np.float32)
                    flow = np.concatenate([flow, pad], axis=0)
                padded.append(torch.from_numpy(flow).float())
            data_dict[key] = torch.stack(padded, dim=0)
        elif key in ['flow_is_valid', 'flow_category_indices']:
            max_len = max_points.get('pc0', max(b[key].shape[0] for b in batch))
            padded = []
            for b in batch:
                arr = b[key]
                if len(arr) < max_len:
                    pad = np.zeros(max_len - len(arr), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=0)
                padded.append(torch.from_numpy(arr))
            data_dict[key] = torch.stack(padded, dim=0)

    return data_dict


def generate_trajectories(model, batch, device):
    """
    Generate long-term trajectories by predicting flow for consecutive frame pairs.

    Returns:
        trajectories: dict with accumulated positions and flows for each sample
    """
    model.eval()
    batch_size = len(batch['pose0'])

    # Move batch to device
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value

    with torch.no_grad():
        # Prepare point clouds
        pcs_dict = wrap_batch_pcs_future(batch_device, num_frames=5)

        # Pre-voxelize all frames
        voxel_cache = model.pc2voxel.voxelize_all_frames(pcs_dict, num_frames=5)

        results = []

        for b in range(batch_size):
            # Get pc0 points
            pc0 = pcs_dict['pc0s'][b]
            valid_mask = ~torch.isnan(pc0[:, 0])
            pc0_valid = pc0[valid_mask]

            # Store trajectory: positions at each time step
            # trajectory[t] = position of pc0 points at time t
            trajectory_positions = [pc0_valid.cpu().numpy()]  # t=0
            trajectory_flows = []  # flow from t to t+1

            # Accumulated position (starts at pc0)
            accumulated_pos = pc0_valid.clone()

            # Get valid indices from first forward pass
            result_0 = model.forward_single_with_cache(pcs_dict, voxel_cache, time_idx=0)
            valid_idx_0 = result_0['src_valid_point_idxes'][b]

            # Only track points that go through the network
            pc0_tracked = pc0_valid[valid_idx_0]
            accumulated_pos = pc0_tracked + result_0['flow'][b]

            trajectory_positions_tracked = [pc0_tracked.cpu().numpy()]
            trajectory_flows.append(result_0['flow'][b].cpu().numpy())
            trajectory_positions.append(accumulated_pos.cpu().numpy())

            # Predict flow for frames 1->2, 2->3, 3->4
            for t in range(1, 4):
                result_t = model.forward_single_with_cache(pcs_dict, voxel_cache, time_idx=t)
                flow_t = result_t['flow'][b]
                valid_idx_t = result_t['src_valid_point_idxes'][b]

                # Get pc_t valid points for interpolation
                pc_t = pcs_dict[f'pc{t}s'][b]
                valid_mask_t = ~torch.isnan(pc_t[:, 0])
                pc_t_valid = pc_t[valid_mask_t][valid_idx_t]

                # Interpolate flow to accumulated positions
                from src.models.accflow import interpolate_flow
                if accumulated_pos.shape[0] > 0 and pc_t_valid.shape[0] > 0:
                    interpolated_flow = interpolate_flow(
                        accumulated_pos, pc_t_valid, flow_t,
                        method='knn', k=3
                    )
                    accumulated_pos = accumulated_pos + interpolated_flow
                    trajectory_flows.append(interpolated_flow.cpu().numpy())
                    trajectory_positions.append(accumulated_pos.cpu().numpy())

            results.append({
                'scene_id': batch['scene_id'][b],
                'timestamp': batch['timestamp'][b],
                'pc0_original': pc0_valid.cpu().numpy(),
                'pc0_tracked': pc0_tracked.cpu().numpy(),
                'valid_indices': valid_idx_0.cpu().numpy(),
                'trajectory_positions': trajectory_positions,  # [t0, t1, t2, t3, t4]
                'trajectory_flows': trajectory_flows,  # [f01, f12, f23, f34]
                # Also save the real future point clouds for comparison
                'pc1_real': pcs_dict['pc1s'][b][~torch.isnan(pcs_dict['pc1s'][b][:, 0])].cpu().numpy(),
                'pc2_real': pcs_dict['pc2s'][b][~torch.isnan(pcs_dict['pc2s'][b][:, 0])].cpu().numpy(),
                'pc3_real': pcs_dict['pc3s'][b][~torch.isnan(pcs_dict['pc3s'][b][:, 0])].cpu().numpy(),
                'pc4_real': pcs_dict['pc4s'][b][~torch.isnan(pcs_dict['pc4s'][b][:, 0])].cpu().numpy(),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate long-term trajectories using AccFlow')
    parser.add_argument('--checkpoint', type=str,
                        default='/workspace/OpenSceneFlow/logs/jobs/accflow-accflow-0/12-08-03-38/checkpoints/last.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='/workspace/preprocess_step1/preprocess_step1/sensor/train',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/OpenSceneFlow/outputs/trajectories',
                        help='Output directory for trajectories')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process (0 for all)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from: {args.checkpoint}")

    # Load checkpoint to get hyperparameters
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'hyper_parameters' in ckpt and 'cfg' in ckpt['hyper_parameters']:
        cfg = ckpt['hyper_parameters']['cfg']
        model_cfg = cfg['model']['target']
        voxel_size = model_cfg.get('voxel_size', [0.2, 0.2, 0.2])
        point_cloud_range = model_cfg.get('point_cloud_range', [-51.2, -51.2, -2.4, 51.2, 51.2, 4.0])
        grid_feature_size = model_cfg.get('grid_feature_size', [512, 512, 32])
        num_frames = model_cfg.get('num_frames', 5)
        planes = model_cfg.get('planes', [16, 32, 64, 128, 256, 256, 128, 64, 32, 16])
        num_layer = model_cfg.get('num_layer', [2, 2, 2, 2, 2, 2, 2, 2, 2])
        decay_factor = model_cfg.get('decay_factor', 0.4)
        decoder_option = model_cfg.get('decoder_option', 'time_aware')
        knn_k = model_cfg.get('knn_k', 3)
        interpolation_method = model_cfg.get('interpolation_method', 'knn')
        print(f"Loaded config from checkpoint: voxel_size={voxel_size}, grid_feature_size={grid_feature_size}")
    else:
        # Default config
        voxel_size = [0.2, 0.2, 0.2]
        point_cloud_range = [-51.2, -51.2, -2.4, 51.2, 51.2, 4.0]
        grid_feature_size = [512, 512, 32]
        num_frames = 5
        planes = [16, 32, 64, 128, 256, 256, 128, 64, 32, 16]
        num_layer = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        decay_factor = 0.4
        decoder_option = 'time_aware'
        knn_k = 3
        interpolation_method = 'knn'
        print("Using default config")

    model = AccFlow(
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
        accumulate_probs=None,  # Use full accumulation for inference
    )
    model.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = HDF5DatasetFutureFrames(
        directory=args.data_dir,
        n_frames=5,
        transform=None,
        eval=False,
    )
    print(f"Dataset size: {len(dataset)}")

    # Determine number of samples
    num_samples = args.num_samples if args.num_samples > 0 else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"Processing {num_samples} samples")

    # Create dataloader
    from torch.utils.data import DataLoader, Subset
    subset_indices = list(range(num_samples))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_pad,
    )

    # Process samples
    all_results = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating trajectories")):
        try:
            results = generate_trajectories(model, batch, device)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Save results
    output_file = output_dir / 'long_term_trajectories.npz'
    print(f"Saving {len(all_results)} trajectories to {output_file}")

    # Convert to arrays for saving
    save_dict = {
        'num_samples': len(all_results),
        'scene_ids': np.array([r['scene_id'] for r in all_results], dtype=object),
        'timestamps': np.array([r['timestamp'] for r in all_results]),
    }

    # Save each sample's data with index prefix
    for i, result in enumerate(all_results):
        save_dict[f'{i:06d}_pc0_tracked'] = result['pc0_tracked']
        save_dict[f'{i:06d}_valid_indices'] = result['valid_indices']

        # Save trajectory positions (list of arrays)
        for t, pos in enumerate(result['trajectory_positions']):
            save_dict[f'{i:06d}_pos_t{t}'] = pos

        # Save trajectory flows
        for t, flow in enumerate(result['trajectory_flows']):
            save_dict[f'{i:06d}_flow_t{t}to{t+1}'] = flow

        # Save real point clouds for comparison
        save_dict[f'{i:06d}_pc1_real'] = result['pc1_real']
        save_dict[f'{i:06d}_pc2_real'] = result['pc2_real']
        save_dict[f'{i:06d}_pc3_real'] = result['pc3_real']
        save_dict[f'{i:06d}_pc4_real'] = result['pc4_real']

    np.savez_compressed(output_file, **save_dict)
    print(f"Results saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
