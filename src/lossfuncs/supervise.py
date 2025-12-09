"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Define the supervised (needed GT) loss function for training.
#
"""
import torch
import numpy as np
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)
from src.utils.av2_eval import CATEGORY_TO_INDEX, BUCKETED_METACATAGORIES

# check: https://arxiv.org/abs/2508.17054
def deltaflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    instances = res_dict['gt_instance']
    
    reassign_meta = torch.zeros_like(classes, dtype=torch.int, device=classes.device)
    for i, cats in enumerate(BUCKETED_METACATAGORIES):
        selected_classes_ids = [CATEGORY_TO_INDEX[cat] for cat in BUCKETED_METACATAGORIES[cats]]
        reassign_meta[torch.isin(classes, torch.tensor(selected_classes_ids, device=classes.device))] = i

    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)
    speed = torch.linalg.vector_norm(gt, dim=-1) / 0.1
    
    weight_loss = deflowLoss(res_dict)['loss']

    classes_loss = 0.0
    weight = [0.1, 1.0, 2.0, 2.5, 1.5] # BACKGROUND, CAR, PEDESTRIAN, WHEELED, OTHER
    for class_id in range(len(BUCKETED_METACATAGORIES)):
        mask = reassign_meta == class_id
        for loss_ in [0.1 * pts_loss[(speed < 0.4) & mask].mean(), 
                      0.4 * pts_loss[(speed >= 0.4) & (speed <= 1.0) & mask].mean(), 
                      0.5 * pts_loss[(speed > 1.0) & mask].mean()]:
            classes_loss += torch.nan_to_num(loss_, nan=0.0) * weight[class_id]

    instance_loss, cnt = 0.0, 0
    if instances is not None:
        for instance_id in torch.unique(instances):
            mask = instances == instance_id
            reassign_meta_instance = reassign_meta[mask]
            class_id = torch.mode(reassign_meta_instance, 0).values.item()
            loss_ = pts_loss[mask].mean()
            if speed[mask].mean() <= 0.4:
                continue
            instance_loss += (loss_ * torch.exp(loss_) * weight[class_id])
            cnt += 1
        instance_loss /= (cnt if cnt > 0 else 1)
    return {'loss': weight_loss + classes_loss + instance_loss}

# check: https://arxiv.org/abs/2401.16122
def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    for loss_ in [pts_loss[speed < 0.4].mean(), 
                  pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean(), 
                  pts_loss[speed > 1.0].mean()]:
        weight_loss += torch.nan_to_num(loss_, nan=0.0)

    return {'loss': weight_loss}

# designed from MambaFlow: https://github.com/SCNU-RISLAB/MambaFlow
def mambaflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    velocities = speed.cpu().numpy()

    # 计算直方图，返回每个区间的计数和区间边界
    counts, bin_edges = np.histogram(velocities, bins=100, density=False)

    # 计算每个区间的点数占总点数的比例
    total_points = len(velocities)
    proportions = counts / total_points

    # 计算每个区间的中心位置，用于绘图
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 设置占比阈值
    proportion_threshold = 0.01  # 可以根据需要调整这个值

    # 找出第一个占比小于阈值的柱子
    first_below_threshold = next((i for i, prop in enumerate(proportions) if prop < proportion_threshold), None)
    turning_speed = bin_centers[first_below_threshold]

    weight_loss = 0.0
    for loss_ in [pts_loss[speed < turning_speed].mean(), 
                  pts_loss[(speed >= turning_speed) & (speed <= 2)].mean(), 
                  pts_loss[speed > 2].mean()]:
        weight_loss += torch.nan_to_num(loss_, nan=0.0)
    return {'loss': weight_loss}

# ref from zeroflow loss class FastFlow3DDistillationLoss()
def zeroflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    # gt_speed = torch.norm(gt, dim=1, p=2) * 10.0
    gt_speed = torch.linalg.vector_norm(gt, dim=-1) * 10.0
    
    mins = torch.ones_like(gt_speed) * 0.1
    maxs = torch.ones_like(gt_speed)
    importance_scale = torch.max(mins, torch.min(1.8 * gt_speed - 0.8, maxs))
    # error = torch.norm(pred - gt, dim=1, p=2) * importance_scale
    error = error * importance_scale
    return {'loss': error.mean()}

# ref from zeroflow loss class FastFlow3DSupervisedLoss()
def ff3dLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    # error = torch.norm(pred - gt, dim=1, p=2)
    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    is_foreground_class = (classes > 0) # 0 is background, ref: FOREGROUND_BACKGROUND_BREAKDOWN
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return {'loss': error.mean()}


def bboxSpeedLoss(res_dict):
    """
    Weighted loss based on predicted flow's bounding box area (x*y projection) and speed.

    This loss function dynamically computes weights based on:
    1. Bounding box area (x*y projection) of each object instance
    2. Predicted flow speed

    Inspired by DeltaFlow's category-based weighting, but uses geometric features
    instead of category labels for more generalizable weighting.

    Statistical basis from AV2 training data analysis:
    - PEDESTRIAN: bbox_area ~0.4 m², speed ~5 m/s (small, moderate speed -> high weight)
    - WHEELED_VRU: bbox_area ~1.3 m², speed ~6 m/s (small-medium, moderate speed -> high weight)
    - CAR: bbox_area ~8.6 m², speed ~6.8 m/s (medium, moderate speed -> standard weight)
    - OTHER_VEHICLES: bbox_area ~17 m², speed ~7.3 m/s (large, moderate-high speed -> lower weight)

    res_dict should contain:
    - est_flow: predicted flow [N, 3]
    - gt_flow: ground truth flow [N, 3]
    - gt_classes: category indices [N]
    - gt_instance: instance IDs [N] (optional, for instance-level weighting)
    - pc0: point cloud [N, 3] (for computing bbox area)
    """
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    instances = res_dict.get('gt_instance', None)
    pc0 = res_dict.get('pc0', None)  # Need point cloud for bbox computation

    # Handle NaN/Inf
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    valid_mask = mask_no_nan.all(dim=-1) if mask_no_nan.dim() > 1 else mask_no_nan

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    classes_valid = classes[valid_mask] if classes is not None else None
    instances_valid = instances[valid_mask] if instances is not None else None
    pc0_valid = pc0[valid_mask] if pc0 is not None else None

    # Point-wise loss
    pts_loss = torch.linalg.vector_norm(pred_valid - gt_valid, dim=-1)

    # Predicted flow speed (m/s, assuming 10Hz sensor)
    pred_speed = torch.linalg.vector_norm(pred_valid, dim=-1) * 10.0
    gt_speed = torch.linalg.vector_norm(gt_valid, dim=-1) * 10.0

    # Speed-based weights (similar to DeltaFlow):
    # - Static objects (speed < 1 m/s): lower weight (0.2)
    # - Slow moving (1-5 m/s): medium weight (0.5)
    # - Fast moving (> 5 m/s): higher weight (1.0)
    speed_weight = torch.ones_like(pts_loss)
    speed_weight[gt_speed < 1.0] = 0.2
    speed_weight[(gt_speed >= 1.0) & (gt_speed <= 5.0)] = 0.5
    speed_weight[gt_speed > 5.0] = 1.0

    # BBox area-based weights (computed per instance if available)
    bbox_weight = torch.ones_like(pts_loss)

    if instances_valid is not None and pc0_valid is not None:
        unique_instances = torch.unique(instances_valid)
        for inst_id in unique_instances:
            if inst_id == 0:  # Skip background
                continue
            inst_mask = instances_valid == inst_id
            if inst_mask.sum() < 10:  # Skip small clusters
                continue

            # Compute bbox area (x*y projection) for this instance
            inst_pc = pc0_valid[inst_mask]
            x_range = inst_pc[:, 0].max() - inst_pc[:, 0].min()
            y_range = inst_pc[:, 1].max() - inst_pc[:, 1].min()
            bbox_area = x_range * y_range

            # BBox area-based weight:
            # Small objects (area < 1 m²): high weight 2.5 (hard to track, like pedestrians)
            # Medium objects (1-5 m²): medium-high weight 1.5 (like bikes, small cars)
            # Large objects (5-15 m²): standard weight 1.0 (like cars)
            # Very large objects (> 15 m²): lower weight 0.8 (like trucks, easier to track)
            if bbox_area < 1.0:
                inst_weight = 2.5
            elif bbox_area < 5.0:
                inst_weight = 1.5
            elif bbox_area < 15.0:
                inst_weight = 1.0
            else:
                inst_weight = 0.8

            bbox_weight[inst_mask] = inst_weight
    elif classes_valid is not None:
        # Fallback: use category-based bbox weight if no instance info
        # Based on average bbox areas from analysis
        reassign_meta = torch.zeros_like(classes_valid, dtype=torch.int, device=classes_valid.device)
        for i, cats in enumerate(BUCKETED_METACATAGORIES):
            selected_classes_ids = [CATEGORY_TO_INDEX[cat] for cat in BUCKETED_METACATAGORIES[cats]]
            reassign_meta[torch.isin(classes_valid, torch.tensor(selected_classes_ids, device=classes_valid.device))] = i

        # Weights based on average bbox areas:
        # BACKGROUND (0): 0.1, CAR (1): 1.0, PEDESTRIAN (2): 2.5, WHEELED_VRU (3): 2.0, OTHER_VEHICLES (4): 0.8
        category_bbox_weight = torch.tensor([0.1, 1.0, 2.5, 2.0, 0.8], device=classes_valid.device, dtype=pts_loss.dtype)
        bbox_weight = category_bbox_weight[reassign_meta]

    # Combined weight
    combined_weight = speed_weight * bbox_weight

    # Weighted loss
    weighted_loss = (pts_loss * combined_weight).mean()

    # Base DeFlow-style speed-bucketed loss for stability
    base_loss = 0.0
    for loss_ in [pts_loss[gt_speed < 1.0].mean(),
                  pts_loss[(gt_speed >= 1.0) & (gt_speed <= 5.0)].mean(),
                  pts_loss[gt_speed > 5.0].mean()]:
        base_loss += torch.nan_to_num(loss_, nan=0.0)

    return {'loss': weighted_loss + 0.5 * base_loss,
            'weighted_loss': weighted_loss,
            'base_loss': base_loss}
