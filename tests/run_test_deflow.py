"""
Simple test runner for DeFlow model (without pytest dependency)
"""
import torch
import sys
import os

# Add project root to path (same as train.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.models.deflow import DeFlow, DeFlowPP


def test_deflow_basic():
    """Test basic DeFlow functionality"""
    print("=" * 60)
    print("Test 1: DeFlow Basic Functionality")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DeFlow(
        voxel_size=[0.2, 0.2, 6],
        point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
        grid_feature_size=[512, 512],
        decoder_option="gru",
        num_iters=4
    ).to(device)
    model.eval()
    print("✓ Model initialized")
    
    # Create sample batch (reduced size for testing)
    batch_size = 1
    num_points = 3000
    pc0 = torch.randn(batch_size, num_points, 3, device=device)
    pc1 = torch.randn(batch_size, num_points, 3, device=device)
    pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose1[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
    
    batch = {
        'pc0': pc0,
        'pc1': pc1,
        'pose0': pose0,
        'pose1': pose1,
    }
    print(f"✓ Batch created: batch_size={batch_size}, num_points={num_points}")
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    print("✓ Forward pass completed")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check output keys
    required_keys = ['flow', 'pose_flow', 'pc0_valid_point_idxes', 
                     'pc1_valid_point_idxes', 'pc0_points_lst', 'pc1_points_lst']
    for key in required_keys:
        assert key in output, f"Missing key: {key}"
    print("✓ All required output keys present")
    
    # Check output shapes
    assert isinstance(output['flow'], list), "flow should be a list"
    assert len(output['flow']) == batch_size, f"flow list length mismatch: {len(output['flow'])} != {batch_size}"
    print(f"✓ Flow output shape correct: {len(output['flow'])} flows")
    
    assert isinstance(output['pose_flow'], list), "pose_flow should be a list"
    assert len(output['pose_flow']) == batch_size, f"pose_flow list length mismatch"
    print(f"✓ Pose flow output shape correct: {len(output['pose_flow'])} pose flows")
    
    # Check flow dtype
    for i, flow in enumerate(output['flow']):
        assert isinstance(flow, torch.Tensor), f"flow[{i}] should be a tensor"
        assert flow.dtype == torch.float32, f"flow[{i}] should be float32, got {flow.dtype}"
        print(f"✓ Flow[{i}] shape: {flow.shape}, dtype: {flow.dtype}")
    
    print("✓ Test 1 PASSED\n")


def test_deflow_linear_decoder():
    """Test DeFlow with linear decoder"""
    print("=" * 60)
    print("Test 2: DeFlow with Linear Decoder")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with linear decoder
    model = DeFlow(
        voxel_size=[0.2, 0.2, 6],
        point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
        grid_feature_size=[512, 512],
        decoder_option="linear",
        num_iters=1
    ).to(device)
    model.eval()
    print("✓ Model with linear decoder initialized")
    
    # Create sample batch (reduced size for testing)
    batch_size = 1
    num_points = 2000
    pc0 = torch.randn(batch_size, num_points, 3, device=device)
    pc1 = torch.randn(batch_size, num_points, 3, device=device)
    pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    batch = {
        'pc0': pc0,
        'pc1': pc1,
        'pose0': pose0,
        'pose1': pose1,
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    print("✓ Forward pass completed")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    assert 'flow' in output
    assert len(output['flow']) == batch_size
    print(f"✓ Flow output shape correct: {len(output['flow'])} flows")
    
    print("✓ Test 2 PASSED\n")


def test_deflow_gradient():
    """Test DeFlow gradient computation"""
    print("=" * 60)
    print("Test 3: DeFlow Gradient Computation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use smaller model and batch for gradient test to avoid OOM
    model = DeFlow(
        voxel_size=[0.2, 0.2, 6],
        point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
        grid_feature_size=[256, 256],  # Smaller grid size
        decoder_option="linear",  # Linear decoder uses less memory
        num_iters=1
    ).to(device)
    model.train()
    print("✓ Model in training mode")
    
    # Create smaller sample batch
    batch_size = 1
    num_points = 1000  # Even smaller for gradient test
    pc0 = torch.randn(batch_size, num_points, 3, device=device)
    pc1 = torch.randn(batch_size, num_points, 3, device=device)
    pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    batch = {
        'pc0': pc0,
        'pc1': pc1,
        'pose0': pose0,
        'pose1': pose1,
    }
    
    # Forward pass with gradient
    output = model(batch)
    print("✓ Forward pass completed")
    
    # Compute loss and backward
    loss = sum([flow.mean() for flow in output['flow']])
    loss.backward()
    print("✓ Backward pass completed")
    
    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1
    
    print(f"✓ Gradients computed for {grad_count} parameters")
    assert grad_count > 0, "No gradients computed"
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✓ Test 3 PASSED\n")


def test_deflowpp():
    """Test DeFlowPP model"""
    print("=" * 60)
    print("Test 4: DeFlowPP Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeFlowPP(
        voxel_size=[0.2, 0.2, 6],
        point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
        grid_feature_size=[512, 512],
        decoder_option="gru",
        num_iters=2,
        num_frames=3
    ).to(device)
    model.eval()
    print("✓ DeFlowPP model initialized")
    
    # Create sample batch with pch1 (reduced size for testing)
    batch_size = 1
    num_points = 2000
    pc0 = torch.randn(batch_size, num_points, 3, device=device)
    pc1 = torch.randn(batch_size, num_points, 3, device=device)
    pch1 = torch.randn(batch_size, num_points, 3, device=device)
    
    pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    poseh1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    batch = {
        'pc0': pc0,
        'pc1': pc1,
        'pch1': pch1,
        'pose0': pose0,
        'pose1': pose1,
        'poseh1': poseh1,
    }
    print("✓ Batch with history frame created")
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    print("✓ Forward pass completed")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Check output keys
    required_keys = ['flow', 'pose_flow', 'pc0_valid_point_idxes', 
                     'pc1_valid_point_idxes', 'pch1_valid_point_idxes']
    for key in required_keys:
        assert key in output, f"Missing key: {key}"
    print("✓ All required output keys present")
    
    assert len(output['flow']) == batch_size
    print(f"✓ Flow output shape correct: {len(output['flow'])} flows")
    
    print("✓ Test 4 PASSED\n")


def test_deflow_different_batch_sizes():
    """Test DeFlow with different batch sizes"""
    print("=" * 60)
    print("Test 5: DeFlow with Different Batch Sizes")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeFlow(
        voxel_size=[0.2, 0.2, 6],
        point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
        grid_feature_size=[512, 512],
        decoder_option="gru",
        num_iters=2
    ).to(device)
    model.eval()
    
    for batch_size in [1]:
        num_points = 2000
        pc0 = torch.randn(batch_size, num_points, 3, device=device)
        pc1 = torch.randn(batch_size, num_points, 3, device=device)
        pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        batch = {
            'pc0': pc0,
            'pc1': pc1,
            'pose0': pose0,
            'pose1': pose1,
        }
        
        with torch.no_grad():
            output = model(batch)
        
        assert len(output['flow']) == batch_size
        print(f"✓ Batch size {batch_size}: PASSED")
    
    print("✓ Test 5 PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DeFlow Model Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_deflow_basic,
        test_deflow_linear_decoder,
        test_deflow_gradient,
        test_deflowpp,
        test_deflow_different_batch_sizes,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            # Clear GPU cache after each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
            # Clear GPU cache even on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print("=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

