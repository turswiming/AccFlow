"""
Test suite for AccFlow model (Accumulated Error Flow)
"""
import torch
import pytest
import sys
import os

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.models.accflow import (
    AccFlow,
    wrap_batch_pcs_future,
    knn_interpolate_flow,
    three_nn_interpolate_flow,
    rbf_interpolate_flow,
    idw_interpolate_flow,
    interpolate_flow,
    INTERPOLATION_METHODS,
    HAS_PYTORCH3D,
    HAS_POINTNET_OPS,
)


class TestInterpolationMethods:
    """Test cases for flow interpolation methods"""

    @pytest.fixture
    def device(self):
        """Get device (CUDA if available, else CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def sample_points(self, device):
        """Create sample points for interpolation testing"""
        # Query points: 100 random points
        query_points = torch.randn(100, 3, device=device)
        # Reference points: 200 random points
        ref_points = torch.randn(200, 3, device=device)
        # Reference flow: random flow vectors
        ref_flow = torch.randn(200, 3, device=device)
        return query_points, ref_points, ref_flow

    def test_knn_interpolate_flow_shape(self, sample_points):
        """Test KNN interpolation output shape"""
        query_points, ref_points, ref_flow = sample_points
        result = knn_interpolate_flow(query_points, ref_points, ref_flow, k=3)
        assert result.shape == (100, 3)

    def test_knn_interpolate_flow_device(self, sample_points):
        """Test KNN interpolation preserves device"""
        query_points, ref_points, ref_flow = sample_points
        result = knn_interpolate_flow(query_points, ref_points, ref_flow, k=3)
        assert result.device == query_points.device

    def test_knn_interpolate_flow_dtype(self, sample_points):
        """Test KNN interpolation preserves dtype"""
        query_points, ref_points, ref_flow = sample_points
        result = knn_interpolate_flow(query_points, ref_points, ref_flow, k=3)
        assert result.dtype == ref_flow.dtype

    def test_knn_interpolate_flow_different_k(self, sample_points):
        """Test KNN interpolation with different k values"""
        query_points, ref_points, ref_flow = sample_points
        for k in [1, 3, 5, 10]:
            result = knn_interpolate_flow(query_points, ref_points, ref_flow, k=k)
            assert result.shape == (100, 3)

    def test_three_nn_interpolate_flow_shape(self, sample_points):
        """Test three_nn interpolation output shape"""
        query_points, ref_points, ref_flow = sample_points
        result = three_nn_interpolate_flow(query_points, ref_points, ref_flow)
        assert result.shape == (100, 3)

    def test_rbf_interpolate_flow_shape(self, sample_points):
        """Test RBF interpolation output shape"""
        query_points, ref_points, ref_flow = sample_points
        result = rbf_interpolate_flow(query_points, ref_points, ref_flow, sigma=0.5, max_neighbors=16)
        assert result.shape == (100, 3)

    def test_rbf_interpolate_flow_different_sigma(self, sample_points):
        """Test RBF interpolation with different sigma values"""
        query_points, ref_points, ref_flow = sample_points
        for sigma in [0.1, 0.5, 1.0, 2.0]:
            result = rbf_interpolate_flow(query_points, ref_points, ref_flow, sigma=sigma)
            assert result.shape == (100, 3)

    def test_idw_interpolate_flow_shape(self, sample_points):
        """Test IDW interpolation output shape"""
        query_points, ref_points, ref_flow = sample_points
        result = idw_interpolate_flow(query_points, ref_points, ref_flow, power=2, k=8)
        assert result.shape == (100, 3)

    def test_idw_interpolate_flow_different_power(self, sample_points):
        """Test IDW interpolation with different power values"""
        query_points, ref_points, ref_flow = sample_points
        for power in [1, 2, 3]:
            result = idw_interpolate_flow(query_points, ref_points, ref_flow, power=power)
            assert result.shape == (100, 3)

    def test_interpolate_flow_unified_interface(self, sample_points):
        """Test unified interpolation interface"""
        query_points, ref_points, ref_flow = sample_points
        for method in INTERPOLATION_METHODS.keys():
            result = interpolate_flow(query_points, ref_points, ref_flow, method=method)
            assert result.shape == (100, 3)

    def test_interpolate_flow_invalid_method(self, sample_points):
        """Test unified interface raises error for invalid method"""
        query_points, ref_points, ref_flow = sample_points
        with pytest.raises(ValueError):
            interpolate_flow(query_points, ref_points, ref_flow, method='invalid_method')

    def test_interpolation_same_point_identity(self, device):
        """Test that interpolating at same points returns similar flow"""
        ref_points = torch.randn(50, 3, device=device)
        ref_flow = torch.randn(50, 3, device=device)
        # Query at same positions as reference
        query_points = ref_points.clone()

        result = knn_interpolate_flow(query_points, ref_points, ref_flow, k=1)
        # With k=1, should return exact flow values
        assert torch.allclose(result, ref_flow, atol=1e-5)

    def test_interpolation_gradient(self, sample_points):
        """Test that interpolation supports gradient computation"""
        query_points, ref_points, ref_flow = sample_points
        ref_flow_grad = ref_flow.clone().requires_grad_(True)

        result = knn_interpolate_flow(query_points, ref_points, ref_flow_grad, k=3)
        loss = result.mean()
        loss.backward()

        assert ref_flow_grad.grad is not None

    @pytest.mark.skipif(not HAS_PYTORCH3D, reason="pytorch3d not installed")
    def test_pytorch3d_knn_available(self, sample_points):
        """Test that pytorch3d KNN is used when available"""
        query_points, ref_points, ref_flow = sample_points
        # Large point cloud should trigger pytorch3d path
        large_query = torch.randn(1000, 3, device=query_points.device)
        result = knn_interpolate_flow(large_query, ref_points, ref_flow, k=3)
        assert result.shape == (1000, 3)

    @pytest.mark.skipif(not HAS_POINTNET_OPS, reason="torch-points-kernels not installed")
    def test_pointnet_ops_available(self, sample_points):
        """Test that torch-points-kernels is used when available"""
        query_points, ref_points, ref_flow = sample_points
        result = three_nn_interpolate_flow(query_points, ref_points, ref_flow)
        assert result.shape == (100, 3)


class TestWrapBatchPcsFuture:
    """Test cases for wrap_batch_pcs_future function"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def sample_batch_5frames(self, device):
        """Create a sample batch with 5 future frames"""
        batch_size = 2
        num_points = 1000

        batch = {}
        for i in range(5):
            batch[f'pc{i}'] = torch.randn(batch_size, num_points, 3, device=device)
            batch[f'pose{i}'] = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            # Add small translations
            batch[f'pose{i}'][:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1 * i

        return batch

    def test_wrap_batch_pcs_future_keys(self, sample_batch_5frames):
        """Test that wrap_batch_pcs_future returns correct keys"""
        pcs_dict = wrap_batch_pcs_future(sample_batch_5frames, num_frames=5)

        # Check point cloud keys
        for i in range(5):
            assert f'pc{i}s' in pcs_dict

        # Check pose_flows
        assert 'pose_flows' in pcs_dict
        for i in range(4):
            assert i in pcs_dict['pose_flows']

    def test_wrap_batch_pcs_future_shapes(self, sample_batch_5frames):
        """Test output shapes from wrap_batch_pcs_future"""
        pcs_dict = wrap_batch_pcs_future(sample_batch_5frames, num_frames=5)

        batch_size = sample_batch_5frames['pc0'].shape[0]
        num_points = sample_batch_5frames['pc0'].shape[1]

        # Check point cloud shapes
        for i in range(5):
            assert pcs_dict[f'pc{i}s'].shape == (batch_size, num_points, 3)


class TestAccFlow:
    """Test cases for AccFlow model"""

    @pytest.fixture
    def device(self):
        """Get device (CUDA if available, else CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def num_points(self):
        return 5000

    @pytest.fixture
    def sample_batch_5frames(self, batch_size, num_points, device):
        """Create a sample batch for 5-frame AccFlow"""
        batch = {}
        for i in range(5):
            batch[f'pc{i}'] = torch.randn(batch_size, num_points, 3, device=device)
            batch[f'pose{i}'] = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            batch[f'pose{i}'][:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1 * i

        batch['scene_id'] = ['test_scene'] * batch_size
        return batch

    @pytest.fixture
    def accflow_model(self, device):
        """Create an AccFlow model instance"""
        model = AccFlow(
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            grid_feature_size=[512, 512, 32],
            num_frames=5,
            planes=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
            num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            decay_factor=0.4,
            decoder_option="default",
            knn_k=3,
            interpolation_method="knn",
        ).to(device)
        model.eval()
        return model

    def test_accflow_initialization(self, device):
        """Test AccFlow model initialization"""
        model = AccFlow(num_frames=5).to(device)
        assert model is not None
        assert hasattr(model, 'pc2voxel')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'flowdecoder')
        assert hasattr(model, 'timer')
        assert model.num_frames == 5
        assert model.knn_k == 3
        assert model.interpolation_method == "knn"

    def test_accflow_initialization_different_interpolation(self, device):
        """Test AccFlow initialization with different interpolation methods"""
        for method in ['knn', 'three_nn', 'rbf', 'idw']:
            model = AccFlow(
                num_frames=5,
                interpolation_method=method,
            ).to(device)
            assert model.interpolation_method == method

    def test_accflow_forward_shape(self, accflow_model, sample_batch_5frames):
        """Test AccFlow forward pass output shapes"""
        with torch.no_grad():
            output = accflow_model(sample_batch_5frames)

        # Check output keys
        assert 'flow' in output
        assert 'pose_flow' in output
        assert 'pc0_valid_point_idxes' in output
        assert 'pc0_points_lst' in output

        # Check flow shape
        batch_size = sample_batch_5frames['pc0'].shape[0]
        assert isinstance(output['flow'], list)
        assert len(output['flow']) == batch_size

    def test_accflow_forward_single(self, accflow_model, sample_batch_5frames):
        """Test AccFlow forward_single for different time indices"""
        pcs_dict = wrap_batch_pcs_future(sample_batch_5frames, num_frames=5)

        with torch.no_grad():
            for time_idx in range(4):  # 0->1, 1->2, 2->3, 3->4
                output = accflow_model.forward_single(pcs_dict, time_idx=time_idx)
                assert 'flow' in output
                assert output['time_idx'] == time_idx

    def test_accflow_forward_dtype(self, accflow_model, sample_batch_5frames):
        """Test AccFlow forward pass output dtypes"""
        with torch.no_grad():
            output = accflow_model(sample_batch_5frames)

        for flow in output['flow']:
            assert isinstance(flow, torch.Tensor)
            assert flow.dtype == torch.float32

    def test_accflow_flow_dimension(self, accflow_model, sample_batch_5frames):
        """Test that flow vectors are 3D"""
        with torch.no_grad():
            output = accflow_model(sample_batch_5frames)

        for flow in output['flow']:
            assert flow.shape[-1] == 3, "Flow vectors should be 3D"

    def test_accflow_forward_gradient(self, accflow_model, sample_batch_5frames):
        """Test AccFlow forward pass with gradient computation"""
        model = accflow_model
        model.train()

        output = model(sample_batch_5frames)

        # Check that gradients can be computed
        loss = sum([flow.mean() for flow in output['flow']])
        loss.backward()

        # Check that gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        assert grad_count > 0, "Should have some gradients"

    def test_accflow_different_batch_sizes(self, accflow_model, device):
        """Test AccFlow with different batch sizes"""
        for batch_size in [1, 2]:
            num_points = 3000
            batch = {}
            for i in range(5):
                batch[f'pc{i}'] = torch.randn(batch_size, num_points, 3, device=device)
                batch[f'pose{i}'] = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

            batch['scene_id'] = ['test'] * batch_size

            with torch.no_grad():
                output = accflow_model(batch)

            assert len(output['flow']) == batch_size

    def test_accflow_time_embedding(self, device):
        """Test that time embedding is properly applied"""
        model = AccFlow(num_frames=5).to(device)

        # Check time embedding dimensions
        assert model.pc2voxel.time_embed_dim == 4  # num_frames - 1
        assert model.pc2voxel.time_proj.in_features == 4
        assert model.pc2voxel.time_proj.out_features == model.pc2voxel.num_feature

    def test_accflow_training_mode(self, device):
        """Test AccFlow training mode (accumulated error)"""
        model = AccFlow(
            num_frames=5,
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            grid_feature_size=[512, 512, 32],
        ).to(device)
        model.train()

        batch_size = 1
        num_points = 2000
        batch = {}
        for i in range(5):
            batch[f'pc{i}'] = torch.randn(batch_size, num_points, 3, device=device)
            batch[f'pose{i}'] = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        batch['scene_id'] = ['test'] * batch_size

        with torch.no_grad():
            # Test training mode forward
            output = model(batch, training_mode=True)

        assert 'flow' in output
        assert 'accumulated_target_frame' in output
        assert output['accumulated_target_frame'] == 4  # Last frame index

    def test_accflow_deflow_decoder(self, device):
        """Test AccFlow with deflow decoder option"""
        model = AccFlow(
            num_frames=5,
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            grid_feature_size=[512, 512, 32],
            decoder_option="deflow",
        ).to(device)
        model.eval()

        batch_size = 1
        num_points = 3000
        batch = {}
        for i in range(5):
            batch[f'pc{i}'] = torch.randn(batch_size, num_points, 3, device=device)
            batch[f'pose{i}'] = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        batch['scene_id'] = ['test'] * batch_size

        with torch.no_grad():
            output = model(batch)

        assert 'flow' in output
        assert len(output['flow']) == batch_size


class TestAccFlowEncoder:
    """Test cases for AccFlowEncoder"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_accflow_encoder_time_embedding(self, device):
        """Test AccFlowEncoder time embedding"""
        from src.models.accflow import AccFlowEncoder

        encoder = AccFlowEncoder(
            voxel_size=[0.2, 0.2, 0.2],
            pseudo_image_dims=[512, 512, 32],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            feat_channels=16,
            num_frames=5,
        ).to(device)

        # Check time embedding setup
        assert encoder.time_embed_dim == 4  # num_frames - 1
        assert encoder.time_proj.in_features == 4
        assert encoder.time_proj.out_features == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
