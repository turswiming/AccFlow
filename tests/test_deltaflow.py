"""
Test suite for DeltaFlow model
"""
import torch
import pytest
import sys
import os

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.models.deltaflow import DeltaFlow


class TestDeltaFlow:
    """Test cases for DeltaFlow model"""

    @pytest.fixture
    def device(self):
        """Get device (CUDA if available, else CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def batch_size(self):
        """Batch size for testing"""
        return 2

    @pytest.fixture
    def num_points(self):
        """Number of points per point cloud"""
        return 10000

    @pytest.fixture
    def sample_batch_2frames(self, batch_size, num_points, device):
        """Create a sample batch for 2-frame DeltaFlow"""
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
        return batch

    @pytest.fixture
    def sample_batch_5frames(self, batch_size, num_points, device):
        """Create a sample batch for 5-frame DeltaFlow (with history frames)"""
        pc0 = torch.randn(batch_size, num_points, 3, device=device)
        pc1 = torch.randn(batch_size, num_points, 3, device=device)
        pch1 = torch.randn(batch_size, num_points, 3, device=device)
        pch2 = torch.randn(batch_size, num_points, 3, device=device)
        pch3 = torch.randn(batch_size, num_points, 3, device=device)

        pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poseh1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poseh2 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poseh3 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Add small translations
        pose1[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
        poseh1[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
        poseh2[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.2
        poseh3[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.3

        batch = {
            'pc0': pc0,
            'pc1': pc1,
            'pch1': pch1,
            'pch2': pch2,
            'pch3': pch3,
            'pose0': pose0,
            'pose1': pose1,
            'poseh1': poseh1,
            'poseh2': poseh2,
            'poseh3': poseh3,
        }
        return batch

    @pytest.fixture
    def deltaflow_2frame_model(self, device):
        """Create a 2-frame DeltaFlow model instance"""
        model = DeltaFlow(
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            grid_feature_size=[512, 512, 32],
            num_frames=2,
            planes=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
            num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            decay_factor=0.4,
            decoder_option="default",
        ).to(device)
        model.eval()
        return model

    @pytest.fixture
    def deltaflow_5frame_model(self, device):
        """Create a 5-frame DeltaFlow model instance"""
        model = DeltaFlow(
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            grid_feature_size=[512, 512, 32],
            num_frames=5,
            planes=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
            num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            decay_factor=0.4,
            decoder_option="default",
        ).to(device)
        model.eval()
        return model

    def test_deltaflow_initialization_2frame(self, device):
        """Test DeltaFlow 2-frame model initialization"""
        model = DeltaFlow(num_frames=2).to(device)
        assert model is not None
        assert hasattr(model, 'pc2voxel')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'flowdecoder')
        assert hasattr(model, 'timer')
        assert model.num_frames == 2

    def test_deltaflow_initialization_5frame(self, device):
        """Test DeltaFlow 5-frame model initialization"""
        model = DeltaFlow(num_frames=5).to(device)
        assert model is not None
        assert model.num_frames == 5

    def test_deltaflow_forward_2frame_shape(self, deltaflow_2frame_model, sample_batch_2frames):
        """Test DeltaFlow 2-frame forward pass output shapes"""
        with torch.no_grad():
            output = deltaflow_2frame_model(sample_batch_2frames)

        # Check output keys
        assert 'flow' in output
        assert 'pose_flow' in output
        assert 'pc0_valid_point_idxes' in output
        assert 'pc1_valid_point_idxes' in output
        assert 'pc0_points_lst' in output
        assert 'pc1_points_lst' in output
        assert 'd_num_voxels' in output

        # Check flow shape
        batch_size = sample_batch_2frames['pc0'].shape[0]
        assert isinstance(output['flow'], list)
        assert len(output['flow']) == batch_size

        # Check pose_flow shape
        assert isinstance(output['pose_flow'], list)
        assert len(output['pose_flow']) == batch_size

        # Check valid point indices
        assert isinstance(output['pc0_valid_point_idxes'], list)
        assert len(output['pc0_valid_point_idxes']) == batch_size

    def test_deltaflow_forward_5frame_shape(self, deltaflow_5frame_model, sample_batch_5frames):
        """Test DeltaFlow 5-frame forward pass output shapes"""
        with torch.no_grad():
            output = deltaflow_5frame_model(sample_batch_5frames)

        # Check output keys
        assert 'flow' in output
        assert 'pose_flow' in output

        # Check flow shape (5 frames still outputs single flow from pc0 to pc1)
        batch_size = sample_batch_5frames['pc0'].shape[0]
        assert isinstance(output['flow'], list)
        assert len(output['flow']) == batch_size

    def test_deltaflow_forward_dtype(self, deltaflow_2frame_model, sample_batch_2frames):
        """Test DeltaFlow forward pass output dtypes"""
        with torch.no_grad():
            output = deltaflow_2frame_model(sample_batch_2frames)

        # Check flow dtype
        for flow in output['flow']:
            assert isinstance(flow, torch.Tensor)
            assert flow.dtype == torch.float32

        # Check pose_flow dtype
        for pose_flow in output['pose_flow']:
            assert isinstance(pose_flow, torch.Tensor)
            assert pose_flow.dtype == torch.float32

    def test_deltaflow_flow_dimension(self, deltaflow_2frame_model, sample_batch_2frames):
        """Test that flow vectors are 3D"""
        with torch.no_grad():
            output = deltaflow_2frame_model(sample_batch_2frames)

        for flow in output['flow']:
            assert flow.shape[-1] == 3, "Flow vectors should be 3D"

    def test_deltaflow_forward_gradient(self, deltaflow_2frame_model, sample_batch_2frames):
        """Test DeltaFlow forward pass with gradient computation"""
        model = deltaflow_2frame_model
        model.train()

        output = model(sample_batch_2frames)

        # Check that gradients can be computed
        loss = sum([flow.mean() for flow in output['flow']])
        loss.backward()

        # Check that gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        assert grad_count > 0, "Should have some gradients"

    def test_deltaflow_different_batch_sizes(self, deltaflow_2frame_model, device):
        """Test DeltaFlow with different batch sizes"""
        # Clear GPU cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for batch_size in [1, 2]:  # Reduced from [1, 2, 4] to avoid OOM
            num_points = 3000  # Reduced from 5000
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
                output = deltaflow_2frame_model(batch)

            assert len(output['flow']) == batch_size
            assert len(output['pose_flow']) == batch_size

            # Clean up
            del batch, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_deltaflow_deflow_decoder(self, device):
        """Test DeltaFlow with deflow decoder option"""
        # Clear GPU cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = DeltaFlow(
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
            grid_feature_size=[512, 512, 32],
            num_frames=2,
            decoder_option="deflow",
        ).to(device)
        model.eval()

        batch_size = 1  # Reduced from 2 to avoid OOM
        num_points = 3000  # Reduced from 5000
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

        assert 'flow' in output
        assert len(output['flow']) == batch_size

        # Clean up
        del model, batch, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_deltaflow_decay_factor(self, device):
        """Test DeltaFlow with different decay factors"""
        for decay_factor in [0.2, 0.4, 1.0]:
            model = DeltaFlow(
                voxel_size=[0.2, 0.2, 0.2],
                point_cloud_range=[-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                grid_feature_size=[512, 512, 32],
                num_frames=2,
                decay_factor=decay_factor,
            ).to(device)
            assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
