"""
Test suite for DeFlow model
"""
import torch
import pytest
import sys
import os

# Add project root to path (same as train.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.models.deflow import DeFlow, DeFlowPP


class TestDeFlow:
    """Test cases for DeFlow model"""
    
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
    def sample_batch(self, batch_size, num_points, device):
        """Create a sample batch for testing"""
        # Create random point clouds
        pc0 = torch.randn(batch_size, num_points, 3, device=device)
        pc1 = torch.randn(batch_size, num_points, 3, device=device)
        
        # Create identity poses (4x4 transformation matrices)
        pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Add small translation to pose1
        pose1[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
        
        batch = {
            'pc0': pc0,
            'pc1': pc1,
            'pose0': pose0,
            'pose1': pose1,
        }
        return batch
    
    @pytest.fixture
    def deflow_model(self, device):
        """Create a DeFlow model instance"""
        model = DeFlow(
            voxel_size=[0.2, 0.2, 6],
            point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
            grid_feature_size=[512, 512],
            decoder_option="gru",
            num_iters=4
        ).to(device)
        model.eval()
        return model
    
    @pytest.fixture
    def deflow_linear_model(self, device):
        """Create a DeFlow model with linear decoder"""
        model = DeFlow(
            voxel_size=[0.2, 0.2, 6],
            point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
            grid_feature_size=[512, 512],
            decoder_option="linear",
            num_iters=1
        ).to(device)
        model.eval()
        return model
    
    def test_deflow_initialization(self, device):
        """Test DeFlow model initialization"""
        model = DeFlow().to(device)
        assert model is not None
        assert hasattr(model, 'embedder')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'head')
        assert hasattr(model, 'timer')
    
    def test_deflow_forward_shape(self, deflow_model, sample_batch):
        """Test DeFlow forward pass output shapes"""
        with torch.no_grad():
            output = deflow_model(sample_batch)
        
        # Check output keys
        assert 'flow' in output
        assert 'pose_flow' in output
        assert 'pc0_valid_point_idxes' in output
        assert 'pc1_valid_point_idxes' in output
        assert 'pc0_points_lst' in output
        assert 'pc1_points_lst' in output
        assert 'num_occupied_voxels' in output
        
        # Check flow shape
        assert isinstance(output['flow'], list)
        assert len(output['flow']) == sample_batch['pc0'].shape[0]
        
        # Check pose_flow shape
        assert isinstance(output['pose_flow'], list)
        assert len(output['pose_flow']) == sample_batch['pc0'].shape[0]
        
        # Check valid point indices
        assert isinstance(output['pc0_valid_point_idxes'], list)
        assert len(output['pc0_valid_point_idxes']) == sample_batch['pc0'].shape[0]
    
    def test_deflow_forward_dtype(self, deflow_model, sample_batch):
        """Test DeFlow forward pass output dtypes"""
        with torch.no_grad():
            output = deflow_model(sample_batch)
        
        # Check flow dtype
        for flow in output['flow']:
            assert isinstance(flow, torch.Tensor)
            assert flow.dtype == torch.float32
        
        # Check pose_flow dtype
        for pose_flow in output['pose_flow']:
            assert isinstance(pose_flow, torch.Tensor)
            assert pose_flow.dtype == torch.float32
    
    def test_deflow_forward_gradient(self, deflow_model, sample_batch):
        """Test DeFlow forward pass with gradient computation"""
        model = deflow_model
        model.train()
        
        output = model(sample_batch)
        
        # Check that gradients can be computed
        loss = sum([flow.mean() for flow in output['flow']])
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_deflow_linear_decoder(self, deflow_linear_model, sample_batch):
        """Test DeFlow with linear decoder"""
        with torch.no_grad():
            output = deflow_linear_model(sample_batch)
        
        assert 'flow' in output
        assert isinstance(output['flow'], list)
        assert len(output['flow']) == sample_batch['pc0'].shape[0]
    
    def test_deflow_different_batch_sizes(self, deflow_model, device):
        """Test DeFlow with different batch sizes"""
        for batch_size in [1, 2, 4]:
            num_points = 5000
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
                output = deflow_model(batch)
            
            assert len(output['flow']) == batch_size
            assert len(output['pose_flow']) == batch_size
    
    def test_deflow_different_point_counts(self, deflow_model, device):
        """Test DeFlow with different point counts per sample"""
        batch_size = 2
        point_counts = [5000, 8000]
        
        pc0_list = [torch.randn(n, 3, device=device) for n in point_counts]
        pc1_list = [torch.randn(n, 3, device=device) for n in point_counts]
        pose0_list = [torch.eye(4, device=device) for _ in range(batch_size)]
        pose1_list = [torch.eye(4, device=device) for _ in range(batch_size)]
        
        # Pad to same length for batch
        max_points = max(point_counts)
        pc0 = torch.zeros(batch_size, max_points, 3, device=device)
        pc1 = torch.zeros(batch_size, max_points, 3, device=device)
        
        for i, (p0, p1) in enumerate(zip(pc0_list, pc1_list)):
            pc0[i, :len(p0)] = p0
            pc1[i, :len(p1)] = p1
        
        pose0 = torch.stack(pose0_list, dim=0)
        pose1 = torch.stack(pose1_list, dim=0)
        
        batch = {
            'pc0': pc0,
            'pc1': pc1,
            'pose0': pose0,
            'pose1': pose1,
        }
        
        with torch.no_grad():
            output = deflow_model(batch)
        
        assert len(output['flow']) == batch_size


class TestDeFlowPP:
    """Test cases for DeFlowPP model"""
    
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
    def sample_batch_pp(self, batch_size, num_points, device):
        """Create a sample batch for DeFlowPP (needs pch1)"""
        # Create random point clouds
        pc0 = torch.randn(batch_size, num_points, 3, device=device)
        pc1 = torch.randn(batch_size, num_points, 3, device=device)
        pch1 = torch.randn(batch_size, num_points, 3, device=device)
        
        # Create identity poses
        pose0 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poseh1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Add small translations
        pose1[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
        poseh1[:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
        
        batch = {
            'pc0': pc0,
            'pc1': pc1,
            'pch1': pch1,
            'pose0': pose0,
            'pose1': pose1,
            'poseh1': poseh1,
        }
        return batch
    
    @pytest.fixture
    def deflowpp_model(self, device):
        """Create a DeFlowPP model instance"""
        model = DeFlowPP(
            voxel_size=[0.2, 0.2, 6],
            point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3],
            grid_feature_size=[512, 512],
            decoder_option="gru",
            num_iters=2,
            num_frames=3
        ).to(device)
        model.eval()
        return model
    
    def test_deflowpp_initialization(self, device):
        """Test DeFlowPP model initialization"""
        model = DeFlowPP(num_frames=3).to(device)
        assert model is not None
        assert hasattr(model, 'embedder')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'head')
        assert hasattr(model, 'timer')
        assert model.num_frames == 3
    
    def test_deflowpp_forward_shape(self, deflowpp_model, sample_batch_pp):
        """Test DeFlowPP forward pass output shapes"""
        with torch.no_grad():
            output = deflowpp_model(sample_batch_pp)
        
        # Check output keys
        assert 'flow' in output
        assert 'pose_flow' in output
        assert 'pc0_valid_point_idxes' in output
        assert 'pc1_valid_point_idxes' in output
        assert 'pch1_valid_point_idxes' in output
        assert 'pc0_points_lst' in output
        assert 'pc1_points_lst' in output
        assert 'pch1_points_lst' in output
        
        # Check flow shape
        assert isinstance(output['flow'], list)
        assert len(output['flow']) == sample_batch_pp['pc0'].shape[0]
        
        # Check pose_flow shape
        assert isinstance(output['pose_flow'], list)
        assert len(output['pose_flow']) == sample_batch_pp['pc0'].shape[0]
    
    def test_deflowpp_forward_dtype(self, deflowpp_model, sample_batch_pp):
        """Test DeFlowPP forward pass output dtypes"""
        with torch.no_grad():
            output = deflowpp_model(sample_batch_pp)
        
        # Check flow dtype
        for flow in output['flow']:
            assert isinstance(flow, torch.Tensor)
            assert flow.dtype == torch.float32
        
        # Check pose_flow dtype
        for pose_flow in output['pose_flow']:
            assert isinstance(pose_flow, torch.Tensor)
            assert pose_flow.dtype == torch.float32
    
    def test_deflowpp_forward_gradient(self, deflowpp_model, sample_batch_pp):
        """Test DeFlowPP forward pass with gradient computation"""
        model = deflowpp_model
        model.train()
        
        output = model(sample_batch_pp)
        
        # Check that gradients can be computed
        loss = sum([flow.mean() for flow in output['flow']])
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

