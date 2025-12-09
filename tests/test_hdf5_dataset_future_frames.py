"""
Test suite for HDF5DatasetFutureFrames
Tests the variant that uses future frames instead of history frames when n_frames > 2
"""
import torch
import pytest
import sys
import os
import h5py
import pickle
import numpy as np
import tempfile
import shutil

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.dataset import HDF5DatasetFutureFrames, ToTensor


class TestHDF5DatasetFutureFrames:
    """Test cases for HDF5DatasetFutureFrames"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample HDF5 files and index for testing with enough frames for 7-frame test"""
        # Create sample data index with enough frames (need at least 7 frames for n_frames=7)
        scene_id = "test_scene_001"
        timestamps = list(range(1000, 1010))  # 10 frames to support up to 7 frames
        data_index = [(scene_id, ts) for ts in timestamps]
        
        # Save index_total.pkl
        with open(os.path.join(temp_dir, 'index_total.pkl'), 'wb') as f:
            pickle.dump(data_index, f)
        
        # Create HDF5 file with sample data
        h5_path = os.path.join(temp_dir, f'{scene_id}.h5')
        with h5py.File(h5_path, 'w') as f:
            for ts in timestamps:
                key = str(ts)
                grp = f.create_group(key)
                
                # Create sample lidar data (N points, 4 features: x, y, z, intensity)
                num_points = 1000
                lidar_data = np.random.randn(num_points, 4).astype(np.float32)
                grp.create_dataset('lidar', data=lidar_data)
                
                # Create ground mask (boolean array)
                ground_mask = np.random.rand(num_points) > 0.3  # 70% non-ground
                grp.create_dataset('ground_mask', data=ground_mask.astype(np.bool_))
                
                # Create pose (4x4 transformation matrix)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, 3] = np.random.randn(3) * 0.1  # Small translation
                grp.create_dataset('pose', data=pose)
                
                # Create flow data (for all frames except last)
                if ts < timestamps[-1]:
                    flow = np.random.randn(num_points, 3).astype(np.float32) * 0.1
                    grp.create_dataset('flow', data=flow)
                    grp.create_dataset('flow_is_valid', data=np.ones(num_points, dtype=np.bool_))
        
        return temp_dir, scene_id, timestamps
    
    def test_future_frames_initialization(self, sample_data):
        """Test HDF5DatasetFutureFrames initialization"""
        temp_dir, _, _ = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=2)
        
        assert dataset.directory == temp_dir
        assert dataset.eval_index == False
        assert dataset.ssl_label is None
        assert dataset.future_frames == 0  # n_frames=2 by default
        assert dataset.n_frames == 2
        assert dataset.transform is None
        assert len(dataset.data_index) == 10
    
    def test_future_frames_n_frames_2(self, sample_data):
        """Test HDF5DatasetFutureFrames with n_frames=2 (should only have pc0 and pc1)"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=2)
        
        data = dataset[0]
        
        # Should have pc0 and pc1
        assert 'pc0' in data
        assert 'pc1' in data
        assert 'gm0' in data
        assert 'gm1' in data
        assert 'pose0' in data
        assert 'pose1' in data
        
        # Should NOT have future frames
        assert 'pc2' not in data
        assert 'pc3' not in data
        
        assert data['scene_id'] == scene_id
        assert data['timestamp'] == timestamps[0]
    
    def test_future_frames_n_frames_3(self, sample_data):
        """Test HDF5DatasetFutureFrames with n_frames=3 (should have pc0, pc1, pc2)"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=3)
        
        assert dataset.future_frames == 1
        assert dataset.n_frames == 3
        
        data = dataset[0]
        
        # Should have pc0, pc1, and pc2
        assert 'pc0' in data
        assert 'pc1' in data
        assert 'pc2' in data
        assert 'gm0' in data
        assert 'gm1' in data
        assert 'gm2' in data
        assert 'pose0' in data
        assert 'pose1' in data
        assert 'pose2' in data
        
        # Should NOT have pc3
        assert 'pc3' not in data
        
        # Verify shapes
        assert data['pc0'].shape[1] == 3
        assert data['pc1'].shape[1] == 3
        assert data['pc2'].shape[1] == 3
        assert data['pose0'].shape == (4, 4)
        assert data['pose1'].shape == (4, 4)
        assert data['pose2'].shape == (4, 4)
    
    def test_future_frames_n_frames_4(self, sample_data):
        """Test HDF5DatasetFutureFrames with n_frames=4 (should have pc0, pc1, pc2, pc3)"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=4)
        
        assert dataset.future_frames == 2
        assert dataset.n_frames == 4
        
        data = dataset[0]
        
        # Should have pc0, pc1, pc2, and pc3
        assert 'pc0' in data
        assert 'pc1' in data
        assert 'pc2' in data
        assert 'pc3' in data
        assert 'gm0' in data
        assert 'gm1' in data
        assert 'gm2' in data
        assert 'gm3' in data
        assert 'pose0' in data
        assert 'pose1' in data
        assert 'pose2' in data
        assert 'pose3' in data
        
        # Should NOT have pc4
        assert 'pc4' not in data
    
    def test_future_frames_n_frames_5(self, sample_data):
        """Test HDF5DatasetFutureFrames with n_frames=5 (should have pc0, pc1, pc2, pc3, pc4)"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=5)
        
        assert dataset.future_frames == 3
        assert dataset.n_frames == 5
        
        data = dataset[0]
        
        # Should have pc0 through pc4
        for i in range(5):
            assert f'pc{i}' in data, f"Missing pc{i}"
            assert f'gm{i}' in data, f"Missing gm{i}"
            assert f'pose{i}' in data, f"Missing pose{i}"
        
        # Should NOT have pc5
        assert 'pc5' not in data
    
    def test_future_frames_n_frames_6(self, sample_data):
        """Test HDF5DatasetFutureFrames with n_frames=6 (should have pc0, pc1, pc2, pc3, pc4, pc5)"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=6)
        
        assert dataset.future_frames == 4
        assert dataset.n_frames == 6
        
        data = dataset[0]
        
        # Should have pc0 through pc5
        for i in range(6):
            assert f'pc{i}' in data, f"Missing pc{i}"
            assert f'gm{i}' in data, f"Missing gm{i}"
            assert f'pose{i}' in data, f"Missing pose{i}"
        
        # Should NOT have pc6
        assert 'pc6' not in data
    
    def test_future_frames_n_frames_7(self, sample_data):
        """Test HDF5DatasetFutureFrames with n_frames=7 (should have pc0, pc1, pc2, pc3, pc4, pc5, pc6)"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=7)
        
        assert dataset.future_frames == 5
        assert dataset.n_frames == 7
        
        data = dataset[0]
        
        # Should have pc0 through pc6
        for i in range(7):
            assert f'pc{i}' in data, f"Missing pc{i}"
            assert f'gm{i}' in data, f"Missing gm{i}"
            assert f'pose{i}' in data, f"Missing pose{i}"
        
        # Should NOT have pc7
        assert 'pc7' not in data
    
    def test_future_frames_boundary_handling(self, sample_data):
        """Test HDF5DatasetFutureFrames handles boundary cases correctly"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=7)
        
        # Test at boundary (should adjust index to ensure enough future frames)
        # With 10 frames and n_frames=7, we need at least 6 future frames after pc0
        # So max valid index is 10 - 7 = 3
        data = dataset[3]  # Should work
        
        # Should still have all required frames (may use last frame if needed)
        assert 'pc0' in data
        assert 'pc1' in data
    
    def test_future_frames_valid_index(self, sample_data):
        """Test HDF5DatasetFutureFrames valid_index method"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=5)
        
        # Test valid index
        eval_flag, valid_idx = dataset.valid_index(0)
        assert eval_flag == False
        assert valid_idx == 0
        
        # Test with index that requires adjustment (need 4 future frames + pc1 = 5 total)
        # With 10 frames, max valid index is 10 - 5 = 5
        eval_flag, valid_idx = dataset.valid_index(8)
        assert valid_idx <= 5  # Should be adjusted
    
    def test_future_frames_with_transform(self, sample_data):
        """Test HDF5DatasetFutureFrames with transform"""
        temp_dir, _, _ = sample_data
        from torchvision import transforms
        
        transform = transforms.Compose([
            ToTensor()
        ])
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=4, transform=transform)
        
        data = dataset[0]
        
        # After ToTensor, numpy arrays should be converted to tensors
        assert isinstance(data['pc0'], torch.Tensor)
        assert isinstance(data['pc1'], torch.Tensor)
        assert isinstance(data['pc2'], torch.Tensor)
        assert isinstance(data['pc3'], torch.Tensor)
        assert isinstance(data['gm0'], torch.Tensor)
        assert isinstance(data['pose0'], torch.Tensor)
    
    def test_future_frames_multiple_samples(self, sample_data):
        """Test HDF5DatasetFutureFrames with multiple samples"""
        temp_dir, _, _ = sample_data
        dataset = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=5)
        
        # Test getting multiple items
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            assert 'pc0' in data
            assert 'pc1' in data
            assert 'pc2' in data
            assert 'pc3' in data
            assert 'pc4' in data
            assert data['pc0'].shape[1] == 3
    
    def test_future_frames_consistency(self, sample_data):
        """Test that future frames are consistent across different n_frames values"""
        temp_dir, scene_id, timestamps = sample_data
        
        # Test that pc0 and pc1 are the same regardless of n_frames
        dataset_2 = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=2)
        dataset_5 = HDF5DatasetFutureFrames(directory=temp_dir, n_frames=5)
        
        data_2 = dataset_2[0]
        data_5 = dataset_5[0]
        
        # pc0 and pc1 should be the same
        np.testing.assert_array_equal(data_2['pc0'], data_5['pc0'])
        np.testing.assert_array_equal(data_2['pc1'], data_5['pc1'])
        np.testing.assert_array_equal(data_2['pose0'], data_5['pose0'])
        np.testing.assert_array_equal(data_2['pose1'], data_5['pose1'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




