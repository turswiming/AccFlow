"""
Test suite for HDF5Dataset
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

from src.dataset import HDF5Dataset, ToTensor, RandomJitter, RandomFlip, RandomHeight


class TestHDF5Dataset:
    """Test cases for HDF5Dataset"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample HDF5 files and index for testing"""
        # Create sample data index
        scene_id = "test_scene_001"
        timestamps = [1000, 1001, 1002, 1003, 1004]
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
                
                # Create flow data (only for first 4 timestamps, last one has no flow)
                if ts < timestamps[-1]:
                    flow = np.random.randn(num_points, 3).astype(np.float32) * 0.1
                    grp.create_dataset('flow', data=flow)
                    grp.create_dataset('flow_is_valid', data=np.ones(num_points, dtype=np.bool_))
        
        return temp_dir, scene_id, timestamps
    
    def test_hdf5_dataset_initialization(self, sample_data):
        """Test HDF5Dataset initialization"""
        temp_dir, _, _ = sample_data
        dataset = HDF5Dataset(directory=temp_dir)
        
        assert dataset.directory == temp_dir
        assert dataset.eval_index == False
        assert dataset.ssl_label is None
        assert dataset.history_frames == 0  # n_frames=2 by default
        assert dataset.transform is None
        assert len(dataset.data_index) == 5
    
    def test_hdf5_dataset_length(self, sample_data):
        """Test HDF5Dataset __len__ method"""
        temp_dir, _, _ = sample_data
        dataset = HDF5Dataset(directory=temp_dir)
        
        assert len(dataset) == 5
    
    def test_hdf5_dataset_getitem_basic(self, sample_data):
        """Test HDF5Dataset __getitem__ with basic settings"""
        temp_dir, scene_id, timestamps = sample_data
        dataset = HDF5Dataset(directory=temp_dir, n_frames=2)
        
        # Test getting first item
        data = dataset[0]
        
        assert 'scene_id' in data
        assert 'timestamp' in data
        assert 'eval_flag' in data
        assert 'pc0' in data
        assert 'gm0' in data
        assert 'pose0' in data
        assert 'pc1' in data
        assert 'gm1' in data
        assert 'pose1' in data
        
        assert data['scene_id'] == scene_id
        assert data['timestamp'] == timestamps[0]
        assert isinstance(data['pc0'], np.ndarray)
        assert data['pc0'].shape[1] == 3  # x, y, z
        assert isinstance(data['gm0'], np.ndarray)
        assert data['gm0'].dtype == np.bool_
        assert data['pose0'].shape == (4, 4)
    
    def test_hdf5_dataset_n_frames(self, sample_data):
        """Test HDF5Dataset with different n_frames"""
        temp_dir, _, _ = sample_data
        
        # Test with n_frames=3 (should have history frame)
        dataset = HDF5Dataset(directory=temp_dir, n_frames=3)
        data = dataset[1]  # Use index 1 to have history
        
        assert 'pch1' in data
        assert 'gmh1' in data
        assert 'poseh1' in data
        assert dataset.history_frames == 1
    
    def test_hdf5_dataset_with_transform(self, sample_data):
        """Test HDF5Dataset with transform"""
        temp_dir, _, _ = sample_data
        from torchvision import transforms
        
        transform = transforms.Compose([
            ToTensor()
        ])
        dataset = HDF5Dataset(directory=temp_dir, transform=transform)
        
        data = dataset[0]
        
        # After ToTensor, numpy arrays should be converted to tensors
        assert isinstance(data['pc0'], torch.Tensor)
        assert isinstance(data['gm0'], torch.Tensor)
        assert isinstance(data['pose0'], torch.Tensor)
        # scene_id and timestamp should remain as original types
        assert isinstance(data['scene_id'], str)
        assert isinstance(data['timestamp'], (int, np.integer))
    
    def test_hdf5_dataset_valid_index(self, sample_data):
        """Test HDF5Dataset valid_index method"""
        temp_dir, scene_id, _ = sample_data
        dataset = HDF5Dataset(directory=temp_dir, n_frames=2)
        
        # Test valid index
        eval_flag, valid_idx = dataset.valid_index(0)
        assert eval_flag == False
        assert valid_idx == 0
        
        # Test with last index (should be adjusted if needed)
        eval_flag, valid_idx = dataset.valid_index(4)
        assert valid_idx <= 4
    
    def test_hdf5_dataset_eval_mode(self, sample_data):
        """Test HDF5Dataset in eval mode"""
        temp_dir, scene_id, timestamps = sample_data
        
        # Create eval index (subset of data)
        eval_index = [(scene_id, timestamps[0]), (scene_id, timestamps[2])]
        with open(os.path.join(temp_dir, 'index_eval.pkl'), 'wb') as f:
            pickle.dump(eval_index, f)
        
        dataset = HDF5Dataset(directory=temp_dir, eval=True)
        
        assert dataset.eval_index == True
        assert len(dataset) == 2
        
        # Test getting item in eval mode
        data = dataset[0]
        assert data['eval_flag'] == True
        assert 'eval_mask' in data
    
    def test_hdf5_dataset_flow_data(self, sample_data):
        """Test HDF5Dataset with flow data"""
        temp_dir, _, _ = sample_data
        dataset = HDF5Dataset(directory=temp_dir, vis_name=['flow'])
        
        data = dataset[0]
        
        # Should have flow data for first 4 timestamps
        if 'flow' in data:
            assert isinstance(data['flow'], np.ndarray)
            assert data['flow'].shape[1] == 3
    
    def test_hdf5_dataset_multiple_items(self, sample_data):
        """Test HDF5Dataset with multiple items"""
        temp_dir, _, _ = sample_data
        dataset = HDF5Dataset(directory=temp_dir)
        
        # Test getting multiple items
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            assert 'pc0' in data
            assert 'pc1' in data
            assert data['pc0'].shape[1] == 3
    
    def test_hdf5_dataset_history_frames_boundary(self, sample_data):
        """Test HDF5Dataset with history frames at boundary"""
        temp_dir, _, _ = sample_data
        dataset = HDF5Dataset(directory=temp_dir, n_frames=3)
        
        # Test at first index (should handle boundary)
        data = dataset[0]
        # Should still work even at boundary
        assert 'pc0' in data
        assert 'pc1' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




