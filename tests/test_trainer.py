"""
Test suite for ModelWrapper (trainer.py)
"""
import torch
import pytest
import sys
import os

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from omegaconf import OmegaConf, DictConfig
from src.trainer import ModelWrapper


def create_minimal_config(model_name='deltaflow', loss_fn='deflowLoss', num_frames=2):
    """Create a minimal config for testing"""
    cfg = OmegaConf.create({
        'model': {
            'name': model_name,
            'target': {
                '_target_': f'src.models.DeltaFlow' if model_name == 'deltaflow' else f'src.models.AccFlow',
                'voxel_size': [0.2, 0.2, 0.2],
                'point_cloud_range': [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                'num_frames': num_frames,
                'planes': [16, 32, 64, 128, 256, 256, 128, 64, 32, 16],
                'num_layer': [2, 2, 2, 2, 2, 2, 2, 2, 2],
                'decay_factor': 0.4,
                'decoder_option': 'default',
            }
        },
        'loss_fn': loss_fn,
        'batch_size': 1,
        'lr': 2e-4,
        'epochs': 3,
        'num_frames': num_frames,
        'optimizer': {
            'name': 'Adam',
            'lr': 2e-4,
        },
        'add_seloss': None,
        'checkpoint': None,
        'leaderboard_version': 2,
        'supervised_flag': True,
        'save_res': False,
        'res_name': 'test',
        'data_mode': 'train',
        'dataset_path': '/tmp/test',
    })

    if model_name == 'accflow':
        cfg.model.target['knn_k'] = 3
        cfg.model.target['interpolation_method'] = 'knn'

    return cfg


def create_accflow_config_with_seloss():
    """Create AccFlow config with self-supervised loss"""
    cfg = create_minimal_config(model_name='accflow', loss_fn='accflowLoss', num_frames=5)
    cfg.add_seloss = {
        'chamfer_dis': 1.0,
        'dynamic_chamfer_dis': 1.0,
        'static_flow_loss': 1.0,
        'cluster_based_pc0pc1': 1.0,
    }
    return cfg


class TestModelWrapperInitialization:
    """Test cases for ModelWrapper initialization"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_wrapper_initialization_deltaflow(self):
        """Test ModelWrapper initialization with DeltaFlow"""
        cfg = create_minimal_config(model_name='deltaflow')
        wrapper = ModelWrapper(cfg)

        assert wrapper is not None
        assert wrapper.model is not None
        assert wrapper.model.__class__.__name__ == 'DeltaFlow'
        assert wrapper.loss_fn is not None
        assert wrapper.batch_size == 1
        assert wrapper.lr == 2e-4
        assert wrapper.epochs == 3

    def test_wrapper_initialization_accflow(self):
        """Test ModelWrapper initialization with AccFlow"""
        cfg = create_minimal_config(model_name='accflow', num_frames=5)
        wrapper = ModelWrapper(cfg)

        assert wrapper is not None
        assert wrapper.model is not None
        assert wrapper.model.__class__.__name__ == 'AccFlow'
        assert wrapper.num_frames == 5

    def test_wrapper_with_seloss_config(self):
        """Test ModelWrapper with self-supervised loss config"""
        cfg = create_accflow_config_with_seloss()
        wrapper = ModelWrapper(cfg)

        assert wrapper.add_seloss is not None
        assert 'chamfer_dis' in wrapper.add_seloss
        assert wrapper.cfg_loss_name == 'accflowLoss'

    def test_wrapper_optimizer_config(self):
        """Test ModelWrapper optimizer configuration"""
        cfg = create_minimal_config()
        wrapper = ModelWrapper(cfg)

        assert wrapper.optimizer is not None
        assert wrapper.optimizer.name == 'Adam'
        assert wrapper.optimizer.lr == 2e-4


class TestModelWrapperConfigureOptimizers:
    """Test cases for configure_optimizers method"""

    def test_configure_adam_optimizer(self):
        """Test Adam optimizer configuration"""
        cfg = create_minimal_config()
        wrapper = ModelWrapper(cfg)

        optimizers = wrapper.configure_optimizers()

        assert 'optimizer' in optimizers
        assert isinstance(optimizers['optimizer'], torch.optim.Adam)

    def test_configure_adamw_optimizer(self):
        """Test AdamW optimizer configuration"""
        cfg = create_minimal_config()
        cfg.optimizer.name = 'AdamW'
        cfg.optimizer.weight_decay = 1e-4
        wrapper = ModelWrapper(cfg)

        optimizers = wrapper.configure_optimizers()

        assert 'optimizer' in optimizers
        assert isinstance(optimizers['optimizer'], torch.optim.AdamW)

    def test_configure_optimizer_with_scheduler(self):
        """Test optimizer with learning rate scheduler"""
        cfg = create_minimal_config()
        cfg.optimizer.scheduler = {
            'name': 'StepLR',
            'step_size': 1,
            'gamma': 0.1,
        }
        wrapper = ModelWrapper(cfg)

        # Need to mock trainer for scheduler
        class MockTrainer:
            max_epochs = 10
        wrapper.trainer = MockTrainer()

        optimizers = wrapper.configure_optimizers()

        assert 'optimizer' in optimizers
        assert 'lr_scheduler' in optimizers


class TestModelWrapperTrainingStep:
    """Test cases for training_step method"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def sample_batch_deltaflow(self, device):
        """Create a sample batch for DeltaFlow training"""
        batch_size = 1
        num_points = 3000

        batch = {
            'pc0': torch.randn(batch_size, num_points, 3, device=device),
            'pc1': torch.randn(batch_size, num_points, 3, device=device),
            'pose0': torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
            'pose1': torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
            'flow': torch.randn(batch_size, num_points, 3, device=device),
            'flow_category_indices': torch.zeros(batch_size, num_points, dtype=torch.long, device=device),
        }
        batch['pose1'][:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1
        return batch

    @pytest.fixture
    def sample_batch_accflow(self, device):
        """Create a sample batch for AccFlow training"""
        batch_size = 1
        num_points = 2000

        batch = {}
        for i in range(5):
            batch[f'pc{i}'] = torch.randn(batch_size, num_points, 3, device=device)
            batch[f'pose{i}'] = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            batch[f'pose{i}'][:, :3, 3] = torch.randn(batch_size, 3, device=device) * 0.1 * i

        batch['scene_id'] = ['test_scene'] * batch_size
        return batch

    def test_training_step_deltaflow(self, sample_batch_deltaflow, device):
        """Test training step with DeltaFlow"""
        cfg = create_minimal_config(model_name='deltaflow')
        wrapper = ModelWrapper(cfg)
        wrapper = wrapper.to(device)

        loss = wrapper.training_step(sample_batch_deltaflow, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_training_step_accflow_standard(self, sample_batch_accflow, device):
        """Test training step with AccFlow (standard mode)"""
        cfg = create_minimal_config(model_name='accflow', num_frames=5)
        wrapper = ModelWrapper(cfg)
        wrapper = wrapper.to(device)

        # Add required fields for standard training
        sample_batch_accflow['flow'] = torch.randn(1, 2000, 3, device=device)
        sample_batch_accflow['flow_category_indices'] = torch.zeros(1, 2000, dtype=torch.long, device=device)

        loss = wrapper.training_step(sample_batch_accflow, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_training_step_accflow_accumulated_error(self, sample_batch_accflow, device):
        """Test training step with AccFlow (accumulated error mode)"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cfg = create_accflow_config_with_seloss()
        wrapper = ModelWrapper(cfg)
        wrapper = wrapper.to(device)

        loss = wrapper.training_step(sample_batch_accflow, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestModelWrapperMetrics:
    """Test cases for metrics handling"""

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized"""
        cfg = create_minimal_config()
        wrapper = ModelWrapper(cfg)

        assert wrapper.metrics is not None
        assert hasattr(wrapper.metrics, 'step')
        assert hasattr(wrapper.metrics, 'normalize')
        assert hasattr(wrapper.metrics, 'print')


class TestModelWrapperLossFunctions:
    """Test cases for different loss functions"""

    def test_deflowloss_import(self):
        """Test deflowLoss import"""
        cfg = create_minimal_config(loss_fn='deflowLoss')
        wrapper = ModelWrapper(cfg)

        assert wrapper.loss_fn is not None
        assert callable(wrapper.loss_fn)

    def test_seflowloss_import(self):
        """Test seflowLoss import"""
        cfg = create_minimal_config(loss_fn='seflowLoss')
        cfg.add_seloss = {
            'chamfer_dis': 1.0,
            'dynamic_chamfer_dis': 1.0,
            'static_flow_loss': 1.0,
            'cluster_based_pc0pc1': 1.0,
        }
        wrapper = ModelWrapper(cfg)

        assert wrapper.loss_fn is not None
        assert callable(wrapper.loss_fn)

    def test_accflowloss_import(self):
        """Test accflowLoss import"""
        cfg = create_accflow_config_with_seloss()
        wrapper = ModelWrapper(cfg)

        assert wrapper.loss_fn is not None
        assert callable(wrapper.loss_fn)
        assert wrapper.cfg_loss_name == 'accflowLoss'


class TestModelWrapperModelDetection:
    """Test cases for model type detection"""

    def test_detect_deltaflow(self):
        """Test DeltaFlow model detection"""
        cfg = create_minimal_config(model_name='deltaflow')
        wrapper = ModelWrapper(cfg)

        is_accflow = wrapper.model.__class__.__name__ == 'AccFlow'
        assert not is_accflow

    def test_detect_accflow(self):
        """Test AccFlow model detection"""
        cfg = create_minimal_config(model_name='accflow', num_frames=5)
        wrapper = ModelWrapper(cfg)

        is_accflow = wrapper.model.__class__.__name__ == 'AccFlow'
        assert is_accflow


class TestModelWrapperGridFeatureSize:
    """Test cases for grid_feature_size calculation"""

    def test_grid_feature_size_calculation(self):
        """Test that grid_feature_size is correctly calculated"""
        cfg = create_minimal_config()
        wrapper = ModelWrapper(cfg)

        # Grid feature size should be calculated from point_cloud_range and voxel_size
        expected_x = abs(int((-51.2 - 51.2) / 0.2))  # 512
        expected_y = abs(int((-51.2 - 51.2) / 0.2))  # 512
        expected_z = abs(int((-2.2 - 4.2) / 0.2))    # 32

        assert cfg.model.target.grid_feature_size == [expected_x, expected_y, expected_z]


class TestModelWrapperHyperparameters:
    """Test cases for hyperparameter saving"""

    def test_save_hyperparameters(self):
        """Test that hyperparameters are saved"""
        cfg = create_minimal_config()
        wrapper = ModelWrapper(cfg)

        # LightningModule should have saved hyperparameters
        assert hasattr(wrapper, 'hparams')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
