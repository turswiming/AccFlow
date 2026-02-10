"""
Learning Rate Finder for AccFlow.

Usage:
  python find_lr.py model=accflow loss_fn=accflowLoss \
      '+add_seloss={chamfer_dis:1.0,static_flow_loss:1.0,dynamic_chamfer_dis:1.0,cluster_based_pc0pc1:1.0}' \
      +find_lr.save_plot=true

  # With custom LR range:
  python find_lr.py model=accflow +find_lr.save_plot=true \
      +find_lr.min_lr=1e-8 +find_lr.max_lr=1e-1 +find_lr.num_training=200

  # Save JSON: +find_lr.save_json=true (default), +find_lr.json_path=/path/to/out.json
"""

import json
import math
import os
from pathlib import Path

import torch
import lightning.pytorch as pl
try:
    from lightning.pytorch.tuner import Tuner
except ImportError:
    from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf, DictConfig

import hydra
from hydra.core.hydra_config import HydraConfig

from src.dataset import HDF5Dataset, HDF5DatasetAccFlow, collate_fn_pad, ToTensor
from src.trainer import ModelWrapper
from train import precheck_cfg_valid

# Default find_lr params (override via +find_lr.xxx=yyy)
FIND_LR_DEFAULTS = {
    "min_lr": 1e-8,
    "max_lr": 1.0,
    "num_training": 100,
    "save_plot": True,  # save LR vs loss plot by default
    "plot_path": None,
    "save_json": True,  # save lr/loss to JSON by default
    "json_path": None,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    flr = OmegaConf.merge(OmegaConf.create(FIND_LR_DEFAULTS), cfg.get("find_lr", OmegaConf.create({})))

    precheck_cfg_valid(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    train_aug = transforms.Compose([ToTensor()])
    is_accflow = cfg.model.name in ['accflow', 'accflow2frame']
    TrainDatasetClass = HDF5DatasetAccFlow if is_accflow else HDF5Dataset

    train_dataset = TrainDatasetClass(
        cfg.train_data,
        n_frames=cfg.num_frames,
        ssl_label=cfg.get('ssl_label', None),
        transform=train_aug
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_pad,
        pin_memory=True
    )

    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    model = ModelWrapper(cfg)

    output_dir = HydraConfig.get().runtime.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=min(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        logger=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        max_epochs=-1,
    )

    print("\n" + "=" * 60)
    print("Running Learning Rate Finder...")
    print(f"Model: {cfg.model.name}, LR range: [{flr.min_lr:.2e}, {flr.max_lr:.2e}]")
    print(f"Steps: {flr.num_training}")
    print("=" * 60 + "\n")

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        train_dataloaders=train_loader,
        min_lr=flr.min_lr,
        max_lr=flr.max_lr,
        num_training=flr.num_training,
    )

    suggested_lr = lr_finder.suggestion()
    print("\n" + "=" * 60)
    print("LR Finder Results")
    print("=" * 60)
    print(f"Suggested learning rate: {suggested_lr:.6e}" if suggested_lr else "Suggestion failed")
    print(f"Config lr: {cfg.optimizer.lr:.6e}")
    if suggested_lr:
        print(f"Ratio: {suggested_lr / cfg.optimizer.lr:.2f}x")
    print("=" * 60 + "\n")

    if flr.save_plot:
        plot_path = flr.plot_path or os.path.join(output_dir, "lr_finder_plot.png")
        try:
            fig = lr_finder.plot(suggest=True, show=False)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

    if flr.save_json:
        json_path = flr.json_path or os.path.join(output_dir, "lr_finder_results.json")
        try:
            def safe_float(x):
                return None if (isinstance(x, float) and math.isnan(x)) else float(x)

            lr_list = [safe_float(x) for x in lr_finder.results["lr"]]
            loss_list = [safe_float(x) for x in lr_finder.results["loss"]]

            data = {
                "lr": lr_list,
                "loss": loss_list,
                "num_steps": len(lr_list),
                "min_lr": flr.min_lr,
                "max_lr": flr.max_lr,
                "num_training": flr.num_training,
                "suggested_lr": float(suggested_lr) if suggested_lr is not None else None,
                "config_lr": cfg.optimizer.lr,
            }
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"LR finder results saved to {json_path}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")

    return suggested_lr


if __name__ == "__main__":
    main()
