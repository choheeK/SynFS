# src/data/datamodule.py

from torch.utils.data import DataLoader
from .dummy_user_data import UserMultiViewDataset


def build_dataloaders(cfg):

    train_dataset = UserMultiViewDataset(
        data_path=cfg.data.train_path,          # can be None -> then generate synthetic dummy 
        views_dims=cfg.model.views_dims,
        device=cfg.device,
    )

    val_dataset = UserMultiViewDataset(
        data_path=cfg.data.val_path,          # can be None -> then generate synthetic dummy 
        views_dims=cfg.model.views_dims,
        device=cfg.device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Optional validation (same dataset for now)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
