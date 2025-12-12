import pickle
from torch.utils.data import DataLoader
from .dataset import SimpleDataset

def build_dataloaders(cfg):
    train = pickle.load(open(cfg.data.train_path, "rb"))
    val = pickle.load(open(cfg.data.val_path, "rb"))

    train_ds = SimpleDataset(train["X"], train["y"])
    val_ds   = SimpleDataset(val["X"], val["y"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    return train_loader, val_loader
