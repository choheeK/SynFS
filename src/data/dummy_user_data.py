# src/data/dummy_user_data.py

from pathlib import Path
import numpy as np
import torch

from .base_dataset import MultiViewDataset
from .synfs_synthetic import generate_multi_dataset


class UserMultiViewDataset(MultiViewDataset):
    """
    Canonical dataset for SynFS.

    Supports:
    - Arbitrary number of views
    - train / val / test splits
    - CSV-based real data OR synthetic fallback
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        device: str = "cpu",
    ):
        super().__init__()

        if split not in cfg.data:
            raise ValueError(f"Split '{split}' not found in cfg.data")

        split_cfg = cfg.data[split]
        views_dims = cfg.views_dims

        if views_dims is None:
            raise ValueError("views_dims must be specified")

        project_root = Path.cwd().resolve()

        # =====================================================
        # CASE 1: Real CSV data
        # =====================================================
        if split_cfg.views is not None and split_cfg.labels is not None:
            views = []

            if len(split_cfg.views) != len(views_dims):
                raise ValueError(
                    f"Number of views ({len(split_cfg.views)}) "
                    f"does not match views_dims ({len(views_dims)})"
                )

            for v_cfg in split_cfg.views:
                path = project_root / v_cfg.path
                if not path.exists():
                    raise FileNotFoundError(f"View file not found: {path}")
                views.append(np.loadtxt(path, delimiter=","))

            y_path = project_root / split_cfg.labels
            if not y_path.exists():
                raise FileNotFoundError(f"Label file not found: {y_path}")
            y = np.loadtxt(y_path, delimiter=",")

        # =====================================================
        # CASE 2: Synthetic fallback
        # =====================================================
        else:
            views, y, _ = generate_multi_dataset(
                n=1000,
                dims=views_dims,
                seed=42,
            )

        # =====================================================
        # Convert to torch tensors
        # =====================================================
        self.views = [
            torch.tensor(v, dtype=torch.float32, device=device)
            for v in views
        ]
        self.labels = torch.tensor(y, dtype=torch.long, device=device)

    # -----------------------------
    # Dataset interface
    # -----------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [v[idx] for v in self.views]
        return xs, self.labels[idx]

    # -----------------------------
    # MultiViewDataset contract
    # -----------------------------
    def num_views(self) -> int:
        return len(self.views)

    def view_dims(self):
        return [v.shape[1] for v in self.views]
