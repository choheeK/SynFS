# src/data/dummy_user_data.py

import torch
from torch.utils.data import Dataset

from .base_dataset import MultiViewDataset
from .synfs_synthetic import generate_multi_dataset


class UserMultiViewDataset(MultiViewDataset):
    """
    Canonical dataset for SynFS.

    If data_path is None:
        -> generate dummy synthetic data (for validation)
    If data_path is provided:
        -> user should load real data there
    """

    def __init__(
        self,
        data_path: str | None,
        views_dims: list[int],
        device: str = "cpu",
    ):
        super().__init__()

        if views_dims is None:
            raise ValueError("views_dims must be provided")

        # =====================================================
        # CASE 1: Dummy / validation mode
        # =====================================================
        if data_path is None or str(data_path).lower() == "none":
            views, y, _ = generate_multi_dataset(
                n=1000,
                dims=views_dims,
                seed=42,
            )

        # =====================================================
        # CASE 2: User real data
        # =====================================================
        else:
            # -----------------------------
            # TODO: USER IMPLEMENTATION
            # -----------------------------
            # Example (user replaces this):
            #
            # views = [
            #     load_view1(data_path),
            #     load_view2(data_path),
            # ]
            # y = load_labels(data_path)
            #
            raise NotImplementedError(
                "Real data loading not implemented. "
                "Replace this block in UserMultiViewDataset."
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
        y = self.labels[idx]
        return xs, y

    # -----------------------------
    # MultiViewDataset contract
    # -----------------------------
    def num_views(self) -> int:
        return len(self.views)

    def view_dims(self):
        return [v.shape[1] for v in self.views]
