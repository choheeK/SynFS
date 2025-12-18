# src/data/base_dataset.py

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class MultiViewDataset(Dataset, ABC):
    """
    Base interface for SynFS-compatible datasets.

    Requirements:
    - __getitem__ returns: (list_of_views, label)
    - list_of_views: List[Tensor] with length = n_views
    """

    @abstractmethod
    def num_views(self) -> int:
        pass

    @abstractmethod
    def view_dims(self):
        """Return list of view dimensions, e.g. [250, 250]"""
        pass
