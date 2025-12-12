import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """
    X_set = [view1, view2, ..., viewN]   (numpy arrays)
    y     = labels (numpy)
    """

    def __init__(self, X_set, y):
        self.views = X_set
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        views = [torch.tensor(v[idx]).float() for v in self.views]
        y = torch.tensor(self.y[idx]).long()
        return views, y
