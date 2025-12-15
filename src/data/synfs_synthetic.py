# src/data/synfs_synthetic.py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# -----------------------------
# Ground-truth feature selectors
# -----------------------------
def generate_views_gt(views):
    """
    Ground-truth informative features used in generating the synthetic data.
    Returns:
        a_gt  - all-informative features (non-syn + syn)
        syn_gt - synergistic-only features
    """
    gt_views = [np.zeros(view.shape[1]) for view in views]

    # All-informative GT features (from original notebook)
    gt_views[0][0] = 1
    gt_views[0][2] = 1
    gt_views[1][1] = 1
    gt_views[1][3] = 1

    syn_views = [np.zeros(view.shape[1]) for view in views]
    syn_views[0][0] = 1
    syn_views[1][1] = 1

    return gt_views, syn_views


# -----------------------------
# Synthetic dataset generator
# -----------------------------
def generate_multi_dataset(n, dims, seed=0):
    """
    Bit-identical synthetic dataset generator from original notebook.
    Args:
        n: number of samples
        dims: list of dimensions for each view (e.g. [250, 250])
        seed: random seed
    Returns:
        views: list of np arrays (one per view)
        y: binary labels (0/1)
        (a_gt, syn_gt): ground truth informative feature masks
    """

    np.random.seed(seed)

    # Generate each view
    views = [np.random.randn(n, dim) for dim in dims]

    # y generation rule (IDENTICAL to notebook)
    y = np.zeros((n, 2))
    logit = np.exp(
        views[0][:, 0] * views[1][:, 1] +
        views[0][:, 2] +
        views[1][:, 3]
    )

    # Compute P(Y = 0 | X)
    prob_0 = (logit / (1 + logit)).reshape(n, 1)

    # Sample labels
    y[:, 0] = np.random.binomial(1, prob_0).reshape(-1)
    y[:, 1] = 1 - y[:, 0]
    print("validate:", np.unique(y[:, 1], return_counts=True))

    y = y[:, 1]   # binary label: 1 = positive class

    a_gt, syn_gt = generate_views_gt(views)
    return views, y, (a_gt, syn_gt)


# -----------------------------
# Train/Val/Test Split Logic
# -----------------------------
def split_dataset(views, y, val_size=0.2, test_size=0.2, seed=0):
    n = len(y)

    # Match notebook logic
    train_idx, test_idx = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=seed)

    tr_X = [v[train_idx] for v in views]
    va_X = [v[val_idx] for v in views]
    te_X = [v[test_idx] for v in views]

    tr_y = y[train_idx]
    va_y = y[val_idx]
    te_y = y[test_idx]

    return (tr_X, tr_y), (va_X, va_y), (te_X, te_y)


# -----------------------------
# PyTorch Dataset Wrapper
# -----------------------------
class SimpleDataset(Dataset):
    def __init__(self, data_set, y, device):
        self.data_set = [torch.tensor(v, dtype=torch.float32).to(device) for v in data_set]
        self.y = torch.tensor(y).long().to(device)
        self.device = device

    def __len__(self):
        return len(self.data_set[0])

    def __getitem__(self, i):
        xs = [v[i] for v in self.data_set]
        return xs, self.y[i]
