import torch
import torch.nn as nn
from models.util import _standard_truncnorm_sample

class FSSelector(nn.Module):
    """
    Wrapper if you later want to replace gating logic.
    For now, this simply re-uses your original code.
    """

    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, views, X_means):
        return [sel(v, xm) for sel, v, xm in zip(self.selector.s_selectors, views, X_means)]
