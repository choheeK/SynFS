import torch
import torch.nn as nn
from models.modules import FS_predictor, predictor
from models.util import _standard_truncnorm_sample

class SynFSModel(nn.Module):
    """
    Represents the three core networks:
      - Synergistic selector + predictor
      - Non-synergistic selector + predictor
      - All-informative predictor
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Synergistic
        self.s_model = FS_predictor(cfg)
        # Non-synergistic
        self.ns_model = FS_predictor(cfg)
        # All informative
        self.all_inf = predictor(cfg)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.1
            m.weight = nn.Parameter(_standard_truncnorm_sample(
                lower_bound=-2*std, upper_bound=2*std, sample_shape=m.weight.shape
            ))
            nn.init.zeros_(m.bias)

    def get_detached_mu(self, model):
        return [seq.hard_sigmoid(seq.mu.detach()) for seq in model.s_selectors]

    def forward_select(self, gates, views, X_means):
        """Apply gates to each view."""
        return [g * v + (1 - g) * xm for g, v, xm in zip(gates, views, X_means)]
