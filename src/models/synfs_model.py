import torch
import torch.nn as nn
from models.modules import FS_predictor, predictor, Selector
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

    def get_detached_mu(self, model):
        return [seq.hard_sigmoid(seq.mu.detach()) for seq in model.s_selectors]
        
    def get_gates(self, model):
        S = [selector.hard_sigmoid(selector.mu.detach()) for selector in model.s_selectors]
        return S
