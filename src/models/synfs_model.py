import torch
import torch.nn as nn
from .modules import FS_predictor, predictor
from .util import _standard_truncnorm_sample

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
        self.s_model.apply(self.init_weights)
        
        # Non-synergistic
        self.ns_model = FS_predictor(cfg)
        self.ns_model.apply(self.init_weights)
        # All informative
        self.all_inf = predictor(cfg)
        self.all_inf.apply(self.init_weights)

    def init_weights(self, m): # same with STG 
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound = -2*stddev, upper_bound = 2 * stddev, sample_shape = shape))
            torch.nn.init.zeros_(m.bias)    

    def get_detached_mu(self, model):
        return [seq.hard_sigmoid(seq.mu.detach()) for seq in model.s_selectors]

    def get_gates(self, model):
        S = [selector.hard_sigmoid(selector.mu.detach()) for selector in model.s_selectors]
        return S
