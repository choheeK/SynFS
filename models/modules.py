# Module blocks to compose SynFS network 
# Selector, MLP, FS Embeded predictor (Selector + MLP), predictor (ML)

import torch 
import torch.nn as nn 
import numpy as np 
import math 


class MLP(nn.Module):
    def __init__(self, input_dim , hidden_dims, output_dim, batch_norm = None, dropout = None, activation='relu'):
        super(MLP, self).__init__()
        modules = self.build_layers(input_dim, hidden_dims, output_dim, batch_norm, dropout, activation)
        self.layers = nn.Sequential(*modules)
        
    def base_layer(self, in_features, out_features, batch_norm, dropout, activation):
        modules = [nn.Linear(in_features, out_features, bias=True)]
        if batch_norm : 
            modules.append(nn.BatchNorm1d(out_features))
        if dropout : 
            modules.append(nn.Dropout(0.5, True))
        if activation :
            modules.append(nn.ReLU(True))
        layer = nn.Sequential(*modules)
        return layer
    
    def build_layers(self, input_dim, hidden_dims, output_dim, batch_norm, dropout, activation) :
        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        nr_hiddens = len(hidden_dims)
        modules = []
        for i in range(nr_hiddens) :
            layer = self.base_layer(dims[i], dims[i+1], batch_norm, dropout, activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        return modules
        
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
                
    def forward(self, input):
        out = self.layers(input)
        return out 



class Selector(nn.Module):
    """Stochastic gate to discover selected features 
    
    Args:
        input_dim : dimension of input features
        sigma : constant for gaussian distribution
        device : device for the tensors to be created 
    """
    def __init__(self, input_dim, sigma, mean=math.sqrt(0.5)) -> None:
        super(Selector, self).__init__()
        self.mean = mean # 0.5
        self.mu = 0.01*torch.randn(input_dim,)
        self.mu = torch.nn.Parameter(self.mu, requires_grad=True)
        self.sigma = sigma
    
    def forward(self, prev_v, X_mean) -> None: 

        self.noise = torch.randn(prev_v.size()).to(self.mu.device)
        z = self.mu + self.sigma*self.noise.normal_()*self.training # noise normal_ ~N(0,1)
        self.z = z  # save z for the same noise
        stochastic_gate = self.hard_sigmoid(self.z)
        new_v = prev_v*stochastic_gate + X_mean*(1-stochastic_gate)
        return new_v
    
    def hard_sigmoid(self, v):
        return torch.clamp(v + self.mean, 0.0, 1.0)
    
    def regularizer(self, v):
        #guassian CDF
        return 0.5*(1 + torch.erf(v/math.sqrt(2)))
    
    def get_gates(self, mode='prob') :
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy()+self.mean))
        

class FS_predictor(nn.Module):
    """
    FS Embeded predictor (Selector + MLP)
    """
    def __init__(self, config):
        super(FS_predictor, self).__init__()
        self.s_selectors = nn.ModuleList()

        self.views_dims = config['views_dims']
        num_views = len(self.views_dims)
        for i in range(num_views):
            v_dim = self.views_dims[i]
            s_selector = Selector(v_dim, config['sigma'], config['selector_mean'])
            self.s_selectors.append(s_selector)
            setattr(self, f's_selector_{i}', s_selector)

        self.shared_predictor = MLP(sum(config['views_dims']), config['hidden_dims'], config['output_dim'],
                                config['batch_norm'],config['dropout'], config['activation'],
                                )

class predictor(nn.Module):
    """
    predictor (MLP)
    """
    def __init__(self, config):
        super(predictor, self).__init__()
        self.fc = MLP(sum(config['views_dims']), config['hidden_dims'], config['output_dim'],
                          config['batch_norm'], config['dropout'], config['activation'],
                        )

    def forward(self, v):
        return self.fc(v)