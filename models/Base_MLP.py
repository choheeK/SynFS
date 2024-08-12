
import sys 
sys.path.append('/home/chohee.kim/SYNFS')
from models.modules import MLP
from models.util import _standard_truncnorm_sample, EarlyStopping
from data.dataset_util import SimpleDataset

import numpy as np 

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader



class BaseNetwork(nn.Module):  # without selector
    def __init__(self, config):
        super(BaseNetwork, self).__init__()
        self.fc = MLP(sum(config['views_dims']), config['hidden_dims'], config['output_dim'],
                          config['batch_norm'], config['dropout'], config['activation'],
                        )

    def forward(self, v):
        return self.fc(v)
    

class Base(object):
    def __init__(self, config):
        
        """
        Base Model without Feature selection 
        input :  set of views -> will concatenate for training inside this class 
        """
        self.config = config
        self.batch_size = config['batch_size']
        self.activation = config['activation']
        self.device = self.get_device()
        self.early_stop = config['early_stop'] 
        self.loss = nn.CrossEntropyLoss()
        self.ckpt_dir = config['save_dir']
        self.model = BaseNetwork(self.config)
        self.model.apply(self.init_weights)
        self.model.to(self.device)
        
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr = config['learning_rate'], weight_decay = config['weight_decay']
                                    ) 
        self.lr_decay = self.config['lr_decay']
        #print("upated on 2023.09.05, base mlp model")

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound = -2*stddev, upper_bound = 2 * stddev, sample_shape = shape))
            torch.nn.init.zeros_(m.bias)

    def get_dataloader(self, X_set, y, batch_size=None):
        if batch_size is None:
            batch_size = self.config['batch_size']
        dataset = SimpleDataset(X_set, y, device=self.device)
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        return data_loader

    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def train_step(self, data):
        self.model.train()
        view, y = torch.cat(data[0], dim=1), data[1]
        logits = self.model(view)
        loss = self.loss(logits, y) 
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss
    
    def fit(self, tr_X_set, tr_y, va_X_set, va_y, nr_epochs, single_view=False):

        loss_list = []
        train_data_loader = self.get_dataloader(tr_X_set, tr_y)  #tr_X is a set of one view element 
        val_loss_list = []
        val_data_loader = self.get_dataloader(va_X_set, va_y)
        
        
        if self.config['early_stop']:
            early_stopping = EarlyStopping(patience=self.config['patience'], delta=self.config['delta'], ckpt_dir=self.ckpt_dir)

        best_loss = np.inf
        for epoch in range(nr_epochs):
            ttl_loss = 0
            for batch in train_data_loader:
                loss = self.train_step(batch) 
                ttl_loss += loss.item()
            avg_loss = ttl_loss/len(train_data_loader)
            loss_list.append(avg_loss)
            
            if self.lr_decay:
                for g in self.opt.param_groups:  # clssfier and as selector 
                    g["lr"] *= self.lr_decay
            ttl_val_loss = 0
            for batch in val_data_loader:
                val_loss = self.evaluate(batch)
                ttl_val_loss += val_loss.item()
            avg_val_loss = ttl_val_loss/len(val_data_loader)
            val_loss_list.append(avg_val_loss)

            if avg_val_loss < best_loss:
                self.save_ckpt()
                best_loss = avg_val_loss

            if self.config['early_stop']:
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
    def evaluate(self, data):
        self.model.eval()
        view, y = torch.cat(data[0], dim=1), data[1]
        with torch.no_grad():
            logits = self.model(view)   
            val_loss = self.loss(logits, y)    
        return val_loss
    
    def predict(self, view):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(view)
        return logits

    def save_ckpt(self):
        torch.save(self.model.state_dict(), self.ckpt_dir + 'model_ckpt.pt')
        #print('ckpt_saved')
    
    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_dir + 'model_ckpt.pt'))
        #print('ckpt_loaded')
        