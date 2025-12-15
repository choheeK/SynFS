# Selector 
# Network (selector for each veiw & shared predictor for each interaction )

# open source library 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import math 
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


import sys 
sys.path.append('/home/chohee.kim/SYNFS')
from models.util import _standard_truncnorm_sample
from models.modules import MLP,  FS_predictor, predictor
from data.dataset_util import SimpleDataset


class SynFS(object):
    def __init__(self, config):
        
        self.config = config
        self.batch_size = self.config['batch_size']
        self.activation = self.config['activation']
        self.device = self.get_device()
        self.ckpt_dir = self.config['save_dir']
        
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.cos_loss = nn.CosineSimilarity(dim=0)

        # Synergistic
        self.s_model = FS_predictor(config)
        self.s_model.apply(self.init_weights)
        self.s_model.to(self.device)

        # Non-synergistic
        self.ns_model = FS_predictor(config)
        self.ns_model.apply(self.init_weights)
        self.ns_model.to(self.device)

        # Informative (Syn U Non-Syn)
        self.all_inf_model = predictor(config)
        self.all_inf_model.apply(self.init_weights)
        self.all_inf_model.to(self.device)
        
        self.s_h_params = list(self.s_model.shared_predictor.parameters()) 
        self.ns_h_params = list(self.ns_model.shared_predictor.parameters()) 

        self.s_params = [p for selector in self.s_model.s_selectors for p in selector.parameters()] 
        self.ns_params = [p for selector in self.ns_model.s_selectors for p in selector.parameters()] 
        
        # predictors only 
        self.h_opt = torch.optim.Adam(self.s_h_params+self.ns_h_params, 
                                      lr=config['learning_rate'], 
                                      weight_decay=config['weight_decay']) 
        self.synergy_opt = torch.optim.Adam(self.s_params, lr=config['s_learning_rate'])# list of optimizers
        self.nsynergy_opt = torch.optim.Adam(self.ns_params, lr=config['s_learning_rate'])
        self.all_inf_opt = torch.optim.Adam(self.s_params + self.ns_params + list(self.all_inf_model.parameters()),
                                            lr=config['learning_rate'], 
                                            weight_decay=config['weight_decay'])

    def get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def init_weights(self, m): # same with STG 
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound = -2*stddev, upper_bound = 2 * stddev, sample_shape = shape))
            torch.nn.init.zeros_(m.bias)
    
    def get_dataloader(self, X_set, y, batch_size):
        dataset = SimpleDataset(X_set, y, device=self.device)
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        return data_loader
    
    def get_reg(self, selector):
        reg = selector.regularizer
        reg = torch.mean(reg(selector.mu / selector.sigma))
        return reg
    
    def mask_generator(self, batch_size=None):
        # generating mask for latent space, return a list of masks for each view 
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        masks = []
        view_sum = sum(self.config['views_dims'])
        cumsum = np.cumsum(self.config['views_dims'])
        blank_mask = torch.zeros(batch_size, view_sum, device=self.device)
        for v in range(len(self.config['views_dims'])):
            v_mask = blank_mask.clone()
            if v == 0 :
                v_mask[:, :cumsum[v]] = 1
            else:
                v_mask[:, cumsum[v-1]:cumsum[v]] = 1
            masks.append(v_mask)
        return masks

    def get_detached_mu(self, model):
        S = [selector.hard_sigmoid(selector.mu.detach()) for selector in model.s_selectors]
        return S
    def get_attached_mu(self, model):
        S = [selector.hard_sigmoid(selector.mu) for selector in model.s_selectors]
        return S
    
    def eval_forward(self, model, S, views, X_mean_set):
        masks = self.mask_generator(batch_size=views[0].shape[0])
        s_z = [S[i]*views[i]+(1-S[i])*X_mean_set[i] for i in range(len(views))]

        logits = []
        for v in range(len(self.config['views_dims'])):
            logit = model.shared_predictor(torch.cat(s_z, dim=1)*masks[v])
            logits.append(logit)
        bar_logit = model.shared_predictor(torch.cat(s_z, dim=1))
        logits.append(bar_logit)
        
        return logits 


    def train_step(self, data, X_mean_set):
        self.s_model.train()
        self.ns_model.train()
        self.all_inf_model.train()
        
        views, y = data[:-1][0], data[-1]
        
        masks = self.mask_generator()
        
        # new v 
        v_s = [selector(views[i], X_mean_set[i]) for i, selector in enumerate(self.s_model.s_selectors)] 
        v_n = [selector(views[i], X_mean_set[i]) for i, selector in enumerate(self.ns_model.s_selectors)]
        
        z_s = [selector.z for selector in self.s_model.s_selectors]
        z_n = [selector.z for selector in self.ns_model.s_selectors]
        
        ###############################
        ###       PREDICTORS       ###
        ###############################

        # with the same noise as with selectors  
        #  non-synergistic
        n_logits = []
        s_logits = []
        for v in range(len(self.config['views_dims'])):
            logit = self.ns_model.shared_predictor(torch.cat(v_n, dim=1)*masks[v])
            n_logits.append(logit)
        n_v_bar_logits = self.ns_model.shared_predictor(torch.cat(v_n, dim=1))
        n_logits.append(n_v_bar_logits)
        
        #  synergistic
        for v in range(len(self.config['views_dims'])):
            logit = self.s_model.shared_predictor(torch.cat(v_s, dim=1)*masks[v])
            s_logits.append(logit)
        s_v_bar_logits = self.s_model.shared_predictor(torch.cat(v_s, dim=1))
        s_logits.append(s_v_bar_logits)
        
        n_losses = [self.loss(logit, y) for logit in n_logits]
        s_losses = [self.loss(logit, y) for logit in s_logits]

        h_loss = torch.mean(torch.sum(torch.stack(n_losses+s_losses, dim=1), dim=1))

        self.h_opt.zero_grad() 
        h_loss.backward(retain_graph=True) # s_selector, n_selector, predictors , retain_grph =True for v_s, n_s, z_s,z_n
        self.h_opt.step()  # update predictors
        
        ##############################################
        ###   Synergistic Selector & Predictor   ###
        ##############################################
        
        n_logits = []
        s_logits = []
        
        # all informative gate (max(synergic gate , non-synergic gate))
        all_gates = [self.s_model.s_selectors[0].hard_sigmoid(torch.max(s, n)) for s, n in zip(z_s, z_n)]  # with noise
        
        all_v = [views[i]*gate + X_mean_set[i]*(1-gate) for i, gate in enumerate(all_gates)]
        #gate_s, gate_n =  self.get_detached_mu(self.s_model), self.get_detached_mu(self.ns_model) # wo noise hard sigmoid

        gate_s, gate_n = [self.s_model.s_selectors[0].hard_sigmoid(s) for s in z_s], [self.s_model.s_selectors[0].hard_sigmoid(n) for n in z_n] 
        s_regs =[self.get_reg(selector) for selector in self.s_model.s_selectors]
        
        #  all-informative
        all_bar_logits = self.all_inf_model(torch.cat(all_v, dim=1))
        all_bar_loss = self.loss(all_bar_logits, y)

        for v in range(len(self.config['views_dims'])):
            logit = self.s_model.shared_predictor(torch.cat(v_s, dim=1)*masks[v])
            s_logits.append(logit)
        s_v_bar_logits = self.s_model.shared_predictor(torch.cat(v_s, dim=1))

        s_losses = [self.loss(logit, y) for logit in s_logits]
        s_v_bar_loss = self.loss(s_v_bar_logits, y)
        all_inf_loss = torch.mean(all_bar_loss) 
        
        synergy_loss = torch.mean(s_v_bar_loss - torch.sum(torch.stack(s_losses, dim=1), dim=1) \
                                + self.config['s_lam']*torch.mean(torch.stack(s_regs)) \
                                )
                    #+ self.config['ns_alpha']*sim 

        self.all_inf_opt.zero_grad()
        self.synergy_opt.zero_grad()
        all_inf_loss.backward(retain_graph=True) #s selecotr, n selector, ai predictor 
        synergy_loss.backward() # s selector, predictors 
        self.all_inf_opt.step()  
        self.synergy_opt.step()
        
        #############################################
        ###   Non-syneristic Selector & Predictor ###
        #############################################

        sim = torch.nn.functional.cosine_similarity(torch.cat(gate_s, dim=1), torch.cat(gate_n, dim=1), dim=1) #(batch, alldim)
        ns_regs =[self.get_reg(selector) for selector in self.ns_model.s_selectors]
        
        for v in range(len(self.config['views_dims'])):
            logit = self.ns_model.shared_predictor(torch.cat(v_n, dim=1)*masks[v])
            n_logits.append(logit)
        n_v_bar_logits = self.ns_model.shared_predictor(torch.cat(v_n, dim=1))
        
        n_losses = [self.loss(logit, y) for logit in n_logits]
        n_v_bar_loss = self.loss(n_v_bar_logits, y)

        nsynergy_loss = torch.mean(-n_v_bar_loss + torch.sum(torch.stack(n_losses, dim=1), dim=1) 
                                   + self.config['ns_lam']*torch.mean(torch.stack(ns_regs)) # same mu for all batch
                                   + self.config['ns_alpha']*sim 
                                   )
        
        self.nsynergy_opt.zero_grad()
        nsynergy_loss.backward() # n selector, predcitor 
        self.nsynergy_opt.step()
        
        #print(f"sim | {torch.mean(sim).item():.3f}, nsynergy_loss | {nsynergy_loss.item():.3f}, percent | {self.config['ns_alpha']*torch.mean(sim).item()/nsynergy_loss.item()*100:.1f}")
         
        try:
            ai_auroc = roc_auc_score(y.detach().cpu().numpy(), all_bar_logits.detach().cpu().numpy()[:,1])
            n_auroc = roc_auc_score(y.detach().cpu().numpy(), n_v_bar_logits.detach().cpu().numpy()[:,1])
            s_auroc = roc_auc_score(y.detach().cpu().numpy(), s_v_bar_logits.detach().cpu().numpy()[:,1])
        except:
            ai_auroc, n_auroc, s_auroc = np.array([0]), np.array([0]), np.array([0])
        
        return ai_auroc, n_auroc, s_auroc, all_inf_loss, nsynergy_loss, synergy_loss


    def save_log(self, log_list, item_list):
        a_list, n_list, s_list = log_list
        a, n, s = item_list
        a_list.append(a.item())
        n_list.append(n.item())
        s_list.append(s.item())
        return a_list, n_list, s_list

    def fit(self, tr_X_set, tr_y, va_X_set, va_y, nr_epochs):
        
        np.set_printoptions(precision=1, suppress=True, linewidth=200)
        best_score = 0
        train_data_loader = self.get_dataloader(tr_X_set, tr_y, batch_size=self.config['batch_size'])
        val_data_loader = self.get_dataloader(va_X_set, va_y, batch_size=self.config['batch_size'])
        
        X_mean_set = [torch.mean(torch.Tensor(X).to(self.device), dim=0) for X in tr_X_set]

        for epoch in tqdm(range(nr_epochs)):
            # Train
            train_auroc, train_n_auroc, train_s_auroc = [], [], []
            train_loss, train_n_loss, train_s_loss = [], [], []

            for data in train_data_loader:
      
                a_auroc, n_auroc, s_auroc, a_loss, n_loss, s_loss = self.train_step(data, X_mean_set)
                train_auroc, train_n_auroc, train_s_auroc = self.save_log([train_auroc, train_n_auroc, train_s_auroc], [a_auroc, n_auroc, s_auroc])
                train_loss, train_n_loss, train_s_loss = self.save_log([train_loss, train_n_loss, train_s_loss], [a_loss, n_loss, s_loss])

            avg_train_auroc, avg_train_n_auroc, avg_train_s_auroc =  np.mean(train_auroc), np.mean(train_n_auroc), np.mean(train_s_auroc)
            avg_train_loss, avg_train_n_loss, avg_train_s_loss =  np.mean(train_loss), np.mean(train_n_loss), np.mean(train_s_loss)


            if epoch == 0:  #initial ckpt
                self.save_ckpt()  
            
            # Validation
            val_auroc, val_n_auroc, val_s_auroc = [], [], []
            val_loss, val_n_loss, val_s_loss = [], [], []
            s_gates = []
            ns_gates = []
            for data in val_data_loader:
                a_auroc, n_auroc, s_auroc, s_gate, ns_gate, a_loss, n_loss, s_loss = self.evaluate(data, X_mean_set)
                self.save_log([val_auroc, val_n_auroc, val_s_auroc], [a_auroc, n_auroc, s_auroc])
                self.save_log([val_loss, val_n_loss, val_s_loss], [a_loss, n_loss, s_loss])
                s_gates.append(s_gate)
                ns_gates.append(ns_gate)

            avg_val_auroc, avg_val_n_auroc, avg_val_s_auroc = np.mean(val_auroc), np.mean(val_n_auroc), np.mean(val_s_auroc)
            avg_s_gates, avg_ns_gates = torch.mean(torch.stack(s_gates), dim=0), torch.mean(torch.stack(ns_gates), dim=0)
            avg_val_loss, avg_val_n_loss, avg_val_s_loss = np.mean(val_loss), np.mean(val_n_loss), np.mean(val_s_loss)
            #print(f"\n epoch{epoch}, \nsgate:{avg_s_gates}, \nnsgate:{avg_ns_gates}")
            delta = 0.001
            if avg_val_auroc > best_score + delta :
                print(f"update bestscore: {best_score:.3f} --> {avg_val_auroc:.3f}")
                best_score = avg_val_auroc
                self.save_ckpt()  
    
    def evaluate(self, data, X_mean_set):
        self.s_model.eval()
        self.ns_model.eval()
        self.all_inf_model.eval()
        
        views, y = data[:-1][0], data[-1]
        
        with torch.no_grad():
            S = self.get_detached_mu(self.s_model)
            NS = self.get_detached_mu(self.ns_model)
            
            all_mu = [torch.max(s, ns) for s, ns in zip(S, NS)]
            all_z = [all_mu[i]*views[i]+(1-all_mu[i])*X_mean_set[i] for i in range(len(views))]

            #  all-informative
            all_bar_logits = self.all_inf_model(torch.cat(all_z, dim=1))
            
            s_logits = self.eval_forward(self.s_model, S, views, X_mean_set)
            n_logits = self.eval_forward(self.ns_model, NS, views, X_mean_set)
            
            n_losses = [self.loss(logit, y) for logit in n_logits]
            s_losses = [self.loss(logit, y) for logit in s_logits]

            all_bar_loss = torch.mean(self.loss(all_bar_logits, y))
            h_loss = torch.mean(torch.sum(torch.stack(n_losses+s_losses, dim=1), dim=1))

            s_v_bar_loss = s_losses.pop(-1)
            n_v_bar_loss = n_losses.pop(-1)
            synergy_loss = torch.mean(s_v_bar_loss - torch.sum(torch.stack(s_losses, dim=1), dim=1))
            nsynergy_loss = -torch.mean(n_v_bar_loss - torch.sum(torch.stack(n_losses, dim=1), dim=1)) 
            
            try:     
                ai_auroc = roc_auc_score(y.detach().cpu().numpy(), all_bar_logits.detach().cpu().numpy()[:,1])
                n_auroc = roc_auc_score(y.detach().cpu().numpy(), n_logits[-1].detach().cpu().numpy()[:,1])
                s_auroc = roc_auc_score(y.detach().cpu().numpy(), s_logits[-1].detach().cpu().numpy()[:,1])
            except: # batch error
                ai_auroc, n_auroc, s_auroc = np.array([0]), np.array([0]), np.array([0])
    
        return ai_auroc, n_auroc, s_auroc, torch.cat(S), torch.cat(NS), all_bar_loss, nsynergy_loss, synergy_loss
    
    def predict(self, views, te_X_set):
        #t_index to check the target view , if none logits will be the same 
        self.s_model.eval()
        self.ns_model.eval()
        self.all_inf_model.eval()
        
        X_mean_set = [torch.mean(torch.Tensor(X).to(self.device), dim=0) for X in te_X_set] 

        with torch.no_grad():
            S = self.get_detached_mu(self.s_model)
            NS = self.get_detached_mu(self.ns_model)
            all_mu = [torch.max(s, ns) for s, ns in zip(S, NS)]
            all_z = [all_mu[i]*views[i]+(1-all_mu[i])*X_mean_set[i] for i in range(len(views))]
            all_bar_logits = self.all_inf_model(torch.cat(all_z, dim=1))

        return all_bar_logits
                
    def save_ckpt(self):
        torch.save(self.s_model.state_dict(), self.ckpt_dir + 's_model_ckpt.pt')
        torch.save(self.ns_model.state_dict(), self.ckpt_dir + 'ns_model_ckpt.pt')
        torch.save(self.all_inf_model.state_dict(), self.ckpt_dir + 'all_inf_model_ckpt.pt')
        print('ckpt_saved')
    
    def load_ckpt(self):
        self.s_model.state_dict(torch.load(self.ckpt_dir + 's_model_ckpt.pt'))
        self.ns_model.state_dict(torch.load(self.ckpt_dir + 'ns_model_ckpt.pt'))
        self.all_inf_model.state_dict(torch.load(self.ckpt_dir + 'all_inf_model_ckpt.pt'))
        print('ckpt_loaded')