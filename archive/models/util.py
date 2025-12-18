
import numpy as np 
import torch

def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte()
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x**2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound**2 - proposed_x**2)
        else:  # both positive
            assert(lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound**2 - proposed_x**2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0) #inplace
        accept = torch.bernoulli(prob_accept).byte() & ~done # return the prob_accept shape matrix where done is 0 
        if accept.any():
            accept = accept.bool()
            x[accept] = proposed_x[accept]
            accept = accept.byte()
            done |= accept # |=, in-place bitwise OR operator to done 
    return x


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, verbose=False, ckpt_dir=None):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf 
        self.delta = delta
        self.ckpt_dir = './' if None else ckpt_dir
        

    def __call__(self, val_loss, model):
        
        """
        othermodels : list [model1, modelname1, model2, modelname2, ...] 
        """
        
        score = val_loss # lower the better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        if score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience :
                if self.verbose:
                    print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
                self.early_stop = True
        else: 
            self.best_score = score
            if self.verbose: 
                print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            self.save_checkpoint(val_loss, model)
            self.counter = 0

                
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            self.early_stop = True
        else:
            self.early_stop = False
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose : 
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.ckpt_dir + f'model_ckpt.pt')
