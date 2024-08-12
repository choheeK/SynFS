#Data Generating (ref Invase, Composite-Feature-Selection.compfs.datasets.datasets)

# Necessary packages
import numpy as np
import pdb 
import numpy as np 
import pandas as pd 
import torch 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset
import pickle
import os
import argparse

###############################
######  Multiple Views   ######
###############################

def generate_views(n, dims):
    views = [np.random.randn(n, dim) for dim in dims]
    return views

def generate_corr_features(n, dims, rho):
    """
    correlation between features from different views 
    """
    total_dims = np.sum(dims)
    dim1 = int(total_dims/2)

    cov = torch.eye(total_dims)
    cov[dim1:dim1*2, 0:dim1] = rho * torch.eye(dim1)
    cov[0:dim1, dim1:dim1*2] = rho * torch.eye(dim1)
    rho_data = np.random.multivariate_normal(mean=[0]*total_dims,
                                            cov=cov,
                                            size=n)
    
    views = np.split(rho_data, np.cumsum(dims)[:-1], axis=1)

    return views


def generate_y_for_views(views:np.array, data_type:str, syn_rho=0):
    print(f"generating y for {data_type}")

    n = views[0].shape[0] 
    print("n:", n)
    y = np.zeros((n, 2))
    
    if "gauss" not in data_type:
        # Logit computation
        if (data_type == 'syn1') or (data_type =='large_syn1'):
            logit = np.exp(views[0][:,0]*views[1][:,1] \
                    + views[0][:,2] + views[1][:,3])

        elif (data_type == 'red_syn1'):
            views[1][:,4] = views[0][:, 2].copy()  # view2 w5 is redundant feature for view1 w3 
            logit = np.exp(views[0][:,0]*views[1][:,1] \
                    + views[0][:,2] + views[1][:,3])
            
        elif (data_type == 'corr_syn1'):
            views[1][:,1] = views[0][:, 0].copy()*0.7 + np.random.normal(scale=0.09, size=n) # correlated synergic feature 
            logit = np.exp(views[0][:,0]*views[1][:,1] \
                    + views[0][:,2] + views[1][:,3])
        
        elif (data_type == 'small_corr_syn1'):
            views[1][:,1] = views[0][:, 0].copy()*0.1 + np.random.normal(scale=0.09, size=n) # correlated synergic feature 
            logit = np.exp(views[0][:,0]*views[1][:,1] \
                    + views[0][:,2] + views[1][:,3])

        elif (data_type == 'rho_corr_syn1'):
            rho_data = np.random.multivariate_normal(mean=[0,0],
                                        cov=[[1,syn_rho],[syn_rho,1]],
                                        size = n,
                                        )
            views[0][:, 0] = rho_data[:,0]
            views[1][:,1] = rho_data[:,1]
            
            logit = np.exp(views[0][:,0]*views[1][:,1] \
                    + views[0][:,2] + views[1][:,3])

        elif data_type == 'syn2':
            logit = np.exp(\
                    views[0][:,0]+ views[1][:,1] + views[0][:,2] + views[1][:,3])

        elif data_type == 'syn3':
            logit = np.exp(views[0][:,0]*views[1][:,1]\
                    + views[0][:,2] + views[1][:,3] + views[0][:,6] + views[1][:,7])

        elif (data_type == 'syn4') or (data_type == 'large_syn4'):
            logit = np.exp(views[0][:,0]*views[1][:,1] + views[0][:,6]*views[1][:,7] \
                    + views[0][:,2] + views[1][:,3])
            
        elif (data_type == 'syn4_overlap'):
            logit = np.exp(views[0][:,0]*views[1][:,1] + views[0][:,0]*views[1][:,7] \
                    + views[0][:,2] + views[1][:,3])
            
        elif data_type == 'large_syn5':
            logit = np.exp(np.sum(views[0][:,0:6]*views[1][:,1:7], axis=1) + \
                    + views[0][:,9] + views[1][:,10])
            
        elif data_type == '3views_syn1':
            logit = np.exp(views[0][:, 0]*views[1][:, 1]*views[2][:, 2] \
                          + views[0][:, 5] + views[1][:, 5] + views[2][:, 5])
            
        elif data_type == '3views_syn2':
            logit = np.exp(views[0][:, 0]*views[1][:, 1] + views[1][:, 2]*views[2][:, 2] \
                          + views[0][:, 5] + views[1][:, 5] + views[2][:, 5])
            
        elif data_type == '3views_syn3':
            logit = np.exp(views[1][:, 0]*views[2][:, 1] + views[0][:, 2]*views[2][:, 2] \
                        + views[0][:, 5] + views[1][:, 5] + views[2][:, 5])
            
        elif data_type == '4views_syn1':
            logit = np.exp(views[0][:, 0]*views[1][:, 1]*views[2][:, 2]*views[3][:, 3] \
                          + views[0][:, 5] + views[1][:, 5] + views[2][:, 5] + views[3][:, 5])
            
        elif data_type == '3views_dummy':
            logit = np.exp(views[0][:, 0]*views[1][:, 1] \
                            + views[0][:, 2] + views[1][:, 3])
            
        # Compute P(Y=0|X)
        prob_0 = np.reshape((logit / (1+logit)), [n, 1])
        
        # Sampling process
        y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n,])
        
    # Compfs 
    else:
        if data_type == 'gauss_rule1':
            y[:, 0] = (views[0][:,1]> 0.55) | (views[1][:,2] > 0.55)

        elif data_type == 'gauss_rule2':
            y[:, 0] = (views[0][:,0]*views[1][:,1] > 0.3) | (views[0][:,2]*views[1][:,3] > 0.3)

        elif data_type == 'gauss_rule3':
            y[:, 0] = (views[0][:,0]*views[1][:,1] > 0.3) | (views[0][:,0]*views[1][:,2] > 0.3)
        
        elif data_type == 'gauss_rule4':
            y[:, 0] = (views[0][:,1]*views[1][:,2] > 0.3) | (views[0][:,7]*views[1][:,10] > 0.3)
            
        elif data_type == 'gauss_rule5':
            y[:, 0] = (views[0][:,1]*views[1][:,2] > 0.3) \
                    | ((views[0][:,7]>0.5) | (views[1][:,7]>0.5)) 
            
        elif data_type == '3views_gauss_rule5':
            y[:, 0] = (views[0][:,1]*views[1][:,2] > 0.3) \
                    | ((views[2][:,7]>0.5) | (views[1][:,7]>0.5)) 
    
    y[:,1] = 1-y[:, 0]
    print("validate:",np.unique(y[:,1], return_counts=True))
    return y[:,1]

def generate_views_ai_gt(views, data_type:str):
    
    print(f"generating ai_gt_for {data_type}")

    # Number of samples and features
    gt_views = [np.zeros((view.shape[1])) for view in views]

    # For each data_type
    if (data_type == 'syn1') or (data_type =='large_syn1') or (data_type == 'corr_syn1') or (data_type == 'small_corr_syn1'):
        gt_views[0][0] = 1
        gt_views[0][2] = 1
        gt_views[1][1] = 1
        gt_views[1][3] = 1

    elif (data_type == 'red_syn1'):
        gt_views[0][0] = 1
        gt_views[0][2] = 1
        gt_views[1][1] = 1
        gt_views[1][3] = 1
        gt_views[1][4] = 1
        
    elif data_type == 'syn2':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][2] = 1
        gt_views[1][3] = 1

    elif data_type == 'syn3':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][2] = 1
        gt_views[1][3] = 1
        gt_views[0][6] = 1
        gt_views[1][7] = 1
        
    elif (data_type == 'syn4') or (data_type == 'large_syn4'):
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][6] = 1
        gt_views[1][7] = 1
        gt_views[0][2] = 1
        gt_views[1][3] = 1
        
    elif (data_type == 'syn4_overlap'):
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][6] = 1
        gt_views[1][7] = 1
        gt_views[0][2] = 1
        gt_views[1][3] = 1
        
    elif (data_type == 'large_syn5'):
        gt_views[0][0:6] = 1
        gt_views[1][1:7] = 1
        gt_views[0][9] = 1
        gt_views[1][10] = 1
    
    elif data_type == '3views_syn1':
        gt_views[0][0] = 1
        gt_views[0][5] = 1
        gt_views[1][1] = 1
        gt_views[1][5] = 1
        gt_views[2][2] = 1
        gt_views[2][5] = 1
        
    elif data_type == '3views_syn2':
        gt_views[0][0] = 1
        gt_views[0][5] = 1
        gt_views[1][1] = 1
        gt_views[1][2] = 1
        gt_views[1][5] = 1
        gt_views[2][2] = 1
        gt_views[2][5] = 1

    elif data_type == '3views_syn3':
        gt_views[0][2] = 1
        gt_views[0][5] = 1
        gt_views[1][0] = 1
        gt_views[1][5] = 1
        gt_views[2][1] = 1
        gt_views[2][2] = 1
        gt_views[2][5] = 1
        
    elif data_type == '4views_syn1':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[2][2] = 1
        gt_views[3][3] = 1
        gt_views[0][5] = 1
        gt_views[1][5] = 1
        gt_views[2][5] = 1
        gt_views[3][5] = 1       
    
    elif data_type == '3views_dummy':
        gt_views[0][0] = 1
        gt_views[0][2] = 1
        gt_views[1][1] = 1
        gt_views[1][3] = 1

    elif data_type == 'gauss_rule1':
        gt_views[0][1] = 1
        gt_views[1][2] = 1

    elif data_type == 'gauss_rule2':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][2] = 1
        gt_views[1][3] = 1

    elif data_type == 'gauss_rule3':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[1][2] = 1

    elif data_type == 'gauss_rule4':
        gt_views[0][1] = 1
        gt_views[1][2] = 1
        gt_views[0][7] = 1
        gt_views[1][10] = 1

    elif data_type == 'gauss_rule5':
        gt_views[0][1] = 1
        gt_views[0][7] = 1
        gt_views[1][2] = 1
        gt_views[1][7] = 1
    
    elif data_type == '3views_gauss_rule5':
        gt_views[0][1] = 1
        gt_views[1][2] = 1
        gt_views[2][7] = 1
        gt_views[1][7] = 1

    else :
        print(f"there's no datatype:{data_type}")
    return gt_views

def generate_views_syn_gt(views, data_type:str):
    # Number of samples and features

    print(f"generating syn_gt_for {data_type}")
    gt_views = [np.zeros((view.shape[1])) for view in views]

    # For each data_type
    if (data_type == 'syn1') or (data_type =='large_syn1') or (data_type == 'red_syn1') or (data_type == 'corr_syn1') or (data_type == 'small_corr_syn1'):
        gt_views[0][0] = 1
        gt_views[1][1] = 1
            
    elif data_type == 'syn2':
        gt_views=gt_views  #no synergic features

    elif data_type == 'syn3':
        gt_views[0][0] = 1
        gt_views[1][1] = 1

    elif (data_type == 'syn4') or (data_type == 'large_syn4'):
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][6] = 1
        gt_views[1][7] = 1
        
    elif (data_type == 'large_syn5'):
        gt_views[0][0:6] = 1
        gt_views[1][1:7] = 1
        
    elif data_type == '3views_syn1':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[2][2] = 1
        
    elif data_type == '3views_syn2':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[1][2] = 1
        gt_views[2][2] = 1
        
    elif data_type == '3views_syn3':
        gt_views[0][2] = 1
        gt_views[1][0] = 1
        gt_views[2][1] = 1
        gt_views[2][2] = 1

    elif data_type == '4views_syn1':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[2][2] = 1
        gt_views[3][3] = 1 
    
    elif data_type == '3views_dummy':
        gt_views[0][0] = 1
        gt_views[1][1] = 1

    elif data_type == 'gauss_rule1':
        gt_views=gt_views  #no synergic features
        
    elif data_type == 'gauss_rule2':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[0][2] = 1
        gt_views[1][3] = 1

    elif data_type == 'gauss_rule3':
        gt_views[0][0] = 1
        gt_views[1][1] = 1
        gt_views[1][2] = 1

    elif data_type == 'gauss_rule4':
        gt_views[0][1] = 1
        gt_views[1][2] = 1
        gt_views[0][7] = 1
        gt_views[1][10] = 1
        
    elif data_type == 'gauss_rule5':
        gt_views[0][1] = 1
        gt_views[1][2] = 1

    elif data_type == '3views_gauss_rule5':
        gt_views[0][1] = 1
        gt_views[1][2] = 1
        
    else :
        print(f"there's no datatype:{data_type}")

    return gt_views

def generate_multi_dataset(n, data_type:str, seed, dims, rho=0, syn_rho=0):
    
    # Seed
    np.random.seed(seed)
    
    # x generation
    
    if rho == 0:
        views = generate_views(n, dims)
    else: # generate correlated X data
        print("generate correlated features")
        views = generate_corr_features(n, dims, rho)
    
    # y generation
    y = generate_y_for_views(views, data_type, syn_rho)
    # ground truth generation
    ai_gt  = generate_views_ai_gt(views, data_type)
    syn_gt  = generate_views_syn_gt(views, data_type)

    return views, y, (ai_gt, syn_gt)

def main(args):
    views, y, gt = generate_multi_dataset(n=args.sample_n, 
                                    data_type=args.datatype, 
                                    seed=args.seed, 
                                    dims=list(map(int,args.views_dims)),
                                    rho=args.rho,
                    )
    
    for i, view in enumerate(views):
        print(f"{i}-th view shape:", view.shape)
        print(f"{i}-th gt shape", gt[0][i].shape)
    print("y shape:", y.shape)
    return views, y, gt
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_n', default=20000)
    parser.add_argument('--datatype', default='gauss_rule5')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--views_dims', nargs='+', type=int, default=[10,10])
    parser.add_argument('--rho', type=float, default=0)
    parser.add_argument('--syn_rho', type=float, default=0)
    parser.add_argument('--save_dir', default='/mnt/storage/personal/chkim/synergy/data/')
    args = parser.parse_args()

    views, y, gt = main(args)
    data = (views, y)  # views(np.array)views[v] = v-th view 

    if not os.path.isdir(args.save_dir) :
        os.mkdir(args.save_dir)
    with open(args.save_dir+f'{args.datatype}.pickle', 'wb') as f:
        pickle.dump(data, f)
    with open(args.save_dir+f'{args.datatype}_gt.pickle', 'wb') as f:
        pickle.dump(gt, f)
    print(f'{args.datatype} synthetic data save done')

#3views_syn3 10, 20, 10
#gauss_rule5 10, 10