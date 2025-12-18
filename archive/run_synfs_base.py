# hyp search for base models, synfs 

import numpy as np
import torch
import torch.nn as nn
import sys 
sys.path.append('/home/chkim/Synergy/Synergy/')
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import pdb
import os.path as osp


from models.SynFS import SynFS
from util import make_lambda_threshold
from metric import accuracy, tpr_fdr, strict_jaccard, mlp_predictive_inference
from data.dataset_util import NumpyDataset, train_val_test_split, SimpleDataset
from data.data_generating import *



config = {
    "data_config" :{
        "datatype": 'syn1',
        'views_dims': [250, 250],
    },
    "model": SynFS,
    "model_config": { 
      "hidden_dims": [32, 32], 
      "s_lam": 0.1,
      "ns_lam": 1.07,
      "output_dim": 2, 
      "batch_norm": True, 
      "dropout": True, 
      "activation": "relu", 
      "sigma": 0.5, 
      "s_alpha": 0.05,
      "ns_alpha": 0.25,
      "learning_rate": 1e-3, 
      "s_learning_rate":0.001,
      "batch_size": 250, 
      "weight_decay": 1e-4,
      "patience":5,
      "delta":0.0,
      "epoch": 90,
      "selector_mean": 0.5,
      "hyp_c": 1,
      "lr_decay": 1.,
      "s_lr_decay": 1.,
      "optimizer": 'Adam',
      'shap_m': 10,
      "early_stop": False,
    },
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["device"] = device

def set_seed(x):
    # Set a consistent seed, so we can run across different runs.
    x *= 10000
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--views_dims', nargs='+', type=int,  default=config['data_config']['views_dims'])
    parser.add_argument('--datatype', default=config['data_config']['datatype'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=config['model_config']['hidden_dims'])
    parser.add_argument('--data_n', type=int, default= 20000)
    
    parser.add_argument('--output_dim', default=config['model_config']['output_dim'])
    parser.add_argument('--batch_norm', default=config['model_config']['batch_norm'])
    parser.add_argument('--dropout', default=config['model_config']['dropout'])
    parser.add_argument('--activation', default=config['model_config']['activation'])
    parser.add_argument('--sigma', default=config['model_config']['sigma'])
    parser.add_argument('--ns_lam', type=float, default=config['model_config']['ns_lam'])
    parser.add_argument('--s_lam', type=float, default=config['model_config']['s_lam'])
    parser.add_argument('--learning_rate', type=float, default=config['model_config']['learning_rate'])
    parser.add_argument('--s_alpha', type=float, default=config['model_config']['s_alpha'])
    parser.add_argument('--ns_alpha', type=float, default=config['model_config']['ns_alpha'])
    parser.add_argument('--s_learning_rate', type=float, default=config['model_config']['s_learning_rate'])
    parser.add_argument('--batch_size', type=int, default=config['model_config']['batch_size'])
    parser.add_argument('--weight_decay', type=float, default=config['model_config']['weight_decay'])

    parser.add_argument('--epoch', type=int, default=config['model_config']['epoch'])
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=config['model_config']['patience'])
    parser.add_argument('--delta', type=float, default=config['model_config']['delta'])
    parser.add_argument('--lr_decay', type=float, default=config['model_config']['lr_decay'])
    parser.add_argument('--s_lr_decay', type=float, default=config['model_config']['s_lr_decay'])
    parser.add_argument('--selector_mean', type=float, default=config['model_config']['selector_mean'])
    parser.add_argument('--optimizer', type=str, default=config['model_config']['optimizer'])

    parser.add_argument('--notraining', action='store_true', default=False)
    parser.add_argument('--shap_m', type=int, default=config['model_config']['shap_m'])

    parser.add_argument('--save_dir', default='./experiments/synthetic/')
    parser.add_argument('--experiment_no', type=int, default=0)
    parser.add_argument('--save_result', action = 'store_true', default=False)
    parser.add_argument('--bestmodel', action='store_true', default=False)
    parser.add_argument('--track', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    
    return parser.parse_args()
    
def main(config):
    
    config = vars(config)
    print(config)
    set_seed(config['experiment_no'])
    #  Generate_data
    views, y, (ai_gt, syn_gt) = generate_multi_dataset(config['data_n'], 
                                                        config['datatype'], 
                                                        seed=0, 
                                                        dims=config['views_dims'], 
                                                        rho=0, 
                                                        syn_rho=0)
    #views, y, (ai_gt, syn_gt) = generate_multi_dataset(config.data_n, 
    #                                                config.datatype, 
    #                                                seed=0, 
    #                                                dims=config.views_dims, 
    #                                                rho=0, 
    #                                                syn_rho=0)
    data = (views, y)  # multi view
    
    ns_gt = [ai - syn for ai, syn in zip(ai_gt, syn_gt)]
    ns_gt = np.where(np.concatenate(ns_gt))[0]
    s_gt = np.where(np.concatenate(syn_gt))[0]
    ground_truth_groups = [ns_gt, s_gt]
    
    tr_X_set, tr_y, va_X_set, va_y, te_X_set, te_y, scaler = train_val_test_split(data, test_size=0.2, val_size=0.2, seed=0, scaling=True)
    
    # Train SynFS
    synfs = SynFS(config)
    synfs.fit(tr_X_set, tr_y, va_X_set, va_y, nr_epochs=config['epoch'])
    
    def inference(X_set, y_set): 
        data = SimpleDataset(X_set, y_set, device='cuda')
        loader = DataLoader(data, batch_size=len(y_set))
        # all inf로 수정 
        res = []
        for x, y in loader:
            logits = synfs.predict(x, X_set)
            res.append(logits.detach().cpu().numpy())
        logits = np.concatenate(res)
                
        # Metric 
        def np_softmax(target, all):
            softmax = np.exp(target) / np.sum(np.exp(all) ,axis=1)
            return softmax
        threshold = np.sum(tr_y[tr_y==1])/ len(tr_y)
        threshold = threshold
        y_prob = np_softmax(logits[:,1], logits)
        y_pred = np.where(y_prob > threshold, 1, 0)

        auroc = roc_auc_score(y_set, logits[:,1])
        auprc = average_precision_score(y_set, logits[:,1])
        accuracy = accuracy_score(y_set, y_pred)
        f1 = f1_score(y_set, y_pred)
        print(f"auroc | {auroc:.3f}, auprc | {auprc:.3f}, accuracy | {accuracy:.3f}, f1 | {f1:.3f}")

        # Get group similarity and group structure.
        s = [gate.cpu().numpy() for gate in synfs.get_detached_mu(synfs.s_model)]
        ns = [gate.cpu().numpy() for gate in synfs.get_detached_mu(synfs.ns_model)]
        ns_group = np.where(np.concatenate(ns)>0.7)[0]
        s_group = np.where(np.concatenate(s)>0.7)[0]
        learnt_groups = [ns_group, s_group]

        # Get group similarity and group structure.
        tpr, fdr = tpr_fdr(ground_truth_groups, learnt_groups)
        group_sim, ntrue, npredicted = strict_jaccard(ground_truth_groups, learnt_groups)
        print("ground_truth_groups ", ground_truth_groups)
        print("learnt_groups ", learnt_groups)
        print("\n\nGroup Structure:")
        print(
            "Group Similarity: {:.3f}, True Positive Rate: {:.3f}%, False Discovery Rate: {:.3f}%".format(
                group_sim, tpr, fdr
            )
        )
        print(
            "Number of True Groups: {}, Number of Predicted Groups: {}".format(
                ntrue, npredicted
            )
        )
        return group_sim, tpr, fdr, auroc, auprc, accuracy, f1

    
    # Val
    if args.test is False:
        print("Validation")
        if config['bestmodel']:
            synfs.load_ckpt()

        group_sim, tpr, fdr, auroc, auprc, accuracy, f1 = inference(va_X_set, va_y)

    # Test        
    if args.test:
        print("Test")
        if config['bestmodel']:
            synfs.load_ckpt()
            
        group_sim, tpr, fdr, auroc, auprc, accuracy, f1 = inference(te_X_set, te_y)
            
        # Make folder for results.
        folder = osp.join(
            config.save_dir,
            "results",
            config.datatype,
            "synfs",
            "run_" + str(config.experiment_no),
        )
        if config.save_result:
            if not osp.exists(folder):
                os.makedirs(folder)
                
            np.save(osp.join(folder, "gsim.npy"), np.array([group_sim]))
            np.save(osp.join(folder, "true_positive_rate.npy"), np.array([tpr]))
            np.save(osp.join(folder, "false_discovery_rate.npy"), np.array([fdr]))
            np.save(osp.join(folder, "auroc.npy"), np.array([auroc]))
            np.save(osp.join(folder, "auprc.npy"), np.array([auprc]))
            np.save(osp.join(folder, "accuracy.npy"), np.array([accuracy]))
            np.save(osp.join(folder, "f1.npy"), np.array([f1]))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    # CUDA_VISBLE_DEVICES=3 OMP_NUM_THREADS=10 python run_mvs_base.py --views_dims 50 50 --s_lam 0.03 --ns_lam 0.3
