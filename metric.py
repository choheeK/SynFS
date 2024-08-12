# Necessary packages
# reference INVASE utils.py modified for global feature selection 
# gsim, tpr_fdr from commpfs

from functools import reduce
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import pdb    
import torch
from torch.utils.data import DataLoader

from data.dataset_util import NumpyDataset, SimpleDataset

def sklearn_accuracy(x, y):
    x = x > 0.5
    return 100 * accuracy_score(y, x)

def accuracy(x, y):
    # Accuracy.
    acc = 100 * torch.sum(torch.argmax(x, dim=-1) == y) / len(y)
    return acc.item()

def gsim(true_groups, predicted_groups):
    # Returns gsim, number of true groups, and number of discovered groups, given
    # true groups and predicted groups as input.
    gsim = 0
    if len(true_groups) == 0:  # i.e. we don't know the ground truth.
        return -1, len(true_groups), len(predicted_groups)
    for group in true_groups:
        if group.size == 0:
            true_groups.remove(group)
    if len(predicted_groups) > 0:
        for g in true_groups:
            current_max = 0
            for g_hat in predicted_groups:
                jac = np.intersect1d(g, g_hat).size / np.union1d(g, g_hat).size
                if jac == 1:
                    current_max = 1
                    break
                if jac > current_max:
                    current_max = jac
            gsim += current_max
        gsim /= max(len(true_groups), len(predicted_groups))
        return gsim, len(true_groups), len(predicted_groups)
    else:  # We didn't find anything.
        return 0, len(true_groups), len(predicted_groups)

def compfs_jaccard(true_groups, predicted_groups, views_dims):
    # return jacard for true groups (ns, s), predicted_groups
    # predicted group will be re-sorted to ns, s (if |element| > 1 and elements are from different views -> s, else ns) folloiwng comfps def
    if len(true_groups) == 0:  # i.e. we don't know the ground truth.
        return -1, len(true_groups), len(predicted_groups)
    ns_true, s_true = true_groups  # non-synergistic, synergistic
    if len(predicted_groups) > 0:
        ns_pred = []
        s_pred = []
        for g in predicted_groups:
            cum_sum = np.cumsum(views_dims)
            # checknig the features view 
            bin_list = []
            for feature in g:    
                for bin_idx in range(len(cum_sum)):
                    if bin_idx == 0:
                        if feature < cum_sum[bin_idx]:
                            bin_list.append(bin_idx)
                            #print("feature", feature, "bin", bin_idx)
                    else: 
                        if (feature >= cum_sum[bin_idx-1]) and (feature < cum_sum[bin_idx]) :
                            bin_list.append(bin_idx)
            if (np.unique(bin_list).size > 1) & (len(g) > 1) : # not from the same view, more than one feature 
                s_pred.append(g)
            else:
                ns_pred.append(g)
        print(f"s_pred |{s_pred} \n")
        print(f"ns_pred |{ns_pred} \n")
        if len(ns_pred) > 0: 
            ns_pred = np.concatenate([np.array(ns) for ns in ns_pred])
            ns_jac = np.intersect1d(ns_true, ns_pred).size / np.union1d(ns_true, ns_pred).size
        elif len(ns_pred) == len(ns_true): # no ground truth 
            ns_jac = 1
        else:
            ns_jac = 0
        if len(s_pred) > 0: 
            s_pred = np.concatenate([np.array(s) for s in s_pred])
            s_jac = np.intersect1d(s_true, s_pred).size / np.union1d(s_true, s_pred).size
        elif len(s_pred) == len(s_true): # no ground truth 
            s_jac = 1
        else:
            s_jac =0
        return (ns_jac + s_jac) / 2, len(true_groups), len(predicted_groups)
    else:  # we didn't find anything 
        return 0, len(true_groups), len(predicted_groups)

def strict_jaccard(true_groups, predicted_groups):
    # return jacard for true groups (ns, s), predicted_groups
    # predicted group will be re-sorted to ns, s (if element==1 -> ns) folloiwng comfps def
    if len(true_groups) == 0:  # i.e. we don't know the ground truth.
        return -1, len(true_groups), len(predicted_groups)

    if len(predicted_groups) == 0 or all(len(sg) == 0 for sg in predicted_groups): # we didn't find anything
        return 0, len(true_groups), len(predicted_groups)
    
    ns_true, s_true = true_groups  # non-synergic, synergic
    ns_pred, s_pred = predicted_groups

    if len(ns_pred) > 0: 
        ns_jac = np.intersect1d(ns_true, ns_pred).size / np.union1d(ns_true, ns_pred).size
    elif len(ns_pred) == len(ns_true):  # no ground truth 
        ns_jac = 1
    else:
        ns_jac = 0
        
    if len(s_pred) > 0: 
        s_jac = np.intersect1d(s_true, s_pred).size / np.union1d(s_true, s_pred).size
    elif len(s_pred) == len(s_true):  # no ground truth 
        ns_jac = 1
    else:
        s_jac =0
    return (ns_jac + s_jac) / 2, len(true_groups), len(predicted_groups)



def tpr_fdr(true_groups, predicted_groups):
    # True positive rate and false discovery rate.

    if len(true_groups) == 0:  # Ground truth not known.
        return -1, -1

    if len(predicted_groups) == 0:
        return 0.0, 0.0
    
    if isinstance(predicted_groups, list): # groups of selected features
        if all(len(sg) == 0 for sg in predicted_groups):
            return 0.0, 0.0

    predicted_features = np.unique(reduce(np.union1d, predicted_groups))
    true_features = np.unique(reduce(np.union1d, true_groups))

    overlap = np.intersect1d(predicted_features, true_features).size
    tpr = 100 * overlap / len(true_features)
    fdr = (
        100 * (len(predicted_features) - overlap) / len(predicted_features)
    )  # If len(predicted_features) != 0 else 0.0.
    return tpr, fdr



def feature_performance_metric (ground_truth, importance_score, threshold= 0.5, mode = None):
    """Performance metrics for feature importance (TPR and FDR).
    GLOBAL ( different from invase sampel wise metric)

    Args:
    - ground_truth: ground truth feature importance
    - importance_score: computed importance scores for each feature <mu>
    - threshold : threshold to depreciate the prob
    - mode : by gate , if by gate, get tpr & fdr per gate

    Returns:
    - TPR : tre positive rate , the higher the better 
    - FDR : false discovery rate ,lower the better 

    """

    # For each sample
    importance_score = np.where(importance_score > threshold , importance_score, 0)

    if mode == 'bygate' : 
        tpr_nom = importance_score * ground_truth
        tpr_den = ground_truth
        tpr = 100 * tpr_nom.astype(float)/(tpr_den + 1e-8).astype(float)
        
        fdr_nom = importance_score * (1-ground_truth)
        fdr_den = importance_score
        fdr = 100 * fdr_nom.astype(float)/(fdr_den+1e-8).astype(float)
        
    else : 
        tpr_nom = np.sum(importance_score * ground_truth)
        tpr_den = np.sum(ground_truth)
        tpr = 100 * float(tpr_nom)/float(tpr_den + 1e-8)
            
        # fdr
        fdr_nom = np.sum(importance_score * (1-ground_truth))
        fdr_den = np.sum(importance_score)
        fdr = 100 * float(fdr_nom)/float(fdr_den+1e-8)

    return tpr, fdr



def auroc_prc(y_test ,logits) :
    auroc = roc_auc_score([y.numpy() for y in y_test], logits[:,1])
    auprc = average_precision_score([y.numpy() for y in y_test], logits[:,1])
    print(f'auroc : {auroc:.3f}','\n',f'auprc : {auprc:.3f}')
    
def standard_metrics(y_train, y_test, logits, verbose=False):
    if len(y_test) != len(logits):
        y_test = y_test[:len(logits)]
    if isinstance(y_test, np.object_) :
        y_test = np.array([y.numpy() for y in y_test])
    
    def np_softmax(target, all):
        softmax = np.exp(target) / np.sum(np.exp(all) ,axis=1)
        return softmax
        
    threshold = np.sum(y_train[y_train==1])/ len(y_train)
    threshold = threshold
    y_prob = np_softmax(logits[:,1], logits)
    y_pred = np.where(y_prob > threshold, 1, 0)
    
    auroc = roc_auc_score(y_test, logits[:,1])
    auprc = average_precision_score(y_test, logits[:,1])
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if verbose:
        print(f"auroc | {auroc.item():.3f}, auprc | {auprc.item():.3f}, accuracy | {accuracy.item():.3f}, f1 | {f1.item():.3f}")
    return auroc, auprc, accuracy, f1 

def test_metric(metric_dict, key, logits, tr_y, te_y, verbose=False):
    for i, model in enumerate(metric_dict[key].keys()): 
        results = standard_metrics(tr_y, te_y, logits[i], verbose=verbose)   
        metric_dict[key][model].append(results)
        
def benchmark_predictive_inference(X_data, y_data, model, y_train):
    """
    Compute predictive performance metric (for benchmark)
    input : X_set, y_set
    output : auroc, auprc, accuracy, f1
    """ 

    data = NumpyDataset(X_data, y_data, classification=True)
    loader = DataLoader(data, batch_size=len(y_data))
    # Predictive performance all inf로 수정 
    res = []
    for X, y in loader:
        logits = model.predict(X)
        if torch.is_tensor(logits):
            res.append(logits.detach().cpu().numpy())
        else:
            res.append(logits)
    logits = res[0]
            
    return standard_metrics(y_train, y_data, logits, verbose=False)

def mvs_predictive_inference(X_set, y_set, mvs, tr_y):
    """
    Compute predictive performance metric
    input : X_set, y_set
    output : auroc, auprc, accuracy, f1
    """ 
    data = SimpleDataset(X_set, y_set, device='cuda')
    loader = DataLoader(data, batch_size=len(y_set))
    
    # Predictive performance all inf로 수정 
    res = []
    for x, y in loader:
        logits = mvs.predict(x, X_set)
        res.append(logits.detach().cpu().numpy())
    logits = np.concatenate(res)
    return standard_metrics(tr_y, y_set, logits, verbose=False)


def mlp_predictive_inference(X_set, y_set, model, tr_y):
    """
    mlp input: set of data 
    """ 
    loader = model.get_dataloader(X_set, y_set, batch_size=len(y_set))
    
    # Predictive performance all inf로 수정 
    res = []
    for batch in loader:
        view = torch.cat(batch[0], dim=1)
        logits = model.predict(view)
        res.append(logits.detach().cpu().numpy())
    logits = np.concatenate(res)
            
    return standard_metrics(tr_y, y_set, logits, verbose=False)