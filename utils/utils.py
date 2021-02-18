import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, average_precision_score, ndcg_score

def glorot(shape, name=None, scale=1.):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[-1]+shape[-2])) * scale
    initial = np.random.uniform(-init_range, init_range, shape)
    return torch.Tensor(initial)

def evaluate_auc(pred, label):
    if np.sum(label) == 0:
        m=[1]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    if np.sum(label) == len(label):
        m=[0]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    res=roc_auc_score(y_score=pred, y_true=label)
    return res

def evaluate_acc(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return accuracy_score(y_pred=res, y_true=label)

def evaluate_f1_score(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return f1_score(y_pred=res, y_true=label)

def evaluate_logloss(pred, label):
    if np.sum(label) == 0:
        m=[1]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    if np.sum(label) == len(label):
        m=[0]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])    
    res = log_loss(y_true=label, y_pred=pred,eps=1e-7, normalize=True)
    return res

def evaluate_ndcg(k, pred_list, label_list, batch_size, list_length):
    preds = np.array_split(pred_list.flatten(),pred_list.shape[0]/list_length)
    labels = np.array_split(label_list.flatten(),pred_list.shape[0]/list_length)
    NDCG = ndcg_score(y_true=labels,y_score=preds,k=k)
    '''
    ndcg=[]
    for pred,label in zip(preds,labels):
        
        idx = np.argsort(-pred)
        accumulation = 0.0
        normalization = 0.0 
        sorted_label = label[np.argsort(-label)]
        for i in range(0,k):
            accumulation += float(label[idx[i]])/ np.log2(i+2.0)
            normalization  += float(sorted_label[i])/ np.log2(i+2.0)
        if normalization == 0:
            ndcg.append(0)
        else:
            ndcg.append(accumulation/normalization)
        
    NDCG=np.mean(ndcg)
    '''
    return NDCG

def evaluate_map(k, pred_list, label_list, batch_size, list_length):
    preds = np.array_split(pred_list.flatten(),pred_list.shape[0]/list_length)
    labels = np.array_split(label_list.flatten(),pred_list.shape[0]/list_length)
    Map=[]
    for pred,label in zip(preds,labels):
        '''
        if np.sum(label) == 0:
            m=[1]
            label=np.vstack([label,m])
            pred=np.vstack([pred,m])  
        if np.sum(label) == len(label):
            m=[0]
            label=np.vstack([label,m])
            pred=np.vstack([pred,m])  
        Map.append(average_precision_score(y_true=label.flatten(),y_score=pred.flatten()))
        '''
        idx = np.argsort(-pred)
        accumulation = 0.0
        count = 0.0
        for i in range(0,k):
            if label[idx[i]] == 1 and pred[idx[i]] >= 0.5:
                accumulation += (count+1.0)/ (i+1.0)
                count += 1.0
        x = label.sum()
        if x == 0 :
            Map.append(0)
        else:
            Map.append(float(accumulation/k))
    MAP=np.mean(Map)
    return MAP
