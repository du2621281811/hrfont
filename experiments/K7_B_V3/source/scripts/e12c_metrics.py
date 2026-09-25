"""Exact binary ROC-AUC by average ranks, including score ties."""
import numpy as np


def roc_auc_score(labels,scores):
    y=np.asarray(labels);s=np.asarray(scores,dtype=np.float64)
    assert y.ndim==s.ndim==1 and y.shape==s.shape and np.isfinite(s).all()
    assert set(y.tolist())=={0,1}
    order=np.argsort(s,kind='stable');ranks=np.empty(len(s),dtype=np.float64)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    n=int(y.sum());m=len(y)-n
    return float((ranks[y==1].sum()-n*(n+1)/2)/(n*m))
