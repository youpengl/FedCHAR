import numpy as np
from copy import deepcopy
import torch


# gaussian_attack
def A2(v, malicious_cid=None, scaling_factor=None):
    if malicious_cid == None:
        flatten_vi = torch.cat([vv.reshape((-1, 1)) for vv in v], dim=0) 
        std = torch.std(flatten_vi, unbiased=False).item()
        v = [torch.tensor(np.random.normal(0, std, size=vv.shape).astype('float32')).to(flatten_vi.device) for vv in v]
    
    else:
        for i in malicious_cid:
            flatten_vi = torch.cat([vv.reshape((-1, 1)) for vv in v[i]], dim=0) 
            std = torch.std(flatten_vi, unbiased=False).item()
            v[i] = [torch.tensor(np.random.normal(0, std, size=vv.shape).astype('float32')).to(flatten_vi.device) for vv in v[i]]
    
    return v

# reverse_attack
def A4(v, malicious_cid=None, scaling_factor=None):
    if malicious_cid == None:
        v = [-vv for vv in v]
    
    else:
        for i in malicious_cid:
            v[i] = [-vv for vv in v[i]]
    
    return v

# scaling_attack
def A3(v, malicious_cid=None, scaling_factor=None):

    if malicious_cid == None:
        v = [vv * scaling_factor for vv in v]

    elif scaling_factor != None:
        for i in malicious_cid:
            v[i] = [vv * scaling_factor for vv in v[i]]
            
    else:
        scaling_factor = len(v) 
        for i in malicious_cid:
            v[i] = [vv * scaling_factor for vv in v[i]]
    
    return v