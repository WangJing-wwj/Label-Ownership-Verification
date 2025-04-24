
import os
import pdb
import torch
from torchvision import transforms
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
import umap
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

class SCALE_UP:
    """Identify and filter malicious testing samples (SCALE-UP).

    Args:
        model (nn.Module): The original backdoored model.
        scale_set (List):  The hyper-parameter for a set of scaling factors. Each integer n in the set scales the pixel values of an input image "x" by a factor of n.
        T (float): The hyper-parameter for defender-specified threshold T. If SPC(x) > T , we deem it as a backdoor sample.
        valset (Dataset): In data-limited scaled prediction consistency analysis, we assume that defenders have a few benign samples from each class.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
    """
    def __init__(self, model):
        self.model = model
        self.model.cuda()
        self.model.eval()

    
    def probability_available_test(self,target_label, spc_dataset,sample_number):
        spc_dataloader = torch.utils.data.DataLoader(spc_dataset, batch_size=128, shuffle=False)
        self.model.eval()
        all_clean_pred=[]
        all_dirty_pred=[]
        with torch.no_grad():
            for x_t, y_t, x_d, y_true, _ in spc_dataloader: #纯脏样本xt_yt, 对应的干净样本xd_ytrue
                imgs = x_d
                imgs = imgs.cuda()  # batch * channels * hight * width
                clean_pred =self.model(imgs)
                all_clean_pred.append(F.softmax(clean_pred, dim=1)[:,target_label])
                
                dirty_imgs = x_t
                dirty_imgs = dirty_imgs.cuda()  # batch * channels * hight * width
                dirty_pred =self.model(dirty_imgs)
                all_dirty_pred.append(F.softmax(dirty_pred,dim=1)[:,target_label])
        
        all_clean_pred = torch.cat(all_clean_pred, dim=0)[:sample_number]
        all_dirty_pred = torch.cat(all_dirty_pred, dim=0)[:sample_number]
        all_delta_probability=all_dirty_pred-all_clean_pred
        if np.average(all_delta_probability.cpu().numpy())<0.01: 
            all_clean_pred=all_clean_pred+0.01
        all_clean_pred[0] += (0.0000001 if torch.all(all_clean_pred == 0) else 0)  
        all_dirty_pred[0] += (0.0000001 if torch.all(all_dirty_pred == 0) else 0) 
        t_stat, p_value = stats.ttest_rel(all_dirty_pred.cpu().numpy(), all_clean_pred.cpu().numpy(), alternative='greater')
        delta_P=np.average(all_delta_probability.cpu().numpy())

        return delta_P,t_stat, p_value
