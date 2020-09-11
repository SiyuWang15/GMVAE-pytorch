import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

class GenerationNet(nn.Module):
    def __init__(self, v_dim, h_dim, w_dim, n_classes):
        super(GenerationNet, self).__init__()
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.w_dim = w_dim
        self.n_classes = n_classes

        # P(h|w,c) c = 1,2,3,...,K
        self.Ph_wc_mean_list = list()
        for i in range(self.n_classes):
            Ph_wc_mean = nn.Sequential(
                nn.Linear(w_dim, 512), 
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(), 
                nn.Linear(512, h_dim)
            )
            self.Ph_wc_mean_list.append(Ph_wc_mean)
        self.Ph_wc_mean_list = nn.ModuleList(self.Ph_wc_mean_list)
        self.Ph_wc_var_list = list()
        for i in range(self.n_classes):
            Ph_wc_var = nn.Sequential(
                nn.Linear(w_dim, 512), 
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(), 
                nn.Linear(512, h_dim)
            )
            self.Ph_wc_var_list.append(Ph_wc_var)
        self.Ph_wc_var_list = nn.ModuleList(self.Ph_wc_var_list)
        
        # P(v|h)
        self.Pv_h_mean = nn.Sequential(
            nn.Linear(h_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, v_dim)
        )
        self.Pv_h_var = nn.Sequential(
            nn.Linear(h_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, v_dim)
        )
    
    def gen_h(self, w, c):
        h_mean = self.Ph_wc_mean_list[c](w)
        h_var = self.Ph_wc_var_list[c](w)
        return h_mean, h_var
    
    def gen_v(self, h):
        h = h.view(h.shape[0], -1)
        v_mean = self.Pv_h_mean(h)
        v_var = self.Pv_h_var(h)
        return v_mean, v_var
    
    def forward(self, w, c):
        h, _ = self.gen_h(w, c)
        v, _ = self.gen_v(h)
        return v