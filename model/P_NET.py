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
        
        self.Ph_wc_mean = nn.Sequential(
            nn.Linear(w_dim + n_classes, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, h_dim)
        )

        self.Ph_wc_var = nn.Sequential(
            nn.Linear(w_dim + n_classes, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, h_dim)
        )

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

    def infer_c(self, w_sample, h_sample):
        # w, h : [M, bs, w_dim or h_dim]
        M, bs, _ = w_sample.shape
        w_sample = w_sample.unsqueeze(2).expand(-1,-1,self.n_classes,-1) # w: [M, bs, n_classes, w_dim]
        h_sample = h_sample.unsqueeze(2).expand(-1,-1,self.n_classes,-1)
        c = torch.eye(self.n_classes).expand(M, bs, -1, -1).cuda() # c: [M, bs, n_classes, n_classes]
        h_mean, h_logstd = self.gen_h(w_sample, c)  # [M, bs, n_classes, h_dim]
        ph_wc = torch.pow(h_sample - h_mean, 2) / (h_logstd * 2).exp()
        ph_wc = torch.sum(ph_wc, axis = -1) / 2. # [M, bs, n_classes]
        ph_wc = ph_wc.exp() / torch.sqrt(torch.sum(h_logstd * 2, axis = -1).exp())
        probs = torch.sum(ph_wc, axis = -1, keepdim = True).expand(-1,-1,self.n_classes)
        probs = ph_wc / probs
        return torch.mean(probs, axis = 0)
    
    def gen_h(self, w, c):
        concat = torch.cat([w, c], axis = -1)
        h_mean = self.Ph_wc_mean(concat)
        h_var = self.Ph_wc_var(concat)
        return h_mean, h_var
        # h_mean = self.Ph_wc_mean_list[c](w)
        # h_var = self.Ph_wc_var_list[c](w)
        # return h_mean, h_var
    
    def gen_v(self, h):
        v_mean = self.Pv_h_mean(h)
        v_var = self.Pv_h_var(h)
        return v_mean, v_var
    
    def forward(self, w, c):
        h, _ = self.gen_h(w, c)
        v, _ = self.gen_v(h)
        return v