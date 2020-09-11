import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

# Inference Network
class InferenceNet(nn.Module):
    def __init__(self, v_dim, h_dim, w_dim, n_classes):
        super(InferenceNet, self).__init__()

        self.h_dim = h_dim
        self.v_dim = v_dim
        self.w_dim = w_dim
        self.n_classes = n_classes

        # Q(h|v)
        self.Qh_v_mean = torch.nn.Sequential(
            nn.Linear(v_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, h_dim)
        )

        # output is logstd / 2.
        self.Qh_v_var = torch.nn.Sequential(
            nn.Linear(v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, h_dim)
        )

        # Q(w|v)
        self.Qw_v_mean = torch.nn.Sequential(
            nn.Linear(v_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, w_dim)
        )

        # output is logstd / 2.
        self.Qw_v_var = torch.nn.Sequential(
            nn.Linear(v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, w_dim)
        )

        # P(c|w, h)
        self.Qc_wh = torch.nn.Sequential(
            nn.Linear(w_dim + h_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            nn.Softmax(dim = 1)
        )
    
    def infer_h(self, v, n_particle = 1):
        v = v.view(v.shape[0], -1)
        h_mean = self.Qh_v_mean(v)
        h_var = self.Qh_v_var(v)
        h_sample = self.sample(h_mean, h_var, n_particle)
        return h_mean, h_var, h_sample

    def infer_w(self, v, n_particle = 1):
        v = v.view(v.shape[0], -1)
        w_mean = self.Qw_v_mean(v)
        w_var = self.Qw_v_var(v)
        w_sample = self.sample(w_mean, w_var, n_particle)
        return w_mean, w_var, w_sample
    
    def infer_c(self, w, h):
        cat = torch.cat([w, h], axis = -1)
        c = self.Qc_wh(cat)
        return c
    
    def sample(self, mean, logstd, n_particle = 1):
        assert n_particle == 1
        sample = mean + torch.randn_like(mean) * (logstd * 2).exp()
        return sample
    
    def forward(self, X):
        h, *_ = self.infer_h(X)
        w, *_ = self.infer_w(X)
        return self.infer_c(w, h)