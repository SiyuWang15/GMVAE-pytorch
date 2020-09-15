import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import sys 
sys.path.append('..')
from layers.base import MLP, convNet, fullconvNet

class GenerationNet(nn.Module):
    def __init__(self, channels, image_size, h_dim, w_dim, n_classes):
        super(GenerationNet, self).__init__()
        self.h_dim = h_dim
        # self.v_dim = v_dim
        self.channels = channels
        self.image_size = image_size
        self.w_dim = w_dim
        self.n_classes = n_classes

        self.v_h = fullconvNet(h_dim, 512, channels, image_size)

    def forward(self, h):
        assert h.shape[-1] == self.h_dim
        v = self.v_h(h)
        return v
        
    
class PriorNet(nn.Module):
    def __init__(self, n_classes, w_dim, h_dim):
        super(PriorNet, self).__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.n_classes = n_classes
        hidden_size = 512
        self.hidden_layer = nn.Sequential(
            nn.Linear(w_dim, hidden_size),
            nn.Tanh()
        )
        h_w_mean = list()
        h_w_logvar = list()
        for k in range(n_classes):
            h_w_mean.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, h_dim)
                )
            )
            h_w_logvar.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, h_dim)
                )
            )
        self.h_w_mean = nn.ModuleList(h_w_mean)
        self.h_w_logvar = nn.ModuleList(h_w_logvar)
    
    def infer_c(self, h_sample, w_sample):
        # h_sample, w_sample: [M, bs, h_dim or w_dim]
        def loglikelihood(h, h_mean, h_logstd):
            logp = - torch.pow(h - h_mean, 2) / ((h_logstd * 2).exp() * 2.0) - np.log(2 * np.pi) / 2.0 - h_logstd
            return torch.sum(logp, axis = -1, keepdim = True) # [M, bs, 1]
        c_prob = list()
        M, bs, _ = h_sample.shape
        feature = self.hidden_layer(w_sample) 
        for c in range(self.n_classes):
            mean = self.h_w_mean[c](feature)  # [M, bs, h_dim]
            logstd = self.h_w_logvar[c](feature)
            c_prob.append(loglikelihood(h_sample, mean, logstd))
        c_prob = torch.cat(c_prob, axis = -1) # [M, bs, n_classes]
        c_prob = nn.Softmax(dim = -1)(c_prob)
        return c_prob

    def forward(self, w, c):
        hidden = self.hidden_layer(w)
        h_w_mean = self.h_w_mean[c](hidden)
        h_w_logvar = self.h_w_logvar[c](hidden)
        return h_w_mean, h_w_logvar