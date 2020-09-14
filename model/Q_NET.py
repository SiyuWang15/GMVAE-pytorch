import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import sys
sys.path.append('..')
from layers.base import MLP, convNet, fullconvNet


# Inference Network
class InferenceNet(nn.Module):
    # def __init__(self, v_dim, h_dim, w_dim, n_classes):
    def __init__(self, in_channel, image_size, h_dim, w_dim, n_classes):
        super(InferenceNet, self).__init__()

        self.h_dim = h_dim
        # self.v_dim = v_dim
        self.in_channel = in_channel
        self.image_size = image_size
        self.w_dim = w_dim
        self.n_classes = n_classes
        
        hidden_size = 512
        nef = 16
        self.hidden_layer = convNet(in_channel, image_size, hidden_size)
        # Q(h|v)
        self.Qh_v_mean = nn.Linear(hidden_size, h_dim)
        self.Qh_v_logvar = nn.Linear(hidden_size, h_dim)
        # Q(w|v)
        self.Qw_v_mean = nn.Linear(hidden_size, w_dim)
        self.Qw_v_logvar = nn.Linear(hidden_size, w_dim)

        self.Qc = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim = -1)
        )
    
    def sample(self, mean, logstd, n_particle = 1):
        assert n_particle == 1
        sample = mean + torch.randn_like(mean) * (logstd * 2).exp()
        return sample
    
    def forward(self, inputs):
        inputs = inputs.view(-1, self.in_channel, self.image_size, self.image_size)
        hidden_feature = self.hidden_layer(inputs)
        h_v_mean = self.Qh_v_mean(hidden_feature)
        h_v_logvar = self.Qh_v_logvar(hidden_feature)
        w_v_mean = self.Qw_v_mean(hidden_feature)
        w_v_logvar = self.Qw_v_logvar(hidden_feature)
        c_v = self.Qc(hidden_feature)
        h_sample = self.sample(h_v_mean, h_v_logvar)
        w_sample = self.sample(w_v_mean, w_v_logvar)
        return h_v_mean, h_v_logvar, h_sample, w_v_mean, w_v_logvar, w_sample, c_v
    
    def predict(self, inputs):
        inputs = inputs.view(-1, self.in_channel, self.image_size, self.image_size)
        hidden_feature = self.hidden_layer(inputs)
        return self.Qc(hidden_feature)
    
    # def forward(self, X):
    #     h, *_ = self.infer_h(X)
    #     w, *_ = self.infer_w(X)
    #     return self.infer_c(w, h)