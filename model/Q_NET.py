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
    def __init__(self, in_channel, image_size, h_dim, w_dim, n_classes, n_particles,  M):
        super(InferenceNet, self).__init__()
        self.n_particles = n_particles
        self.M = M
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
        # Q(w|h)
        self.Qw_h_mean = nn.Linear(h_dim, w_dim)
        self.Qw_h_logvar = nn.Linear(h_dim, w_dim)

        self.Qc = nn.Sequential(
            nn.Linear(h_dim + w_dim, n_classes),
            nn.Softmax(dim = -1)
        )
    
    
    def sample(self, mean, logstd, n_sample):
        # [bs, sample_dim]
        sample = mean + torch.randn_like(mean.expand(n_sample, -1, -1)) * logstd.exp() # [n_particles or M, bs, samples_dim]
        sample = sample.view(-1, mean.shape[-1]) # [-1, sample_dim]
        return sample
    
    def forward(self, inputs):
        inputs = inputs.view(-1, self.in_channel, self.image_size, self.image_size)
        hidden_feature = self.hidden_layer(inputs)
        h_v_mean = self.Qh_v_mean(hidden_feature)
        h_v_logvar = self.Qh_v_logvar(hidden_feature)
        h_sample = self.sample(h_v_mean, h_v_logvar, self.n_particles)
        w_h_mean = self.Qw_h_mean(h_sample)
        w_h_logvar = self.Qw_h_logvar(h_sample)
        w_sample = self.sample(w_h_mean, w_h_logvar, self.M) #  [M * n_particles * bs, h_sample]
        dup_h = h_sample.unsqueeze(0).expand(self.M, -1, -1).contiguous().view(-1, h_sample.shape[-1]) # [M * n_particles * bs, h_sample]
        concat = torch.cat([w_sample, dup_h], axis = -1)
        c_v = self.Qc(concat)
        return h_v_mean, h_v_logvar, h_sample, w_h_mean, w_h_logvar, w_sample, c_v
    
    def predict(self, inputs):
        inputs = inputs.view(-1, self.in_channel, self.image_size, self.image_size)
        hidden_feature = self.hidden_layer(inputs)
        h = self.Qh_v_mean(hidden_feature)
        w = self.Qw_h_mean(h)
        concat = torch.cat([w,h], axis = -1)
        return self.Qc(concat)
    
    # def forward(self, X):
    #     h, *_ = self.infer_h(X)
    #     w, *_ = self.infer_w(X)
    #     return self.infer_c(w, h)
