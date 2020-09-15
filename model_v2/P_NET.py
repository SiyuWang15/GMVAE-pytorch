import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import sys 
sys.path.append('..')
from layers.base import MLP, convNet, fullconvNet

class GenerationNet(nn.Module):
    def __init__(self, channels, image_size, h_dim, n_classes):
        super(GenerationNet, self).__init__()
        self.h_dim = h_dim
        # self.v_dim = v_dim
        self.channels = channels
        self.image_size = image_size
        self.n_classes = n_classes

        v_hc_list = list()
        for k in range(n_classes):
            v_hc_list.append(fullconvNet(h_dim, 512, channels, image_size))
        self.v_hc_list = nn.ModuleList(v_hc_list)

    def forward(self, h_sample):
        # hsample: [M * bs, h_dim]
        p_v = list()
        for c in range(self.n_classes):
            p_v.append(self.v_hc_list[c](h_sample)) # [M * bs, channel, imagesize, imagesize]
        p_v = torch.stack(p_v, axis = 0) # [n_classes, M * bs, channel, imagesize, imagesize]
        return p_v
