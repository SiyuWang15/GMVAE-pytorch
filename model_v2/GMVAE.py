import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from model.Q_NET import InferenceNet
from model.P_NET import GenerationNet

class GMVAE(nn.Module):
    def __init__(self, args):
        super(GMVAE, self).__init__()
        # self.v_dim = v_dim
        self.args = args
        self.Q = InferenceNet(args.channels, args.image_size, args.h_dim, args.n_classes, args.M)
        self.P = GenerationNet(args.channels, args.image_size, args.h_dim, args.n_classes)
        

    def ELBO(self, X):
        h_mean, h_logstd, h_sample, c_probs = self.Q(X)
        # h_sample [M * bs, h_dim]
        # c_probs [M * bs, n_classes]
        recon_x = self.P(h_sample)  # [M * bs, n_classes, channel, image_size, image_size]
        # recon_x = recon_x.view(self.args.n_classes, self.args.M, -1, self.args.channels, self.args.image_size, self.args.image_size)
        recon_loss = self.recon_loss(recon_x, X, c_probs)
        kl_loss_c = self.kl_c_loss(c_probs)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd)
        # print('recon loss:{}, loss_c:{}, loss_w:{}, loss_h:{}'.format(recon_loss.mean().item(), kl_loss_c.mean().item(), \
        #     kl_loss_w.mean().item(), kl_loss_c.mean().item()))
        loss = recon_loss +  kl_loss_c + kl_loss_h
        loss = torch.mean(loss)
        return loss, recon_loss, kl_loss_c, kl_loss_h
    
    def recon_loss(self, recon_x, X, c_probs):
        dup_X = X.unsqueeze(1).expand(-1,self.args.n_classes, -1,-1,-1) # [bs, nclass, c, w, h]
        dup_X = dup_X.unsqueeze(0).expand(self.args.M, *dup_X.shape).reshape(-1, self.args.n_classes, *X.shape[1:])
        bceloss = nn.BCELoss(reduction = 'none')(input = recon_x,  target=dup_X) 
        bceloss = torch.sum(bceloss, axis = [-1,-2,-3]) # [M*bs, n_clases]
        # c_probs.reshape(self.args.M * self.args.batch_size, self.args.n_classes)
        bceloss = torch.sum(bceloss * c_probs, axis = -1)
        return torch.mean(bceloss)

    
    def kl_h_loss(self, h_mean, h_logstd):
        # KL(q(h|v)||p(h))
        kl = -h_logstd + ((h_logstd * 2).exp() + torch.pow(h_mean, 2) - 1.) / 2.
        kl = kl.sum(dim=-1)
        return torch.mean(kl)
    
    def kl_c_loss(self, c_probs):
        # c_probs [M * bs, num_classes]
        kl = c_probs * (torch.log(c_probs + 1e-10) + np.log(self.args.n_classes, dtype = 'float32'))
        kl = torch.sum(kl, axis = -1)
        return torch.mean(kl)

    def forward(self, X):
        return self.ELBO(X)
        