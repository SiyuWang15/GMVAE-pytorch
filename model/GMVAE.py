import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from model.Q_NET import InferenceNet
from model.P_NET import GenerationNet, PriorNet

class GMVAE(nn.Module):
    def __init__(self, args):
        super(GMVAE, self).__init__()
        # self.v_dim = v_dim
        self.args = args
        self.channels = args.channels
        self.image_size = args.image_size
        self.w_dim = args.w_dim
        self.h_dim = args.h_dim
        self.n_classes = args.n_classes
        self.M = args.M
        self.Q = InferenceNet(self.channels, self.image_size, self.h_dim, self.w_dim, self.n_classes,self.M)
        self.P = GenerationNet(self.channels, self.image_size, self.h_dim, self.w_dim, self.n_classes)
        self.Prior = PriorNet(self.n_classes, self.w_dim, self.h_dim)

    def ELBO(self, X):
        h_mean, h_logstd, h_sample, w_mean, w_logstd, w_sample = self.Q(X)
        c_probs = self.Prior.infer_c(h_sample, w_sample)  #[M, bs, n_classes]
        # sample: [M, bs, h_dim or w_dim]
        recon_loss = self.recon_loss(h_sample, X)
        kl_loss_c = self.kl_c_loss(c_probs)
        kl_loss_w = self.kl_w_loss(w_mean, w_logstd)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd, w_sample, c_probs)
        # print('recon loss:{}, loss_c:{}, loss_w:{}, loss_h:{}'.format(recon_loss.mean().item(), kl_loss_c.mean().item(), \
        #     kl_loss_w.mean().item(), kl_loss_c.mean().item()))
        # loss = recon_loss + self.args.c_weight * kl_loss_c + self.args.h_weight * kl_loss_h + self.args.w_weight * kl_loss_w
        loss = recon_loss + kl_loss_h + kl_loss_w + kl_loss_c
        return loss, recon_loss, kl_loss_w, kl_loss_c, kl_loss_h


    def recon_loss(self, h_sample, X, type = 'bernoulli'):
        # negative E_{q(h|v)}[log p(v|h)]
        if type is 'gaussian':
            raise NotImplementedError('Gaussian reconstruction loss not implemented!')
        elif type is 'bernoulli':
            # h_sample = h_sample.view(-1, self.h_dim)
            losses = list()
            for i in range(self.M):
                h_ = h_sample[i,:]
                recon_x = self.P(h_)
                recon_x = recon_x.view(-1, self.channels, self.image_size, self.image_size)
                loss = nn.BCELoss(reduction='none')(input = recon_x, target = X)
                losses.append(torch.sum(loss, dim = [1,2,3]))  # [bs]
            losses = torch.stack(losses, axis = 1)
            return torch.mean(losses)

            # recon_x = self.P(h_sample) #[M * bs, c, h, w]
            # recon_x = recon_x.view(self.M, -1, self.channels, self.image_size, self.image_size)
            # loss = nn.BCELoss(reduction='none')(input = recon_x, target = X.expand_as(recon_x))
            # loss = torch.sum(loss, dim = [-1,-2,-3])   
            # return torch.mean(loss, axis = 0) #[bs, 1]
    
    def kl_w_loss(self, w_mean, w_logstd):
        # KL(q(w|v)||p(w))
        kl = -w_logstd + ((w_logstd * 2).exp() + torch.pow(w_mean, 2) - 1.) / 2.
        kl = kl.sum(dim=-1)
        return torch.mean(kl)
    
    # def kl_c_loss(self, c_probs):
    #     # logits [bs, num_classes]
    #     kl = c_probs * (torch.log(c_probs + 1e-10) + np.log(self.n_classes, dtype = 'float32'))
    #     kl = torch.sum(kl, axis = -1)
    #     return kl 

    def kl_c_loss(self, c_probs):
        # [M, bs, n_classes]
        kl = c_probs * (torch.log(c_probs + 1e-10) + np.log(self.n_classes, dtype = 'float32'))
        kl = torch.sum(kl, axis = -1).mean()
        if kl < self.args.lam:
            return torch.Tensor([self.args.lam]).cuda()
        return kl

    def kl_h_loss(self, q_h_v_mean, q_h_v_logstd, w_sample, c_probs):
        # c_probs: [n_classes, M, bs]
        # w_sample: [M, bs]
        # q_h_v_mean, q_h_v_logstd: [bs, h_dim] 

        def kl_loss(q_h_v_mean, q_h_v_logstd, p_h_wc_mean, p_h_wc_logstd):
            kl = (q_h_v_logstd * 2 - p_h_wc_logstd * 2).exp() - 1.0 - q_h_v_logstd * 2 + p_h_wc_logstd * 2
            kl += torch.pow((q_h_v_mean - p_h_wc_mean), 2) / (p_h_wc_logstd * 2).exp()
            kl = kl * 0.5
            return torch.sum(kl, axis = -1, keepdim = True)  # [M, bs, 1]
        kl_losses = list()
        for c in range(self.n_classes):
            h_wc_mean, h_wc_logstd = self.Prior(w_sample, c) # [M, bs, h_dim]
            loss = kl_loss(q_h_v_mean, q_h_v_logstd, h_wc_mean, h_wc_logstd) # [M, bs, 1]
            kl_losses.append(loss)
        kl_losses = torch.cat(kl_losses, axis = -1) # [M, bs, num_classes]
        kl = kl_losses * c_probs
        return torch.mean(torch.sum(kl, axis = -1)) # [bs, 1]

    def forward(self, X):
        return self.ELBO(X)
        