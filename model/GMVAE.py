import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from model.Q_NET import InferenceNet
from model.P_NET import GenerationNet

class GMVAE(nn.Module):
    def __init__(self, v_dim, h_dim, w_dim, n_classes):
        super(GMVAE, self).__init__()
        self.v_dim = v_dim
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.n_classes = n_classes
        self.Q = InferenceNet(v_dim, h_dim, w_dim, n_classes)
        self.P = GenerationNet(v_dim, h_dim, w_dim, n_classes)

    def ELBO(self, X):
        h_mean, h_logstd, h_sample = self.Q.infer_h(X)
        w_mean, w_logstd, w_sample = self.Q.infer_w(X)  # logstd = log(sigma) / 2.0
        logits_c = self.Q.infer_c(w_sample, h_sample)
        # lam = 0.01

        recon_loss = self.recon_loss(h_sample, X)
        kl_loss_c = self.kl_c_loss(logits_c)
        kl_loss_w = self.kl_w_loss(w_mean, w_logstd)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd, w_sample, logits_c)
        # print('recon loss:{}, loss_c:{}, loss_w:{}, loss_h:{}'.format(recon_loss.mean().item(), kl_loss_c.mean().item(), \
            # kl_loss_w.mean().item(), kl_loss_c.mean().item()))
        loss = recon_loss + kl_loss_c + kl_loss_h + kl_loss_w
        loss = torch.mean(loss)
        return loss


    def recon_loss(self, h_sample, X, type = 'bernoulli'):
        if type is 'gaussian':
            x_mean, x_logstd = self.P.gen_v(h_sample)
            recon = (X - x_mean) ** 2 / (2. * (2 * x_logstd).exp()) + np.log(2. * np.pi) / 2. + x_logstd
            return recon
        elif type is 'bernoulli':
            x_logits, _ = self.P.gen_v(h_sample)
            recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X.reshape(X.shape[0], -1), reduction='none')
            return torch.sum(recon, axis = -1)
    
    def kl_w_loss(self, w_mean, w_logstd):
        kl = -w_logstd + ((w_logstd * 2).exp() + torch.pow(w_mean, 2) - 1.) / 2.
        kl = kl.sum(dim=-1)
        return kl
    
    def kl_c_loss(self, c_logits):
        # logits [bs, num_classes]
        kl = c_logits * (torch.log(c_logits + 1e-10) + np.log(self.n_classes, dtype = 'float32'))
        kl = torch.mean(kl, axis = -1)
        return kl 

    def kl_h_loss(self, q_h_v_mean, q_h_v_logstd, w_sample, c_logits):
        # c_logits: [bs, num_classes]
        # w_sample: [bs, M]
        # q_h_v_mean, q_h_v_logstd: [bs, h_dim] 
        def kl_loss(q_h_v_mean, q_h_v_logstd, p_h_wc_mean, p_h_wc_logstd):
            kl = (q_h_v_logstd * 2 - p_h_wc_logstd * 2).exp() - 1.0 - q_h_v_logstd * 2 + p_h_wc_logstd * 2
            kl += torch.pow((q_h_v_mean - p_h_wc_mean), 2) / (p_h_wc_logstd * 2).exp()
            kl = kl * 0.5
            return torch.sum(kl, axis = -1, keepdim = True)  #[bs, 1]
        kl_losses = list()
        for c in range(self.n_classes):
            h_wc_mean, h_wc_logstd = self.P.gen_h(w_sample, c)
            loss = kl_loss(q_h_v_mean, q_h_v_logstd, h_wc_mean, h_wc_logstd)
            kl_losses.append(loss)
        kl_losses = torch.cat(kl_losses, axis = 1) # [bs, num_classes]
        kl = kl_losses * c_logits
        return torch.sum(kl, axis = -1) # [bs, 1]

    def forward(self, X):
        return self.ELBO(X)
        