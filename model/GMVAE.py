import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from model.Q_NET import InferenceNet
from model.P_NET import GenerationNet, PriorNet

class GMVAE(nn.Module):
    def __init__(self, args):
        super(GMVAE, self).__init__()
        self.args = args
        self.Q = InferenceNet(args.channels, args.image_size, args.h_dim, args.w_dim, args.n_classes, args.n_particles, args.M)
        self.P = GenerationNet(args.channels, args.image_size, args.h_dim, args.w_dim, args.n_classes)
        self.Prior = PriorNet(args.n_classes, args.w_dim, args.h_dim)

    def ELBO(self, X):
        h_mean, h_logstd, h_sample, w_mean, w_logstd, w_sample, c_probs = self.Q(X)
        # h_sample, w_mean, w_logstd: [n_particles * bs, h_dim]   w_sample: [M * n_particles * bs, w_dim], c_probs [M * n_particles * bs, n_classes]
        # h_mean, h_logstd: [bs, h_dim]
        recon_loss = self.recon_loss(h_sample, X)
        kl_loss_c = self.kl_c_loss(c_probs)
        kl_loss_w = self.kl_w_loss(w_mean, w_logstd)
        kl_loss_h = self.kl_h_loss(h_mean, h_logstd, w_sample, c_probs)
        # print('recon loss:{}, loss_c:{}, loss_w:{}, loss_h:{}'.format(recon_loss.mean().item(), kl_loss_c.mean().item(), \
        #     kl_loss_w.mean().item(), kl_loss_c.mean().item()))
        loss = recon_loss + kl_loss_c + kl_loss_h + kl_loss_w
        return loss, recon_loss, kl_loss_w, kl_loss_c, kl_loss_h


    def recon_loss(self, h_sample, X, type = 'bernoulli'):
        # negative E_{q(h|v)}[log p(v|h)]
        if type is 'gaussian':
            x_mean, x_logstd = self.P.gen_v(h_sample)
            recon = (X - x_mean) ** 2 / (2. * (2 * x_logstd).exp()) + np.log(2. * np.pi) / 2. + x_logstd
            return recon
        elif type is 'bernoulli':
            recon_x = self.P(h_sample)
            dup_X = X.unsqueeze(0).expand(self.args.n_particles, *X.shape).view(-1, self.args.h_dim).view(-1, *X.shape[1:])
            loss = nn.BCELoss(reduction='sum')(input = recon_x, target = X)
            return loss / recon_x.shape[0]
    
    def kl_w_loss(self, w_mean, w_logstd):
        # KL(q(w|h)||p(w))
        kl = -w_logstd + ((w_logstd * 2).exp() + torch.pow(w_mean, 2) - 1.) / 2.
        kl = kl.sum(dim=-1)
        return torch.mean(kl)
    
    def kl_c_loss(self, c_probs):
        # logits [bs, num_classes]
        kl = c_probs * (torch.log(c_probs + 1e-10) + np.log(self.args.n_classes, dtype = 'float32'))
        kl = torch.sum(kl, axis = -1)
        return torch.mean(kl)

    def kl_h_loss(self, q_h_v_mean, q_h_v_logstd, w_sample, c_probs):
        # c_logits: [M * n_particles * bs, num_classes]
        # w_sample: [M * n_particles * bs, M]
        # q_h_v_mean, q_h_v_logstd: [bs, h_dim]
        dup_h_mean = q_h_v_mean.unsqueeze(0).expand(self.args.M, *q_h_v_mean.shape).contiguous().view(-1, self.args.h_dim)
        dup_h_logstd = q_h_v_logstd.unsqueeze(0).expand(self.args.M, *q_h_v_logstd.shape).contiguous().view(-1, self.args.h_dim)
        def kl_loss(q_h_v_mean, q_h_v_logstd, p_h_wc_mean, p_h_wc_logstd):
            kl = (q_h_v_logstd * 2 - p_h_wc_logstd * 2).exp() - 1.0 - q_h_v_logstd * 2 + p_h_wc_logstd * 2
            kl += torch.pow((q_h_v_mean - p_h_wc_mean), 2) / (p_h_wc_logstd * 2).exp()
            kl = kl * 0.5
            return torch.sum(kl, axis = -1, keepdim = True) 
        kl_losses = list()
        for c in range(self.args.n_classes):
            h_wc_mean, h_wc_logstd = self.Prior(w_sample, c)
            loss = kl_loss(dup_h_mean, dup_h_logstd, h_wc_mean, h_wc_logstd)
            kl_losses.append(loss)
        kl_losses = torch.cat(kl_losses, axis = 1) # [M * n_particles * bs, num_classes]
        kl = torch.sum(kl_losses * c_probs, axis = -1)
        kl = torch.mean(kl) * self.args.n_particles
        return kl

    def forward(self, X):
        return self.ELBO(X)
        