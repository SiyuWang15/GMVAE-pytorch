import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import tensorboardX
import itertools
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model.GMVAE import GMVAE
import logging
import sys
sys.path.append('..')
from utils.draw import draw_grid



class GMVAE_runner():
    def __init__(self, args):
        self.args = args 
        if self.args.dataset == 'mnist':
            logging.info('loading mnist')
            dataset = datasets.MNIST(os.path.join(self.args.datapath, 'mnist'), download = True, train = True, \
                transform = transforms.ToTensor())
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2020)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.8)], indices[int(num_items * 0.8):]
            testset = Subset(dataset, test_indices)
            trainset = Subset(dataset, train_indices)
        
        self.train_loader = DataLoader(trainset, batch_size = self.args.batch_size, shuffle = True, num_workers = 4)
        self.test_loader = DataLoader(testset, batch_size = self.args.batch_size, shuffle = True, num_workers = 4)
        self.test_iter = itertools.cycle(self.test_loader)

    def get_optimizer(self, parameters):
        if self.args.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.5, 0.999))
        elif self.args.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.args.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.args.optimizer))
    
    def train(self):
        # model = GMVAE(v_dim = self.args.v_dim, h_dim = self.args.h_dim, w_dim = self.args.w_dim, \
            # n_classes = self.args.n_classes)
        model = GMVAE(self.args)
        model = torch.nn.DataParallel(model)
        model = model.to(self.args.device)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        optimizer = self.get_optimizer(model.parameters())

        # self.args.log = self.args.run + time_string 
        tb_path = os.path.join(self.args.log, 'tensorboard')
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        tb_logger = tensorboardX.SummaryWriter(log_dir = tb_path)
        
        step = 0
        val_losses = []

        for epoch in range(self.args.n_epochs):
            if (epoch + 1) % 10 == 0:
                self.args.lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.args.lr
                logging.info('learning rate updated: {}'.format(self.args.lr))
            for _,  (X, y) in  enumerate(self.train_loader):
                # step += 1
                model.train()
                X = X.to(self.args.device)
                X = X.view(-1, self.args.channels, self.args.image_size, self.args.image_size)
                loss, *_ = model.ELBO(X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % self.args.test_freq == 0:
                    test_X, _ = next(self.test_iter)
                    test_X = test_X.to(self.args.device)
                    model.eval()
                    test_loss, recon_loss, kl_loss_w, kl_loss_c, kl_loss_h = model.ELBO(test_X)
                    logging.info('Test {} || recon loss:{:.2f}, loss_c:{:.5f}, loss_w:{:.5f}, loss_h:{:.5f}'.format(step, recon_loss.mean().item(), \
                        kl_loss_c.mean().item(), kl_loss_w.mean().item(), kl_loss_h.mean().item()))
                    acc = self.test_accuracy(model)
                    logging.info('step: {} || loss: {:.2f}, test loss: {:.2f}, acc: {:.3f}'.format(step, loss.item(), test_loss.item(), acc))

                    val_losses.append(test_loss.item())
                    tb_logger.add_scalar('loss', loss, global_step = step)
                    tb_logger.add_scalar('test_loss', test_loss, global_step = step)
                    tb_logger.add_scalar('test_acc', acc, global_step = step)
                
                if step % self.args.draw_freq == 0:
                    self.test_cluster(model, step)
                
                step += 1
                
                # if step % self.args.save_freq == 0:
                #     ckpt_path = os.path.join(self.args.ckpt_dir, 'checkpoint{}k.pth'.format(step // 1000))
                #     torch.save(model.state_dict(), ckpt_path)
                #     logging.info('checkpoint{}k.pth saved!'.format(step // 1000))
                # step += 1
            
            ckpt_path = os.path.join(self.args.ckpt_dir, 'checkpoint{}.pth'.format(epoch))
            torch.save(model.state_dict(), ckpt_path)
            logging.info('{} saved'.format(ckpt_path))


    def test_cluster(self, model, step):
        model.eval()
        N_sample = 10
        X_list = list()
        for c in range(self.args.n_classes):
            w_sample = torch.randn(N_sample, self.args.w_dim).cuda()
            h_sample, _ = model.Prior(w_sample, c)
            X = model.P(h_sample)
            X_list.append(X)
        X_sample = torch.cat(X_list).cpu()
        draw_grid(X_sample, os.path.join(self.args.img_dir, 'grid{}.png'.format(step)))
        logging.info('grid{}.png saved!'.format(step))


    def test_accuracy(self, model):
        q_c_v = list()
        labels = np.array([])
        model.eval()
        for i, (val_x, val_y) in enumerate(self.test_loader):
            val_x = val_x.to(self.args.device)
            pred = model.Q.predict(val_x) # [bs, n_classes]
            q_c_v.append(pred.detach().cpu().numpy())
            labels = np.concatenate([labels, val_y])
        q_c_v = np.concatenate(q_c_v, axis = 0)
        # labels: [len(test_loader), 1]  q_c_v: [len(test_loader), n_classes]
        ind = np.argmax(q_c_v, axis = 0)
        # print(q_c_v[ind])
        cluster_to_label = labels[ind]
        pred_cluster = np.argmax(q_c_v, axis = 1)
        # print(cluster_to_label)
        pred_class = pred_cluster
        for i, p in enumerate(pred_cluster):
            pred_class[i] = cluster_to_label[p]
        acc = np.sum(pred_class == labels) / len(labels)
        # wrong = dict()
        # for i, l in enumerate(pred_class):
        #     if pred_class[i] != labels[i]:
        #         k = '%d-%d' %(labels[i], pred_class[i])
        #         if k in wrong.keys():
        #             wrong[k] += 1
        #         else:
        #             wrong[k] = 1
        # print(wrong)
        return acc
    
    def test(self):
        model = GMVAE(channels = self.args.channels, image_size = self.args.image_size, h_dim = self.args.h_dim, w_dim = self.args.w_dim, n_classes = self.args.n_classes)
        state_dict = torch.load(self.args.test_ckpt)
        model.load_state_dict(state_dict)
        if self.args.gpu_list is not None:
            if len(self.args.gpu_list.split(',')) > 1:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        self.test_accuracy(model)

        