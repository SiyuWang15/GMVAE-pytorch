import torch
import torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_blocks):
        super().__init__()
        assert n_blocks > 0
        self.act = nn.Tanh()
        fcs = []
        fcs.append(nn.Linear(in_dim, h_dim))
        for i in range(n_blocks - 1):
            fcs.append(nn.Linear(h_dim, h_dim))
        fcs.append(nn.Linear(h_dim, out_dim))
        self.fcs = nn.ModuleList(fcs)
    
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        for i, fc in enumerate(self.fcs[:-1]):
            out = self.act(fc(out))
        out = self.fcs[-1](out)
        return out


class convNet(nn.Module):
    def __init__(self, in_channel, image_size, hidden_size):
        super().__init__()
        nef = 16
        self.act = nn.ReLU()
        self.in_channel = in_channel
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, nef, 6, 1, 0),
            nn.BatchNorm2d(nef),
            nn.ReLU(inplace=True),
            nn.Conv2d(nef, nef *2, 6, 1, 0),
            nn.BatchNorm2d(nef * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * nef, 4 * nef, 4, 2, 1),
            nn.BatchNorm2d(nef * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * nef, hidden_size, 9),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = x.view(-1, self.in_channel, self.image_size, self.image_size)
        h = self.main(x)
        h = h.view(-1, self.hidden_size)
        return h
    
class fullconvNet(nn.Module):
    def __init__(self, h_dim, hidden_size, channels, image_size):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.hidden_size =  hidden_size
        ndf = 16
        # self.fc = nn.Linear(h_dim, ndf * 8 * 4 * 4)
        self.fc = nn.Sequential(
            nn.Linear(h_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, ndf * 4, 9),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ndf * 2, ndf, 6, 1, 0),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ndf, channels, 6, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, h):
        # x = self.fc(h)
        ndf = 16
        x = self.fc(h)
        x = x.view(-1, self.hidden_size, 1, 1)
        x = self.main(x)
        x = x.view(-1, self.channels, self.image_size, self.image_size)
        return x
