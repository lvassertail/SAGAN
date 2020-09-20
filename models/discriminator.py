import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils

class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super().__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        nn.init.zeros_(self.c1.bias)
        nn.utils.spectral_norm(self.c1)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        nn.init.zeros_(self.c2.bias)
        nn.utils.spectral_norm(self.c2)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight)
            nn.init.zeros_(self.c_sc.bias)
            nn.utils.spectral_norm(self.c_sc)

    def forward(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        if self.learnable_sc:
            sc = self.c_sc(x)
            if self.downsample:
                sc = F.avg_pool2d(sc, 2)
        else:
            sc = x
        return h + sc

class ResDisOptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=pad)
        init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        init.zeros_(self.c1.bias)
        utils.spectral_norm(self.c1)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=pad)
        init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        init.zeros_(self.c2.bias)
        utils.spectral_norm(self.c2)
        self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        init.xavier_uniform_(self.c_sc.weight)
        init.zeros_(self.c_sc.bias)
        utils.spectral_norm(self.c_sc)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)
        sc = self.c_sc(x)
        sc = F.avg_pool2d(sc, 2)
        return h + sc

class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.block1 = ResDisOptimizedBlock(3, ch)
        self.block2 = ResDisBlock(ch, ch * 2, activation=activation, downsample=True)
        self.block3 = ResDisBlock(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = ResDisBlock(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = ResDisBlock(ch * 8, ch * 16, activation=activation, downsample=True)
        self.l6 = nn.Linear(ch * 16, 1)
        init.xavier_uniform_(self.l6.weight)
        init.zeros_(self.l6.bias)
        utils.spectral_norm(self.l6)

        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes, ch * 16)
            init.xavier_uniform_(self.l_y.weight)
            utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = h.sum([2, 3])
        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output