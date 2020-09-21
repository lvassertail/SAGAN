import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils
from models.layers import *

class FirstDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.c1 = sn_conv2d(in_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c2 = sn_conv2d(out_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c_sc = sn_conv2d(in_channels, out_channels, ksize=1, pad=0, init_gain=1.0)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.relu(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        sc = self.c_sc(x)
        sc = F.avg_pool2d(sc, 2)
        return h + sc

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.c1 = sn_conv2d(in_channels, in_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c2 = sn_conv2d(in_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c_sc = sn_conv2d(in_channels, out_channels, ksize=1, pad=0, init_gain=1.0)

    def forward(self, x):
        h = x
        h = F.relu(h)
        h = self.c1(h)
        h = F.relu(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        sc = self.c_sc(x)
        sc = F.avg_pool2d(sc, 2)
        return h + sc

class SNProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, n_classes=10):
        super().__init__()
        self.block1 = FirstDiscriminatorBlock(3, ch)
        self.block2 = DiscriminatorBlock(ch, ch * 2)
        self.block3 = DiscriminatorBlock(ch * 2, ch * 4)
        self.block4 = DiscriminatorBlock(ch * 4, ch * 8)
        self.block5 = DiscriminatorBlock(ch * 8, ch * 16)

        self.l6 = sn_linear(ch * 16, 1)

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
        h = F.relu(h)
        h = h.sum([2, 3])
        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output