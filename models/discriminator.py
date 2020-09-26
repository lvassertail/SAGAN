import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils
from models.layers import *
from models.self_attention import SelfAttention
#from models.self_attention import SelfAttentionOld

class FirstDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        self.downsample = downsample
        self.c1 = sn_conv2d(in_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c2 = sn_conv2d(out_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))

        if downsample:
            self.c_sc = sn_conv2d(in_channels, out_channels, ksize=1, pad=0, init_gain=1.0)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.relu(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        sc = x
        if self.downsample:
            sc = self.c_sc(x)
            sc = F.avg_pool2d(sc, 2)
        return h + sc


class DiscriminatorBlock(FirstDiscriminatorBlock):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__(in_channels, out_channels, downsample)

    def forward(self, x):
        h = x
        h = F.relu(h)

        return super().forward(h)


class SNProjectionDiscriminator(nn.Module):
    def __init__(self, ch=64, n_classes=10):
        super().__init__()

        self.initLayers(ch, n_classes)

    def initLayers(self, ch, n_classes):

        self.blocks = torch.nn.ModuleList()

        self.blocks.append(FirstDiscriminatorBlock(3, ch, True))
        self.blocks.append(DiscriminatorBlock(ch, ch * 2, True))
        self.blocks.append(DiscriminatorBlock(ch * 2, ch * 4, True))
        self.blocks.append(DiscriminatorBlock(ch * 4, ch * 8, True))
        self.blocks.append(DiscriminatorBlock(ch * 8, ch * 16, True))

        self.linear = sn_linear(ch * 16, 1)

        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes, ch * 16)
            init.xavier_uniform_(self.l_y.weight)
            utils.spectral_norm(self.l_y)

    def forward(self, x, y=None):
        h = x
        for block in self.blocks:
            h = block(h)
        h = F.relu(h)
        h = h.sum([2, 3])
        output = self.linear(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output


class SNProjectionDiscriminator32(SNProjectionDiscriminator):

    def initLayers(self, ch, n_classes):
        self.blocks = torch.nn.ModuleList()

        self.blocks.append(FirstDiscriminatorBlock(3, ch, True))
        self.blocks.append(DiscriminatorBlock(ch, ch, True))
        self.blocks.append(DiscriminatorBlock(ch, ch, False))
        self.blocks.append(DiscriminatorBlock(ch, ch, False))

        self.linear = sn_linear(ch, 1, l_bias=False)

        if n_classes > 0:
            self.l_y = nn.Embedding(n_classes, ch)
            init.xavier_uniform_(self.l_y.weight)
            utils.spectral_norm(self.l_y)


class SaganDiscriminator(SNProjectionDiscriminator):
    def __init__(self, ch=64, n_classes=10):
        super().__init__(ch, n_classes)

        self.attn = SelfAttention(ch)

    def forward(self, x, y=None):
        h = x  # 64 * 64
        h = self.block1(h)  # 32 * 32

        h = self.attn(h)  # 32 * 32

        h = self.block2(h)  # 16 * 16
        h = self.block3(h)  # 8 * 8
        h = self.block4(h)  # 4 * 4
        h = self.block5(h)  # 2 * 2
        h = F.relu(h)
        h = h.sum([2, 3])
        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output
