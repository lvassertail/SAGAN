import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as utils
from models.layers import *
from models.self_attention import SelfAttention


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


def init_dis_layers_64(ch, n_classes):

    blocks = torch.nn.ModuleList()
    blocks.append(FirstDiscriminatorBlock(3, ch, True))
    blocks.append(DiscriminatorBlock(ch, ch * 2, True))
    blocks.append(DiscriminatorBlock(ch * 2, ch * 4, True))
    blocks.append(DiscriminatorBlock(ch * 4, ch * 8, True))
    blocks.append(DiscriminatorBlock(ch * 8, ch * 16, True))

    linear = sn_linear(ch * 16, 1)

    l_y = None
    if n_classes > 0:
        l_y = nn.Embedding(n_classes, ch * 16)
        init.xavier_uniform_(l_y.weight)
        utils.spectral_norm(l_y)

    return blocks, linear, l_y


def init_dis_layers_32(ch, n_classes):

    blocks = torch.nn.ModuleList()
    blocks.append(FirstDiscriminatorBlock(3, ch, True))
    blocks.append(DiscriminatorBlock(ch, ch, True))
    blocks.append(DiscriminatorBlock(ch, ch, False))
    blocks.append(DiscriminatorBlock(ch, ch, False))

    linear = sn_linear(ch, 1, l_bias=False)

    l_y = None
    if n_classes > 0:
        l_y = nn.Embedding(n_classes, ch)
        init.xavier_uniform_(l_y.weight)
        utils.spectral_norm(l_y)

    return blocks, linear, l_y


class BaseSNProjectionDiscriminator(nn.Module):
    def __init__(self, init_dis_layers, ch, n_classes):
        super().__init__()

        self.blocks, self.linear, self.l_y = init_dis_layers(ch, n_classes)

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


class SNProjectionDiscriminator(BaseSNProjectionDiscriminator):
    def __init__(self, ch=64, n_classes=10):
        super().__init__(init_dis_layers_64, ch, n_classes)


class SNProjectionDiscriminator32(BaseSNProjectionDiscriminator):
    def __init__(self, ch=128, n_classes=10):
        super().__init__(init_dis_layers_32, ch, n_classes)


class BaseSaganDiscriminator(nn.Module):

    def __init__(self, init_dis_layers, feat_k, imsize, ch, attn_ch, n_classes):
        super().__init__()

        self.blocks, self.linear, self.l_y = init_dis_layers(ch, n_classes)

        self.attn = SelfAttention(attn_ch)
        self.feat_k = feat_k
        self.imsize = imsize

    def forward(self, x, y=None):

        h = x  # imsize * imsize

        for i in range(int(np.log2(self.imsize) - np.log2(self.feat_k))):
            h = self.blocks[i](h)

        h = self.attn(h)  # feat_k * feat_k

        for i in range(int(np.log2(self.imsize) - np.log2(self.feat_k)), len(self.blocks)):
            h = self.blocks[i](h)

        h = F.relu(h)
        h = h.sum([2, 3])
        output = self.linear(h)
        if y is not None:
            w_y = self.l_y(y)
            output = output + (w_y * h).sum(dim=1, keepdim=True)
        return output


class SaganDiscriminator(BaseSaganDiscriminator):
    def __init__(self, feat_k, imsize, ch=64, n_classes=10):
        num_of_blocks_before_attn = int(np.log2(imsize) - np.log2(feat_k))
        attn_ch = ch * (2 ** (num_of_blocks_before_attn - 1))
        super().__init__(init_dis_layers_64, feat_k, imsize, ch, attn_ch, n_classes)


class SaganDiscriminator32(BaseSaganDiscriminator):
    def __init__(self, feat_k, ch=128, n_classes=10):
        super().__init__(init_dis_layers_32, feat_k, 32, ch, ch, n_classes)

