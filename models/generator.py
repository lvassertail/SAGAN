import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conditional_batch_norm import ConditionalBatchNorm
from models.layers import *
from models.self_attention import SelfAttention
#from models.self_attention import SelfAttentionOld

class BaseGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()

        self.init_conv(in_channels, out_channels)

        self.b1 = ConditionalBatchNorm(in_channels, n_classes)
        self.b2 = ConditionalBatchNorm(out_channels, n_classes)

    def init_conv(self, in_channels, out_channels):
        self.c1 = conv2d(in_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c2 = conv2d(out_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c_sc = conv2d(in_channels, out_channels, ksize=1, pad=0, init_gain=1.0)

    def forward(self, x, y):
        h = x
        h = self.b1(h, y)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.c1(h)
        h = self.b2(h, y)
        h = F.relu(h)
        h = self.c2(h)

        x = F.interpolate(x, scale_factor=2)
        sc = self.c_sc(x)

        return h + sc


class SpectralNormGenBlock(BaseGenBlock):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__(in_channels, out_channels, n_classes)

    def init_conv(self, in_channels, out_channels):
        self.c1 = sn_conv2d(in_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c2 = sn_conv2d(out_channels, out_channels, ksize=3, pad=1, init_gain=(2**0.5))
        self.c_sc = sn_conv2d(in_channels, out_channels, ksize=1, pad=0, init_gain=1.0)


def init_gen_layers_64(ch, n_classes, dim_z, bottom_width):

    blocks = torch.nn.ModuleList()
    l = linear(dim_z, (bottom_width ** 2) * ch * 16)
    blocks.append(BaseGenBlock(ch * 16, ch * 8, n_classes=n_classes))
    blocks.append(BaseGenBlock(ch * 8, ch * 4, n_classes=n_classes))
    blocks.append(BaseGenBlock(ch * 4, ch * 2, n_classes=n_classes))
    blocks.append(BaseGenBlock(ch * 2, ch, n_classes=n_classes))
    last_conv = conv2d(ch, 3, ksize=3, pad=1, init_gain=1)

    return blocks, l, last_conv


def init_gen_layers_sn_64(ch, n_classes, dim_z, bottom_width):

    blocks = torch.nn.ModuleList()
    l = sn_linear(dim_z, (bottom_width ** 2) * ch * 16)
    blocks.append(SpectralNormGenBlock(ch * 16, ch * 8, n_classes=n_classes))
    blocks.append(SpectralNormGenBlock(ch * 8, ch * 4, n_classes=n_classes))
    blocks.append(SpectralNormGenBlock(ch * 4, ch * 2, n_classes=n_classes))
    blocks.append(SpectralNormGenBlock(ch * 2, ch, n_classes=n_classes))
    last_conv = sn_conv2d(ch, 3, ksize=3, pad=1, init_gain=1)

    return blocks, l, last_conv


def init_gen_layers_32(ch, n_classes, dim_z, bottom_width):

    blocks = torch.nn.ModuleList()
    l = linear(dim_z, (bottom_width ** 2) * ch, l_bias=False)
    blocks.append(BaseGenBlock(ch, ch, n_classes=n_classes))
    blocks.append(BaseGenBlock(ch, ch, n_classes=n_classes))
    blocks.append(BaseGenBlock(ch, ch, n_classes=n_classes))
    last_conv = conv2d(ch, 3, ksize=3, pad=1, init_gain=1)

    return blocks, l, last_conv


def init_gen_layers_sn_32(ch, n_classes, dim_z, bottom_width):

    blocks = torch.nn.ModuleList()
    l = sn_linear(dim_z, (bottom_width ** 2) * ch, l_bias=False)
    blocks.append(SpectralNormGenBlock(ch, ch, n_classes=n_classes))
    blocks.append(SpectralNormGenBlock(ch, ch, n_classes=n_classes))
    blocks.append(SpectralNormGenBlock(ch, ch, n_classes=n_classes))
    last_conv = sn_conv2d(ch, 3, ksize=3, pad=1, init_gain=1)

    return blocks, l, last_conv


class BaseGenerator(nn.Module):
    def __init__(self, init_gen_layers_func, ch, dim_z=128, n_classes=10):
        super().__init__()

        self.bottom_width = 4
        self.dim_z = dim_z
        self.n_classes = n_classes

        self.blocks, self.linear, self.last_conv = init_gen_layers_func(ch, n_classes, dim_z, self.bottom_width)

        self.bn = nn.BatchNorm2d(ch)

    def forward(self, z, y):
        h = z
        h = self.linear(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)

        for block in self.blocks:
            h = block(h, y)

        h = self.bn(h)
        h = F.relu(h)
        h = torch.tanh(self.last_conv(h))
        return h


class BaselineGenerator(BaseGenerator):
    def __init__(self, ch=64, dim_z=128, n_classes=10):
        super().__init__(init_gen_layers_64, ch, dim_z, n_classes)


class BaselineGenerator32(BaseGenerator):
    def __init__(self, ch=256, dim_z=128, n_classes=10):
        super().__init__(init_gen_layers_32, ch, dim_z, n_classes)


class SNGenerator(BaseGenerator):
    def __init__(self, ch=64, dim_z=128, n_classes=10):
        super().__init__(init_gen_layers_sn_64, ch, dim_z, n_classes)


class SNGenerator32(BaseGenerator):
    def __init__(self, ch=256, dim_z=128, n_classes=10):
        super().__init__(init_gen_layers_sn_32, ch, dim_z, n_classes)


class BaseGeneratorWithAttention(nn.Module):
    def __init__(self, init_gen_layers, feat_k, ch, attn_ch, dim_z, n_classes):
        super().__init__()

        self.bottom_width = 4
        self.dim_z = dim_z
        self.n_classes = n_classes

        self.blocks, self.linear, self.last_conv = init_gen_layers(ch, n_classes, dim_z, self.bottom_width)

        self.bn = nn.BatchNorm2d(ch)

        self.attn = SelfAttention(attn_ch)
        self.feat_k = feat_k

    def forward(self, z, y):
        h = z
        h = self.linear(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)  # 4 X 4

        for i in range(int(np.log2(self.feat_k) - np.log2(self.bottom_width))):
            h = self.blocks[i](h, y)

        h = self.attn(h) # feat_k X feat_k

        for i in range(int(np.log2(self.feat_k) - np.log2(self.bottom_width)), len(self.blocks)):
            h = self.blocks[i](h, y)

        h = self.bn(h)
        h = F.relu(h)
        h = torch.tanh(self.last_conv(h))
        return h


class SaganGenerator(BaseGeneratorWithAttention):
    def __init__(self, feat_k, ch=64, dim_z=128, n_classes=10):
        super().__init__(init_gen_layers_sn_64, feat_k, ch, ch * 2, dim_z, n_classes)


class SaganGenerator32(BaseGeneratorWithAttention):
    def __init__(self, feat_k, ch=256, dim_z=128, n_classes=10):
        super().__init__(init_gen_layers_sn_32, feat_k, ch, ch, dim_z, n_classes)

