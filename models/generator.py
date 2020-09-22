import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conditional_batch_norm import CategoricalConditionalBatchNorm
from models.conditional_batch_norm import BatchNorm2d
from models.layers import *

class BaseGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()

        self.init_conv(in_channels, out_channels)

        self.b1 = CategoricalConditionalBatchNorm(in_channels, n_classes)
        self.b2 = CategoricalConditionalBatchNorm(out_channels, n_classes)

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


class BaselineGenerator(nn.Module):
    def __init__(self, ch=64, dim_z=128, n_classes=10):
        super().__init__()

        self.bottom_width = 4
        self.dim_z = dim_z
        self.n_classes = n_classes

        self.initLayers(ch, n_classes)

        self.b6 = BatchNorm2d(ch)

    def initLayers(self, ch, n_classes):
        self.l1 = linear(self.dim_z, (self.bottom_width ** 2) * ch * 16)
        self.block2 = BaseGenBlock(ch * 16, ch * 8, n_classes=n_classes)
        self.block3 = BaseGenBlock(ch * 8, ch * 4, n_classes=n_classes)
        self.block4 = BaseGenBlock(ch * 4, ch * 2, n_classes=n_classes)
        self.block5 = BaseGenBlock(ch * 2, ch, n_classes=n_classes)
        self.l6 = conv2d(ch, 3, ksize=3, pad=1, init_gain=1)

    def forward(self, z, y):

        h = z
        h = self.l1(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = F.relu(h)
        h = torch.tanh(self.l6(h))
        return h

class SNGenerator(BaselineGenerator):
    def __init__(self, ch=64, dim_z=128, n_classes=10):
        super().__init__(ch, dim_z, n_classes)

    def initLayers(self, ch, n_classes):
        self.l1 = sn_linear(self.dim_z, (self.bottom_width ** 2) * ch * 16)
        self.block2 = SpectralNormGenBlock(ch * 16, ch * 8, n_classes=n_classes)
        self.block3 = SpectralNormGenBlock(ch * 8, ch * 4, n_classes=n_classes)
        self.block4 = SpectralNormGenBlock(ch * 4, ch * 2, n_classes=n_classes)
        self.block5 = SpectralNormGenBlock(ch * 2, ch, n_classes=n_classes)
        self.l6 = sn_conv2d(ch, 3, ksize=3, pad=1, init_gain=1)