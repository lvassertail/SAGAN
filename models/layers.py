import torch
import torch.nn as nn


def conv2d(in_channels, out_channels, ksize, pad=0, init_gain=1.0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
    nn.init.xavier_uniform_(conv.weight, gain=init_gain)
    nn.init.zeros_(conv.bias)
    return conv


def sn_conv2d(in_channels, out_channels, ksize, pad=0, init_gain=1.0):
    conv = conv2d(in_channels, out_channels, ksize, pad, init_gain)
    nn.utils.spectral_norm(conv)
    return conv


def linear(in_features, out_features, l_bias=True):
    l = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(l.weight)

    if l_bias:
        nn.init.zeros_(l.bias)
    return l


def sn_linear(in_features, out_features, l_bias=True):
    l = linear(in_features, out_features, l_bias)
    nn.utils.spectral_norm(l)
    return l