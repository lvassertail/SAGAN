import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import sn_conv2d


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()

        self.k = 8

        self.f = sn_conv2d(in_dim, in_dim // self.k, ksize=1)  # f
        self.g = sn_conv2d(in_dim, in_dim // self.k, ksize=1)  # g
        #self.h = sn_conv2d(in_dim, in_dim // 2, ksize=1)  # h
        self.h = sn_conv2d(in_dim, in_dim, ksize=1)  # h
        #self.v = sn_conv2d(in_dim // 2, in_dim, ksize=1)  # v
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        batch_size, C, width ,height = x.size()
        N = width * height

        downsampled_num = N #// 4

        # f path
        f_x = self.f(x).view(batch_size, C // self.k, N) # B * C//k * N

        # g path
        g_x = self.g(x)
        #g_x = F.max_pool2d(g_x, kernel_size=2, stride=2)
        g_x = g_x.view(batch_size, C // self.k, downsampled_num)  # B * downsampled_num * C//k

        attention_map = F.softmax(torch.bmm(f_x.permute(0, 2, 1), g_x), dim=-1)  # B * N * downsampled_num

        # h path
        h_x = self.h(x)
        #h_x = F.max_pool2d(h_x, kernel_size=2, stride=2)
        #h_x = h_x.view(batch_size, C // 2, downsampled_num) # B * C//2 * downsampled_num
        h_x = h_x.view(batch_size, C, downsampled_num)  # B * C * downsampled_num

        attention_out = torch.bmm(h_x, attention_map.permute(0,2,1))  # B * C * N
        #attention_out = attention_out.view(batch_size, C // 2, width, height)
        attention_out = attention_out.view(batch_size, C, width, height)

        #attention_out = self.v(attention_out)

        return x + self.gamma * attention_out
