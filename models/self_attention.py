import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import sn_conv2d

class SelfAttentionOld(nn.Module):
    """ Self attention Layer"""
    def __init__(self, input_dims, output_dims=None, return_attn=False):
        output_dims = input_dims // 8 if output_dims is None else output_dims
        if output_dims == 0:
            raise Exception(
                "The output dims corresponding to the input dims is 0. Increase the input\
                            dims to 8 or more. Else specify output_dims"
            )
        super(SelfAttentionOld, self).__init__()
        self.query = nn.Conv2d(input_dims, output_dims, 1)
        self.key = nn.Conv2d(input_dims, output_dims, 1)
        self.value = nn.Conv2d(input_dims, input_dims, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.return_attn = return_attn

    def forward(self, x):
        r"""Computes the output of the Self Attention Layer
        Args:
            x (torch.Tensor): A 4D Tensor with the channel dimension same as ``input_dims``.
        Returns:
            A tuple of the ``output`` and the ``attention`` if ``return_attn`` is set to ``True``
            else just the ``output`` tensor.
        """
        dims = (x.size(0), -1, x.size(2) * x.size(3))
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)
        out_value = torch.bmm(out_value, attn).view(x.size())
        out = self.gamma * out_value + x
        if self.return_attn:
            return out, attn
        return out


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()

        self.k = 8

        self.f = sn_conv2d(in_dim, in_dim // self.k, ksize=1)  # f
        self.g = sn_conv2d(in_dim, in_dim // self.k, ksize=1)  # g
        self.h = sn_conv2d(in_dim, in_dim // 2, ksize=1)  # h
        self.v = sn_conv2d(in_dim // 2, in_dim, ksize=1)  # v
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        batch_size, C, width ,height = x.size()
        N = width * height

        downsampled_num = N // 4

        # f path
        f_x = self.f(x).view(batch_size, C // self.k, N) # B * C//k * N

        # g path
        g_x = self.g(x)
        g_x = F.max_pool2d(g_x, kernel_size=2, stride=2)
        g_x = g_x.view(batch_size, C // self.k, downsampled_num).permute(0, 2, 1) # B * downsampled_num * C//k

        attention_map = F.softmax(torch.bmm(g_x, f_x), dim=-1) #  B * downsampled_num * N

        # h path
        h_x = self.h(x)
        h_x = F.max_pool2d(h_x, kernel_size=2, stride=2)
        h_x = h_x.view(batch_size, C // 2, downsampled_num) # B * C//2 * downsampled_num

        attention_out = torch.bmm(h_x, attention_map) # B * C//2 * N
        attention_out = attention_out.view(batch_size, C // 2, width, height)

        attention_out = self.v(attention_out)

        return x + self.gamma * attention_out
