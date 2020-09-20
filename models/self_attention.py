import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.k = 8

        self.query = nn.Conv2d(in_dim, in_dim // self.k, kernel_size=1)  # f
        self.key = nn.Conv2d(in_dim, in_dim // self.k, kernel_size=1)  # g
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)  # h
        #self.value = nn.Conv2d(in_dim, in_dim // self.k, kernel_size=1)
        #self.last_conv = nn.Conv2d(in_dim // self.k, in_dim, kernel_size=1)  # v
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention_map: B X N X N (N = Width*Height)
        """

        batch_size, C, width ,height = x.size()
        N = width * height
        query_res = self.query(x).view(batch_size, -1, N)
        key_res = self.key(x).view(batch_size, -1, N).permute(0, 2, 1)
        attention_map = F.softmax(torch.bmm(key_res, query_res), dim=-1)
        value_res = self.value(x).view(batch_size, -1, N)
        attention_out = torch.bmm(value_res, attention_map).view(x.size())
        #attention_out = attention_out.view(batch_size, C // self.k, width, height)
        #attention_out = self.v(attention_out)
        out = self.gamma * attention_out + x

        return out, attention_map