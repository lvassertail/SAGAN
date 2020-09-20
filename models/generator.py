import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conditional_batch_norm import CategoricalConditionalBatchNorm
from models.conditional_batch_norm import BatchNorm2d

class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, n_classes=0):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, gain=(2**0.5))
        nn.init.zeros_(self.c1.bias)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, gain=(2**0.5))
        nn.init.zeros_(self.c2.bias)
        if n_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm(in_channels, n_classes)
            self.b2 = CategoricalConditionalBatchNorm(hidden_channels, n_classes)
        else:
            self.b1 = BatchNorm2d(in_channels)
            self.b2 = BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight)
            nn.init.zeros_(self.c_sc.bias)

    def forward(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        if self.upsample:
            h = F.upsample(h, scale_factor=2)
        h = self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.learnable_sc:
            if self.upsample:
                x = F.upsample(x, scale_factor=2)
            sc = self.c_sc(x)
        else:
            sc = x
        return h + sc

class ResNetGenerator(nn.Module):
    def __init__(self, ch=64, dim_z=128, n_classes=0, bottom_width=4, activation=F.relu):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        self.block2 = ResGenBlock(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = ResGenBlock(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = ResGenBlock(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
        self.block5 = ResGenBlock(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b6 = BatchNorm2d(ch)
        self.l6 = nn.Conv2d(ch, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.l6.weight)
        nn.init.zeros_(self.l6.bias)

    def sample(self, n_samples, with_grad=False):
        """
        Samples from the Generator.
        :param n_samples: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device

        torch.autograd.set_grad_enabled(with_grad)

        sample_dim = [n_samples, self.dim_z]

        z_fake = torch.randn(sample_dim, dtype=torch.float, device=device, requires_grad=with_grad)
        y_fake = torch.randint(0, self.n_classes, (n_samples,), device=device, dtype=torch.long)
        x_fake = self.forward(n_samples, z=z_fake, y=y_fake)

        torch.autograd.set_grad_enabled(True)

        return x_fake, y_fake

    def forward(self, batchsize=64, z=None, y=None):
        anyparam = next(self.parameters())
        if z is None:
            z = torch.randn(batchsize, self.dim_z, dtype=anyparam.dtype, device=anyparam.device)
        if y is None and self.n_classes > 0:
            y = torch.randint(0, self.n_classes, (batchsize,), device=anyparam.device, dtype=torch.long)
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = h.reshape(h.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.l6(h))
        return h