import torch
import torch.nn as nn
import torch.nn.functional as F

nn.MaxPool2d


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======

        modules.extend([nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True)]),

        modules.extend([nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True)])

        modules.extend([nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True)])

        modules.extend([nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2, inplace=True)])

        modules.append(nn.Conv2d(512, out_channels, 4, 1, 0, bias=False))
        # ========================

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        modules.extend([nn.ConvTranspose2d(in_channels, 512, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(True)])

        modules.extend([nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(True)]),

        modules.extend([nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True)]),

        modules.extend([nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(True)])

        modules.extend([nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False)])

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))

