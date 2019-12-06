from torch.nn import Parameter
import torch.nn as nn
import torch
from ipytorch.models.dqn.quantization import *


__all__ = ["MaskQuanConv2d"]


class MaskQuanConv2d(nn.Conv2d):
    """
    custom convolutional layers for channel pruning
    """

    def __init__(self, in_channels, out_channels, kernel_size, k,
                 stride=1, padding=0, bias=True):
        super(MaskQuanConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.beta = Parameter(torch.ones(self.weight.data.size(1)))
        self.register_buffer("d", torch.ones(self.weight.data.size(1)))
        self.register_buffer('k', torch.FloatTensor([k]))
        if k == 32:
            self.quantized = False
        else:
            self.quantized = True

    def update_mask(self, beta):
        self.beta.data.copy_(beta)

    def forward(self, input):
        # self.beta.data.copy_(self.d)
        if self.quantized:
            normalized_weight = normalization_on_weights(self.weight)
            quantized_weight = 2 * quantization(normalized_weight, self.k) - 1
            new_weight = quantized_weight * self.beta.unsqueeze(0).unsqueeze(
                2).unsqueeze(3).expand_as(quantized_weight)
        else:
            new_weight = self.weight * self.beta.unsqueeze(0).unsqueeze(
                2).unsqueeze(3).expand_as(self.weight)

        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
