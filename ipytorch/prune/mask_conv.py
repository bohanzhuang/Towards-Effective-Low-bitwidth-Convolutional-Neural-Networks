import torch
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["MaskConv2d"]


class MaskConv2d(nn.Conv2d):
    """
    custom convolutional layers for channel pruning
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(MaskConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        self.beta = Parameter(torch.ones(self.weight.data.size(1)))
        self.register_buffer("d", torch.ones(self.weight.data.size(1)))

    def update_mask(self, beta):
        self.beta.data.copy_(beta)

    def forward(self, input):
        # self.beta.data.copy_(self.d)
        new_weight = self.weight * self.beta.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(self.weight)
        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
