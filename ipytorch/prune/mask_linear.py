import torch
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["MaskLinear"]


class MaskLinear(nn.Linear):
    """
    custom fully connected layers for channel pruning
    """

    def __init__(self, in_channels, out_features, feature_h=1,
                 feature_w=1, bias=True):
        super(MaskLinear, self).__init__(in_features=in_channels * feature_h * feature_w,
                                         out_features=out_features, bias=bias)
        self.in_channels = in_channels
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.beta = Parameter(torch.ones(in_channels))
        self.register_buffer("d", torch.ones(in_channels))

    def update_mask(self, beta):
        self.beta.data.copy_(beta)

    def forward(self, input):
        new_weight = self.weight * self.beta.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand(
                self.out_features,
                self.in_channels,
                self.feature_h,
                self.feature_w).contiguous().view(self.out_features, -1)

        return F.linear(input, new_weight, self.bias)
