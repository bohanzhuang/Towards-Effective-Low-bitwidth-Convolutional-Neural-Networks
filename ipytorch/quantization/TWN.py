import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F

__all__ = ["twnConv2d", "twnLinear"]


class twnConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, cRate=0.7):
        super(twnConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias)
        self.weight_ternary = Parameter(torch.zeros(self.weight.data.size()))
        self.weight_alpha = Parameter(torch.ones(1))
        self.weight_delta = 0

        self.cRate = cRate

    def compute_grad(self):
        self.weight.grad = self.weight_ternary.grad
        # print self.weight_ternary
        # print self.weight_ternary.grad.data
        # print "alpha:", self.weight_alpha, "delta: ", self.weight_delta
        # assert False

    def forward(self, input):

        self.weight_delta = self.cRate * \
            self.weight.abs().mean().clamp(min=0, max=10).data[0]

        self.weight_ternary.data.copy_((self.weight.gt(self.weight_delta).float() -
                                        self.weight.lt(-self.weight_delta).float()).data)

        self.weight_alpha.data.copy_(((self.weight.abs() * self.weight_ternary.abs()).sum(
        ) / self.weight_ternary.abs().sum()).clamp(min=0, max=10).data)

        return F.conv2d(input * self.weight_alpha.data[0], self.weight_ternary, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class twnLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True, cRate=0.7):
        super(twnLinear, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias)

        self.weight_ternary = Parameter(torch.zeros(self.weight.data.size()))
        self.weight_alpha = Parameter(torch.ones(1))
        self.weight_delta = 0

        self.cRate = cRate

    def compute_grad(self):
        self.weight.grad = self.weight_ternary.grad
        # print self.weight_ternary.grad.data
        # print "alpha:", self.weight_alpha, "delta: ", self.weight_delta

    def forward(self, input):

        self.weight_delta = self.cRate * \
            self.weight.abs().mean().clamp(min=0, max=10).data[0]

        self.weight_ternary.data.copy_((self.weight.gt(self.weight_delta).float() -
                                        self.weight.lt(-self.weight_delta).float()).data)

        self.weight_alpha.data.copy_(((self.weight.abs() * self.weight_ternary.abs()).sum(
        ) / self.weight_ternary.abs().sum()).clamp(min=0, max=10).data)

        return F.linear(input * self.weight_alpha.data[0], self.weight_ternary, self.bias)
