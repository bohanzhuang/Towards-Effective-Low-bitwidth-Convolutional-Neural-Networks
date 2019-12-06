import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Function

class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

        self.weight_quantization = Parameter(torch.zeros(self.weight.data.size()))
    
    def compute_grad(self):
        self.weight.grad = torch.tensor(self.weight_quantization.grad.data)

    def forward(self, input):
        return F.conv2d(input, self.weight_quantization, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True):
        super(QLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)

        self.weight_quantization = Parameter(torch.zeros(self.weight.data.size()))
    
    def compute_grad(self):
        self.weight.grad = torch.tensor(self.weight_quantization.grad.data)

    def forward(self, input):
        return F.linear(input, self.weight_quantization, self.bias)