import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F

__all__ = ["wcConv2d", "wcLinear", "thinLinear", "thinConv2d"]


# weight channels
# this code needs testing and can only used for fine-tuning
class wcConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, rate=0.):
        super(wcConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.binary_weight = Parameter(torch.ones(self.weight.data.size(1)))
        self.float_weight = Parameter(torch.ones(self.weight.data.size(1)))
        self.register_buffer('rate', torch.ones(1).fill_(rate))

    def compute_grad(self):
        self.float_weight.grad = Variable(self.binary_weight.grad.data)
        # set binary_weight_grad to zero is very very important
        self.binary_weight.grad = None

    def forward(self, input):
        if self.train:
            self.float_weight.clamp(min=0)
            self.binary_weight.data.copy_(
                self.float_weight.data.ge(self.rate[0]).float())
            
        new_weight = self.binary_weight.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(self.weight) * self.weight
        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class wcLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True, rate=0.):
        super(wcLinear, self).__init__(in_features=in_features,
                                       out_features=out_features, bias=bias)

        self.binary_weight = Parameter(
            torch.ones(self.weight.data.size(1)))
        self.float_weight = Parameter(
            torch.ones(self.weight.data.size(1)))
        self.register_buffer('rate', torch.ones(1).fill_(rate))

    def compute_grad(self):
        self.float_weight.grad = Variable(self.binary_weight.grad.data)
        # set binary_weight_grad to zero is very very important
        self.binary_weight.grad = None

    def forward(self, input):
        if self.train:
            self.float_weight.clamp(min=0)
            self.binary_weight.data.copy_(
                self.float_weight.data.ge(self.rate[0]).float())
        
        # get new weight
        new_weight = self.binary_weight.unsqueeze(
            0).expand_as(self.weight) * self.weight

        return F.linear(input, new_weight, self.bias)


class thinLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bw_size=0):
        super(thinLinear, self).__init__(in_features=in_features,
                                         out_features=out_features, bias=bias)

        self.binary_weight = Parameter(torch.ones(bw_size))

    def forward(self, input):

        # use torch.gather to generate new input
        binary_mask = torch.nonzero(self.binary_weight.data).view(1, -1)
        
        thin_input = torch.gather(input, 1, Variable(binary_mask.expand(
            input.size(0), binary_mask.size(1)))).view(input.size(0), -1)

        return F.linear(thin_input, self.weight, self.bias)
"""
class thinBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, bw_size):
        super(thinBatchNorm, self).__init__(num_features, eps, momentum, affine)
        self.binary_weight = Parameter(torch.ones(bw_size))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)

    def forward(self, input):
        binary_mask = torch.nonzero(self.binary_weight.data).view(1, -1, 1, 1)
        thin_input = torch.gather(input, 1, Variable(binary_mask.expand(
            input.size(0), binary_mask.size(1), input.size(2), input.size(3))))

        return F.batch_norm(
            thin_input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)"""

class thinConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bw_size=0):
        super(thinConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)

        self.binary_weight = Parameter(torch.ones(bw_size))

    def forward(self, input):

        binary_mask = torch.nonzero(self.binary_weight.data).view(1, -1, 1, 1)
        thin_input = torch.gather(input, 1, Variable(binary_mask.expand(
            input.size(0), binary_mask.size(1), input.size(2), input.size(3))))

        return F.conv2d(thin_input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
