import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable


# this code still have some problem:
# 1. hard to train: pretrained only
# 2. gradient of wp and wn is computed by the mean of gradient of Wt
class qConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(qConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.weight_p = Parameter(torch.ones(1))
        self.weight_n = Parameter(torch.ones(1))
        self.weight_ternary = Parameter(torch.zeros(self.weight.data.size()))
        if self.bias is not None:
            self.bias_p = Parameter(torch.ones(1))
            self.bias_n = Parameter(torch.ones(1))
            self.bias_ternary = Parameter(torch.zeros(self.bias.data.size()))
        else:
            self.register_parameter('bias_p', None)
            self.register_parameter('bias_n', None)
            self.register_parameter('bias_ternary', None)

        self.weight_p_mask = None
        self.weight_n_mask = None
        self.bias_p_mask = None
        self.bias_n_mask = None
        self.cRate = 0.05
        print("TNS layer")

    def computegrad(self):
        # print self.weight_ternary.grad

        self.weight_p.grad = Variable(
            self.weight_ternary.grad.data * self.weight_p_mask).sum() / self.weight_p_mask.sum()
        self.weight_n.grad = Variable(
            self.weight_ternary.grad.data * self.weight_n_mask).sum() / self.weight_n_mask.sum()

        self.weight.grad = Variable(self.weight_ternary.grad.data *
                                    (self.weight_p_mask * self.weight_p.data + self.weight_n_mask *
                                     self.weight_n.data + (1 - self.weight_p_mask - self.weight_n_mask)))

        if self.bias is not None:

            self.bias_p.grad = Variable(
                self.bias_ternary.grad.data * self.bias_p_mask).sum()
            self.bias_n.grad = Variable(
                self.bias_ternary.grad.data * self.bias_n_mask).sum()

            self.bias.grad = Variable(self.bias_ternary.grad.data *
                                      (self.bias_p_mask * self.bias_p.data + self.bias_n_mask *
                                       self.bias_n.data + (1 - self.bias_p_mask - self.bias_n_mask)))

    def forward(self, input):

        # normalize
        # self.weight.data = self.weight.data / self.weight.data.abs().max()

        self.weight_p.data.clamp(min=0)
        self.weight_n.data.clamp(min=0)

        weight_delta = self.cRate * self.weight.data.abs().max()

        self.weight_p_mask = self.weight.gt(weight_delta).float().data
        self.weight_n_mask = self.weight.lt(-weight_delta).float().data

        self.weight_ternary.data.copy_(
            self.weight_p.data * self.weight_p_mask - self.weight_n.data * self.weight_n_mask)

        if self.bias is not None:
            self.bias_p.data.clamp(min=0)
            self.bias_n.data.clamp(min=0)

            self.bias.data = self.bias.data / self.bias.data.abs().max()
            bias_delta = self.cRate * self.bias.data.abs().max()

            self.bias_p_mask = self.bias.gt(bias_delta).float().data
            self.bias_n_mask = self.bias.lt(-bias_delta).float().data

            self.bias_ternary.data.copy_(
                self.bias_p.data * self.bias_p_mask - self.bias_n.data * self.bias_n_mask)

        else:
            self.bias_ternary = None

        # print self.weight_p.data, self.weight_n.data
        return F.conv2d(input, self.weight_ternary, self.bias_ternary, self.stride,
                        self.padding, self.dilation, self.groups)


class qLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True):
        super(qLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)
        self.weight_p = Parameter(torch.ones(1))
        self.weight_n = Parameter(torch.ones(1))
        self.weight_ternary = Parameter(torch.zeros(self.weight.data.size()))
        if bias:
            self.bias_p = Parameter(torch.ones(1))
            self.bias_n = Parameter(torch.ones(1))
            self.bias_ternary = Parameter(torch.zeros(self.bias.data.size()))
        else:
            self.register_parameter('bias_p', None)
            self.register_parameter('bias_n', None)
            self.register_parameter('bias_ternary', None)

        self.weight_p_mask = None
        self.weight_n_mask = None
        self.bias_p_mask = None
        self.bias_n_mask = None
        self.cRate = 0.05
        print("TNS layer")

    def computegrad(self):
        # print self.weight_ternary.grad

        self.weight_p.grad = Variable(
            self.weight_ternary.grad.data * self.weight_p_mask).sum() / self.weight_p_mask.sum()
        self.weight_n.grad = Variable(
            self.weight_ternary.grad.data * self.weight_n_mask).sum() / self.weight_n_mask.sum()

        self.weight.grad = Variable(self.weight_ternary.grad.data *
                                    (self.weight_p_mask * self.weight_p.data + self.weight_n_mask *
                                     self.weight_n.data + (1 - self.weight_p_mask - self.weight_n_mask)))
        
        if self.bias is not None:

            self.bias_p.grad = Variable(
                self.bias_ternary.grad.data * self.bias_p_mask).sum()
            self.bias_n.grad = Variable(
                self.bias_ternary.grad.data * self.bias_n_mask).sum()

            self.bias.grad = Variable(self.bias_ternary.grad.data *
                                      (self.bias_p_mask * self.bias_p.data + self.bias_n_mask *
                                       self.bias_n.data + (1 - self.bias_p_mask - self.bias_n_mask)))

    def forward(self, input):

        # normalize
        # self.weight.data = self.weight.data / self.weight.data.abs().max()

        self.weight_p.data.clamp(min=0)
        self.weight_n.data.clamp(min=0)

        weight_delta = self.cRate * self.weight.data.abs().max()

        self.weight_p_mask = self.weight.gt(weight_delta).float().data
        self.weight_n_mask = self.weight.lt(-weight_delta).float().data

        self.weight_ternary.data.copy_(
            self.weight_p.data * self.weight_p_mask - self.weight_n.data * self.weight_n_mask)

        if self.bias is not None:
            self.bias.data = self.bias.data / self.bias.data.abs().max()
            bias_delta = self.cRate * self.bias.data.abs().max()

            self.bias_p_mask = self.bias.gt(bias_delta).float().data
            self.bias_n_mask = self.bias.lt(-bias_delta).float().data

            self.bias_ternary.data.copy_(
                self.bias_p.data * self.bias_p_mask - self.bias_n.data * self.bias_n_mask)
            
            self.bias_p.data.clamp(min=0)
            self.bias_n.data.clamp(min=0)

        else:
            self.bias_ternary = None

        return F.linear(input, self.weight_ternary, self.bias_ternary)


def computgrad(layers):
    if isinstance(layers, qConv2d) or isinstance(layers, qLinear):
        layers.computegrad()
