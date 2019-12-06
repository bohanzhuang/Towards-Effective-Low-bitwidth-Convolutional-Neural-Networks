import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import Parameter
from torch.autograd import Variable


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x):
    x = nn.functional.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return x


def normalization_on_activations(x):
    return torch.clamp(x, 0, 1)


class RoundFunction(Function):

    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class SignFunction(Function):

    @staticmethod
    def forward(ctx, x):
        E = torch.mean(torch.abs(x))
        return torch.sign(x / E) * E
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class QuantizationGradientFunction(Function):

    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(k)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        k,  = ctx.saved_tensors
        maxx = torch.max(torch.abs(grad_output))
        grad_input = grad_output / maxx
        n = (2 ** k - 1).item()
        grad_input = grad_input * 0.5 + 0.5 + grad_input.new_zeros(grad_input.size()).uniform_(-0.5/n, 0.5/n)
        grad_input = torch.clamp(grad_input, 0, 1)
        grad_input = quantization(grad_input, k) - 0.5
        return grad_input * maxx * 2, None


class QConv2d(nn.Conv2d):
    """
    Custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        # bit != 1
        # self.normalized_weight = normalization_on_weights(self.weight)
        # self.quantized_weight = 2 * QuantizationFunction.apply(self.normalized_weight, self.k) - 1
        # bit = 1
        self.quantized_weight = SignFunction.apply(self.weight)

        if self.bias is not None:
            # self.normalized_bias = normalization_on_weights(self.bias)
            # self.quantized_bias = 2 * QuantizationFunction.apply(self.normalized_bias, self.k) - 1
            self.quantized_bias = SignFunction.apply(self.bias)
        else:
            self.quantized_bias = None

        return F.conv2d(input, self.quantized_weight, self.quantized_bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    """
    Custom linear layers for quantization
    """

    def __init__(self, in_features, out_features, k, bias=True):
        super(QLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        # self.normalized_weight = normalization_on_weights(self.weight)
        # self.quantized_weight = 2 * QuantizationFunction.apply(self.normalized_weight, self.k) - 1
        self.quantized_weight = SignFunction.apply(self.weight)

        if self.bias is not None:
            # self.normalized_bias = normalization_on_weights(self.bias)
            # self.quantized_bias = 2 * QuantizationFunction.apply(self.normalized_bias, self.k) - 1
            self.quantized_bias = SignFunction.apply(self.bias)
        else:
            self.quantized_bias

        return F.linear(input, self.quantized_weight, self.quantized_bias)


class QReLU(nn.ReLU):
    """
    Custom ReLU for quantization
    """
    def __init__(self, k, inplace=False):
        super(QReLU, self).__init__(inplace=inplace)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        out = normalization_on_activations(input)
        out = quantization(out, self.k)
        return out
