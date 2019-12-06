import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


def quantization(x, k):
    n = torch.pow(2, k) - 1
    # n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def quantization_activation(x, k):
    activation = normalization_on_activations(x)
    return quantization(activation, k)


def normalization_on_weights(x):
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return x


def normalization_on_activations(x):
    return torch.clamp(x, 0, 1)


class SignMeanRoundFunction(Function):

    @staticmethod
    def forward(ctx, x):
        E = torch.mean(torch.abs(x))
        return torch.where(x == 0, torch.ones_like(x), torch.sign(x / E)) * E

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RoundFunction(Function):

    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        return grad_output, None


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, k, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.register_buffer('k', torch.FloatTensor([k]))
        self.quantized = True

    def forward(self, input):
        if self.quantized:
            normalized_weight = normalization_on_weights(self.weight)
            quantized_weight = 2 * quantization(normalized_weight, self.k) - 1

            if self.bias is not None:
                quantized_bias = self.bias
                # normalized_bias = normalization_on_weights(self.bias)
                # quantized_bias = 2 * quantization(normalized_bias, self.k) - 1
            else:
                quantized_bias = None

            return F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, k, bias=True):
        super(QLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)
        self.register_buffer('k', torch.FloatTensor([k]))
        self.quantized = True

    def forward(self, input):
        if self.quantized:
            normalized_weight = normalization_on_weights(self.weight)
            quantized_weight = 2 * quantization(normalized_weight, self.k) - 1

            if self.bias is not None:
                quantized_bias = self.bias
                # normalized_bias = normalization_on_weights(self.bias)
                # quantized_bias = 2 * quantization(normalized_bias, self.k) - 1
            else:
                quantized_bias = None

            return F.linear(input, quantized_weight, quantized_bias)
        else:
            return F.linear(input, self.weight, self.bias)


class QReLU(nn.ReLU):
    """
    custom ReLU for quantization
    """

    def __init__(self, k, inplace=False):
        super(QReLU, self).__init__(inplace=inplace)
        self.register_buffer('k', torch.FloatTensor([k]))
        self.quantized = True

    def forward(self, input):
        # out = F.relu(input, self.inplace)
        out = normalization_on_activations(input)
        if self.quantized:
            return quantization(out, self.k)
        else:
            return out


class ClipReLU(nn.ReLU):
    """
    custom clip relu
    """

    def __init__(self, inplace=False):
        super(ClipReLU, self).__init__(inplace=inplace)

    def forward(self, input):
        return normalization_on_activations(input)


class CabsReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(CabsReLU, self).__init__(inplace=inplace)

    def forward(self, input):
        return normalization_on_activations(torch.abs(input))
