import torch
import torch.nn as nn


# TODO: run experiments and check this code
# paper: ternary weight networks
class Quantization(object):
    """example:
    quantize()--->forward()--->backward()--->recover()
    """
    def __init__(self, cRate=0.7):
        self.mask = None
        self.alpha = 1
        self.delta = 0
        self.cRate = cRate
        self.previous_weight = None

    def data_recover(self, weight):
        """
        recover quantized data, as parameters of layers would be changed with the setting of momentum
        :param weight: <Variable>
        :return weight.data: <Tensor>
        """
        if self.mask is not None:
            weight.data.copy_(self.mask)
        return weight.data

    def quantize(self, weight):
        """
        main algorithm to quantize weights
        :param weight: input <variable> weight
        :return: <tensor> weight
        """
        # compute float-type weight
        if self.previous_weight is None:
            self.previous_weight = torch.Tensor(weight.size()).cuda()
        else:
            weight.data.copy_(weight.data - self.mask + self.previous_weight)
        self.previous_weight.copy_(weight.data)

        weight_abs = weight.data.abs()
        mean_ = weight_abs.mean()
        self.delta = self.cRate * mean_
        self.delta = min(self.delta, 100)
        self.delta = max(self.delta, -100)

        self.mask = weight.data.gt(self.delta).float(
        ) - weight.data.lt(-self.delta).float()

        self.alpha = (weight_abs * self.mask.abs()).sum(
        ) / self.mask.abs().sum()

        self.alpha = min(self.alpha, 100)
        self.alpha = max(self.alpha, -100)

        weight.data.copy_(self.mask)

        return weight.data


class qConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(qConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)
        self.weight_quantizer = Quantization()
        self.bias_quantizer = Quantization()

    def quantize(self):
        self.weight.data.copy_(self.weight_quantizer.quantize(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_quantizer.quantize(self.bias))

    def recover(self):
        self.weight.data.copy_(self.weight_quantizer.data_recover(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_quantizer.data_recover(self.bias))


class qLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True):
        super(qLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)
        self.weight_quantizer = Quantization()
        self.bias_quantizer = Quantization()

    def quantize(self):
        self.weight.data.copy_(self.weight_quantizer.quantize(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_quantizer.quantize(self.bias))

    def recover(self):
        self.weight.data.copy_(self.weight_quantizer.data_recover(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_quantizer.data_recover(self.bias))
