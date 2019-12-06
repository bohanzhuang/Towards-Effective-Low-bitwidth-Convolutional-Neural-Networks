import torch
import math
import torch.nn as nn


# refer to paper: incremental network quantization
class Quantization(object):
    """
    quantize weights, use method from Aojun Zhou, et al. Incremental Network Quantization:
    Towards Loss CNN with Low-Precision Weights
    """
    # [0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0]

    def __init__(self, portion=[0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0], bit_width=2):
        """
        init class
        :param portion: weight partition portion for the quantized group and retrained group
        :param bit_width: bit width to quantize weights, 1 bit for 0, remaining for 2^n
        """
        self.mask = None
        self.previous_weight = None

        self.n_1 = 0
        self.n_2 = 0

        self.portion = portion
        self.bitwidth = bit_width

        self.p_count = 0
        self.q_count = 0

        self.zero_bound = 0

    def data_recover(self, weight):
        """
        recover quantized data, as parameters of layers would be changed with the setting of momentum
        :param weight: <Variable>
        :return weight.data: <Tensor>
        """
        if self.mask is not None:
            # print self.mask
            weight.data.copy_(weight.data * (1 - self.mask) + self.previous_weight * self.mask)
        else:
            self.previous_weight = torch.Tensor(weight.data.size()).cuda()
        
        self.previous_weight.copy_(weight.data)
        return weight.data

    def quantize(self, weight):
        """
        main algorithm to quantize weights
        :param weight: input <variable> weight
        :return: <tensor> weight
        """

        weight_abs = weight.data.abs()
        weight_size = weight_abs.view(-1).size(0)
        # print torch.cuda.current_device()
        if self.q_count == weight_size:
            return weight.data

        if self.mask is None:
            # print torch.cuda.current_device()

            # n_1 = floor(log_2(4/3 max||W||))
            # self.n_1 = int(math.floor(math.log(weight_abs.max() * 4.0 / 3, 2)))
            # n_1 = floor(4/3 log_2 max||W||)
            self.n_1 = int(math.floor(math.log(weight_abs.max()+0.00001, 2) * 4.0 / 3))
            # n_2 = n_1 +1 - 2^(b-2)
            self.n_2 = int(self.n_1 + 1 - math.pow(2.0, self.bitwidth - 2))

            self.mask = torch.zeros(weight.data.size()).cuda()

        quantized_num = int(math.floor(
            weight_size * self.portion[self.p_count]) - self.q_count)

        if quantized_num == 0:
            self.p_count += 1
            return weight.data

        assert quantized_num >= 0, "invalid quantized num: %d, q_count: %d, weight_size: %d" % (
            quantized_num, self.q_count, weight_size)

        # print "quantized num:", quantized_num
        # Note: define quantization strategy here
        # large value quantized first

        weight_bound = (weight_abs * (1 - self.mask) -
                        self.mask).view(-1).topk(quantized_num)[0][-1]

        quantized_mask = (weight_abs * (1 - self.mask)
                          ).ge(weight_bound).float()

        # print "quantized mask sum:", quantized_mask.sum()

        real_quantized_mask = torch.zeros(quantized_mask.size()).cuda()

        # print "n_1, n_2: ", self.n_1, self.n_2
        for i in range(self.n_2, self.n_1 + 1):
            if i == self.n_2:
                alpha = 0
            else:
                alpha = 2.0**(i - 1)
            beta = 2**i

            # print "alpha, beta: ", alpha, beta
            bound_mask = (weight_abs.ge((alpha + beta) / 2.0) *
                          weight_abs.lt(beta * 1.5)).float() * quantized_mask
            if i == self.n_1:
                bound_mask += weight_abs.ge(beta *
                                            1.5).float() * quantized_mask

            if bound_mask.sum() == 0:
                continue
            quantized_weight = beta * weight.data.sign() * bound_mask
            real_quantized_mask += bound_mask

            weight.data.copy_(weight.data * (1 - bound_mask) +
                              quantized_weight)

        remain_mask = quantized_mask - real_quantized_mask
        weight.data.copy_(weight.data * (1 - remain_mask))

        self.mask += quantized_mask
        self.q_count += quantized_mask.sum()
        self.p_count += 1

        if self.previous_weight is None:
            self.previous_weight = torch.zeros(weight.data.size()).cuda()
        self.previous_weight.copy_(weight.data)
        # print "+++quantize successed"
        # return self.previous_weight
        return weight.data


class qConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(qConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)

        self.weight_quantizer = Quantization()
        self.bias_quantizer = Quantization()

    def quantize(self):
        # self.recover()
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
        # self.recover()
        self.weight.data.copy_(self.weight_quantizer.quantize(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_quantizer.quantize(self.bias))

    def recover(self):
        # recover quantized weight and bias
        self.weight.data.copy_(self.weight_quantizer.data_recover(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_quantizer.data_recover(self.bias))


def modelrecover(layers):
    if isinstance(layers, qConv2d) or isinstance(layers, qLinear):
        layers.recover()

def quantizecheck(layers):
    if isinstance(layers, qConv2d) or isinstance(layers, qLinear):
        weight_abs = layers.weight.data.abs()
        n_1, n_2 = layers.weight_quantizer.n_1, layers.weight_quantizer.n_2
        mask = torch.zeros(weight_abs.size()).cuda()
        for i in range(n_2, n_1 + 1):
            mask += weight_abs.eq(2**i).float()
        mask += weight_abs.eq(0).float()

        if mask.sum() != weight_abs.view(-1).size(0):
            print(weight_abs)
            print(mask)
            print(("quantize failed: mask_sum:%d, weight_size: %d" % (mask.sum(), weight_abs.view(-1).size(0))))