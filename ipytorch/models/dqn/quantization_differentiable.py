#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-23 上午11:07
# @Author  : xiezheng
# @Site    : 
# @File    : quantization_differentiable.py

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

# standard_gradList = list(range(15))
# ours_gradList = list(range(15))

# standard_array = [0] * 15
# ours_array = [0] * 15
# standard_name = 'stand_grad'
# ours_name = 'ours_grad'
# standard_count = 0
# ours_count = 0
# standard_epoch = 1
# ours_epoch = 1
#
# count = 0
# limit_count = 1 * 20 * 15
#
# cos_grad_array = [0] * 15
# cos_grad_name = 'cos_grad'
# cos_grad_count = 0
# cos_grad_epoch = 1
#
# x_array = [0] * 15
# x_name = 'x_data'
# x_count = 0
# x_epoch = 1


# def add(grad_array, tensor_data, count_):
#     num = count_ % 15
#     if type(grad_array[num]) == int:
#         grad = tensor_data.clone().cpu().view(-1).numpy()
#         grad_array[num] = grad
#         # print(grad_array[num])
#     else:
#         # print('every layer num is:', len(grad_array[0]))
#         grad = tensor_data.clone().cpu().view(-1).numpy()
#         new_grad = np.concatenate((grad, grad_array[num]), axis=0)
#         grad_array[num] = new_grad
#     count_ += 1
#     return grad_array, count_
#
#
# def draw(name, grad_array, count_, epoch, limit):
#     # gradient histogram
#     # print(name + " |===> count =" + str(count_))
#     if count_ == limit:
#         path = './2bit_noclamp_hist_3/' + name + "_" + str(epoch)
#         if os.path.isdir(path) is False:
#             os.makedirs(path)
#
#         # if ours_epoch == 1 or ours_epoch == 10 or ours_epoch == 20 or ours_epoch == 30:
#         #     pickle_name = name + '-' + str(epoch) + '.pickle'
#         #     file = open(os.path.join(path, pickle_name), 'wb')
#         #     pickle.dump(grad_array, file)
#         #     file.close()
#
#         print("|===>>> draw Hist!!!")
#         for i in range(len(grad_array)):
#             # new_grad = np.hstack((grad, gradList[num]))
#             # tmp = grad_array[i][0]
#             # for j in range(1, len(grad_array[0])):
#             #     tmp = np.hstack((tmp, grad_array[i][j]))
#             image_name = name + "-" + str(i) + '.png'
#             plt.hist(grad_array[i], histtype='stepfilled',
#                      bins=100, range=(grad_array[i].min(), grad_array[i].max()))
#             plt.title(name + "-" + str(i))
#             plt.grid(True)
#             plt.savefig(os.path.join(path, image_name))
#             plt.clf()
#             plt.cla()
#
#         epoch += 1
#         grad_array = [0] * 15
#         count_ = 0
#
#     return grad_array, count_, epoch
#
#
# class RoundFunction(Function):
#
#     @staticmethod
#     def forward(ctx, x, n):
#         ctx.save_for_backward(x)
#         return torch.round(x * n) / n
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         global count
#         global limit_count
#
#         global standard_epoch
#         global standard_array
#         global standard_count
#         global standard_name
#
#         global ours_count
#         global ours_array
#         global ours_name
#         global ours_epoch
#
#         global cos_grad_array
#         global cos_grad_count
#         global cos_grad_epoch
#         global cos_grad_name
#
#         global x_array
#         global x_name
#         global x_count
#         global x_epoch
#
#         # print('grad_output', grad_output)
#         grad = 1.0 - torch.cos(2 * np.pi * x)
#         # grad = torch.clamp(grad, min=0., max=1)
#         new_grad = grad_output * grad
#         # print('new_grad', new_grad)
#         count += 1
#         # print('count', count)
#         if count >= (ours_epoch - 1) * 196 * 15 and ours_count <= limit_count:
#             print('count', count)
#
#             standard_array, standard_count = add(standard_array, grad_output, standard_count)
#             standard_array, standard_count, standard_epoch = \
#                 draw(standard_name, standard_array, standard_count, standard_epoch, limit_count)
#
#             cos_grad_array, cos_grad_count = add(cos_grad_array, grad, cos_grad_count)
#             cos_grad_array, cos_grad_count, cos_grad_epoch = \
#                 draw(cos_grad_name, cos_grad_array, cos_grad_count, cos_grad_epoch, limit_count)
#
#             ours_array, ours_count = add(ours_array, new_grad, ours_count)
#             ours_array, ours_count, ours_epoch = \
#                 draw(ours_name, ours_array, ours_count, ours_epoch, limit_count)
#
#             x_array, x_count = add(x_array, x, x_count)
#             x_array, x_count, x_epoch = draw(x_name, x_array, x_count, x_epoch, limit_count)
#
#         # if ours_epoch == 2 and standard_epoch == 2:
#         #     assert False
#
#         return new_grad, None

# grad_mean = [0]
# count = 0


def quantization_on_weights(x, k):
    if k == 1:
        return SignMeanRoundFunction.apply(x)

    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5   # normalization weights
    n = torch.pow(2, k) - 1

    return 2 * RoundFunction.apply(x, n) - 1
    # return 2 * WeightRoundDifferentiableFunction.apply(x, n) - 1

# def normalization_on_weights(x):
#     x = torch.tanh(x)
#     x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
#     return x


def quantization_on_activations(x, k, beta, clip_val):
    # print('2' * 20)
    # print('beta=', beta.item())
    # print('clip_val=', clip_val.item())

    n = (torch.pow(2, k) - 1) / clip_val
    x = torch.clamp(x, 0, clip_val.item())         # normalization activations
    return ActRoundDifferentiableFunction.apply(x, n, beta)
    # return PiecewiseFunction.apply(x, n)    # 2-order piecewise


def normalization_on_activations(x):
    return torch.clamp(x, 0, 1)


class SignMeanRoundFunction(Function):

    @staticmethod
    def forward(ctx, x):

        # dorefa-net: all filter-E
        # E = torch.mean(torch.abs(x))

        # xnor-net : channel-E
        avg = nn.AdaptiveAvgPool3d(1)
        E = avg(torch.abs(x))
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
        return grad_output, None


# class WeightRoundDifferentiableFunction(Function):
#
#     @staticmethod
#     def forward(ctx, x, n):
#         ctx.save_for_backward(x)
#         ctx.n = n
#         return torch.round(x * n) / n
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         beta = 16
#         grad = 1.0 - torch.cos(2 * ctx.n * np.pi * x) / beta
#         return grad_output*grad, None


class ActRoundDifferentiableFunction(Function):

    @staticmethod
    def forward(ctx, x, n, beta):
        ctx.save_for_backward(x)
        ctx.n = n
        ctx.beta = beta

        # grad = 1.0 - torch.cos(2 * ctx.n * np.pi * x) / ctx.beta
        # print('|===>>>>layer grad mean is:', torch.mean(grad))
        #
        # global grad_mean
        # global count
        #
        # count += 1
        # # grad_mean = torch.mean(grad)
        #
        # if count == 1:
        #     grad_mean = grad.clone().cpu().view(-1).numpy()
        #
        # else:
        #     # print('every layer num is:', len(grad_array[0]))
        #     grad = grad.clone().cpu().view(-1).numpy()
        #     new_grad = np.concatenate((grad, grad_mean), axis=0)
        #     grad_mean = new_grad
        #
        # if count % 18 == 0:
        #     print('|===>>>>batch grad mean is:', np.mean(grad_mean))
        #     grad_mean = [0]

        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors

        # print('3' * 20)
        # print('ctx.beta=', ctx.beta.item())

        # dynamic adjust beta-2bit
        # index = ((x == 1/6) + (x == 1/2) + (x == 5/6)).float()
        # # beta = index * 1000 + (1 - index) * beta    # 1000-8
        # beta = index + (1 - index) * ctx.beta   # 1-8
        # # print(beta)
        # grad = 1.0 - torch.cos(2 * ctx.n * np.pi * x) / beta

        # dynamic adjust beta-1bit
        # index = ((x >= 0.45) * (x <= 0.55)).float()
        # beta = index * 1.0 / 2 + (1 - index) * ctx.beta   # grad-3
        # # beta = index * 1.0 / 3 + (1 - index) * ctx.beta   # grad-4
        # grad = 1.0 - torch.cos(2 * ctx.n * np.pi * x) / beta

        # original beta-grad

        # index = ((x >= 0) * (x <= 1.0/(4 * ctx.n))).float()  # 0 ~ 1/4n -> grad=1

        #  0 and 1.0 -> grad=1

        # index = ((x == 0) + (x == 1.0)).float()
        # grad = 1.0 - torch.cos(2 * ctx.n * np.pi * x) / ctx.beta
        # adjust_grad = index + (1 - index)*grad
        # return grad_output * adjust_grad, None, None

        grad = 1.0 - torch.cos(2 * ctx.n * np.pi * x) / ctx.beta
        return grad_output*grad, None, None

# class PiecewiseFunction(Function):
#
#     @staticmethod
#     def forward(ctx, x, n):
#         ctx.save_for_backward(x)
#         ctx.n = n
#         return torch.round(x * n) / n
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#
#         grad_low = 4 * x
#         grad_up = 4 - 4 * x
#         grad_zeros = x - x
#
#         # grad_input = grad_output.clone()
#         low_index = ((0 <= x) * (x < 0.5)).float()
#         up_index = ((0.5 <= x) * (x < 1)).float()
#         zero_index = ((x < 0)+(x >= 1)).float()
#
#         # grad_input[low_index] = grad_low[low_index]
#         # grad_input[up_index] = grad_up[up_index]
#         # grad_input[zero_index] = grad_zeros[zero_index]
#         grad_input = grad_low * low_index + grad_up * up_index + grad_zeros * zero_index
#
#         grad_input = grad_input * grad_output
#         return grad_input, None


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

    def forward(self, input):
        quantized_weight = quantization_on_weights(self.weight, self.k)
        if self.bias is not None:
            # Don't quantize bias
            quantized_bias = self.bias

            # quantize bias
            # quantized_bias = quantization_on_weights(self.bias, self.k)
        else:
            quantized_bias = None

        return F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, k, bias=True):
        super(QLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)
        self.register_buffer('k', torch.FloatTensor([k]))

    def forward(self, input):
        quantized_weight = quantization_on_weights(self.weight, self.k)
        if self.bias is not None:
            # Don't quantize bias
            quantized_bias = self.bias

            # quantize bias
            # quantized_bias = quantization_on_weights(self.bias, self.k)
        else:
            quantized_bias = None

        return F.linear(input, quantized_weight, quantized_bias)


class QReLU(nn.ReLU):
    """
    custom ReLU for quantization
    """

    def __init__(self, k, inplace=False, beta=8, clip_val=1):
        super(QReLU, self).__init__(inplace=inplace)
        self.register_buffer('k', torch.FloatTensor([k]))
        self.register_buffer('beta', torch.FloatTensor([beta]))
        self.register_buffer('clip_val', torch.FloatTensor([clip_val]))

    def forward(self, input):
        # print('1'*20)
        # print('k=', self.k.item())
        # print('beta=', self.beta.item())
        # print('clip_val=', self.clip_val.item())
        out = quantization_on_activations(input, self.k, beta=self.beta, clip_val=self.clip_val)
        return out


class ClipReLU(nn.ReLU):
    """
    custom clip relu
    """

    def __init__(self, inplace=False):
        super(ClipReLU, self).__init__(inplace=inplace)

    def forward(self, input):
        return normalization_on_activations(input)


# class cabs(nn.ReLU):
#     """
#     cabs
#     """
#
#     def __init__(self, inplace=False):
#         super(cabs, self).__init__(inplace=inplace)
#
#     def forward(self, input):
#         return torch.clamp(torch.abs(input), max=1)


class CabsReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(CabsReLU, self).__init__(inplace=inplace)

    def forward(self, input):
        return normalization_on_activations(torch.abs(input))