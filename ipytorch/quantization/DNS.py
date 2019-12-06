import torch
import random
import torch.nn as nn


# TODO: run experiments and check this code
# paper: dynamic network surgery
class Surgery(object):
    def __init__(self, cRate=3.5):
        self.mask = None
        self.a_k = 0
        self.b_k = 0
        self.cRate = cRate
        self.weight_cache = None

    def surgery(self, weight):
        # recover weight
        if self.weight_cache is None:
            self.weight_cache = torch.Tensor(weight.size()).cuda()
        else:
            weight.data.copy_(
                weight.data + self.weight_cache * (1 - self.mask))
        self.weight_cache.copy_(weight.data)

        weight_abs = weight.data.abs()
        # compute std and mean
        if self.mask is None:
            self.mask = torch.ones(weight.size()).cuda()
            mask_count = weight_abs.ne(0).sum()
            mean_ = weight_abs.sum() / mask_count
            std_ = (weight_abs * weight_abs).sum() / mask_count
            std_ = math.sqrt(std_ - mean_ * mean_)

            # compute up bound and low bound
            self.a_k = 0.9 * max(mean_ + self.cRate * std_, 0)
            self.b_k = 1.1 * max(mean_ + self.cRate * std_, 0)
        # compute mask and update mask
        # mask = 0 if a_k > |W|; mask = 1 if b_k < |W|; otherwise keep the state of mask
        self.mask = self.mask - (
            self.mask.eq(1) * weight_abs.le(self.a_k)).float()
        self.mask = self.mask + (
            self.mask.eq(0) * weight_abs.gt(self.b_k)).float()

        weight.data.copy_(weight.data * self.mask)
        return weight.data

    def data_recover(self, weight):
        if self.mask is not None:
            weight.data.copy_(weight.data * self.mask)
        return weight.data


class qConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(qConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)
        self.weight_doctor = Surgery()
        self.bias_doctor = Surgery()
        self.weight_compress = 0
        self.bias_compress = 0
        self.execute_flag = True

    def surgery(self):
        self.weight.data.copy_(self.weight_doctor.surgery(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_doctor.surgery(self.bias))

    def recover(self):
        self.weight.data.copy_(self.weight_doctor.data_recover(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_doctor.data_recover(self.bias))


class qLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True):
        super(qLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)
        self.weight_doctor = Surgery(5.5)
        self.bias_doctor = Surgery(5.5)
        self.weight_compress = 0
        self.bias_compress = 0
        self.execute_flag = True

    def surgery(self):
        self.weight.data.copy_(self.weight_doctor.surgery(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_doctor.surgery(self.bias))

    def recover(self):
        self.weight.data.copy_(self.weight_doctor.data_recover(self.weight))
        if self.bias is not None:
            self.bias.data.copy_(self.bias_doctor.data_recover(self.bias))


# compute surgery rate, \sigma(r) = (1+\gamma*iters)^(-power_)
# if \sigma(r) > random(r) then execute surgery
def setsurgeryflag(model, epoch, max_iter=20):
    gamma_ = 0.001
    power_ = 1
    r_ = random.random()
    execute_flag = False
    if pow(1 + gamma_ * epoch, -power_) > r_ and epoch < max_iter:
        execute_flag = True

    if isinstance(model, list):
        for i in range(len(model)):
            for m in model[i].modules():
                if isinstance(m, qLinear) or isinstance(m, qConv2d):
                    m.execute_flag = execute_flag
    else:
        for m in model.modules():
            if isinstance(m, qLinear) or isinstance(m, qConv2d):
                m.execute_flag = execute_flag
