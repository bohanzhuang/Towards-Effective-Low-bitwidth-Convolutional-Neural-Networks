import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F

__all__ = ["wcConv2d", "wcLinear", "thinLinear"]


# weight channels
# this code needs testing and can only used for fine-tuning
class wcConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, rate=0):
        super(wcConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.binary_weight = Parameter(torch.ones(self.weight.data.size(1)))
        # self.float_weight = Parameter(torch.ones(self.weight.data.size(1)))
        self.float_weight = Parameter(torch.zeros(self.weight.data.size(1)))
        self.rate = rate

    def compute_grad(self):
        # print "compute backward"
        # print "conv grad data", self.binary_weight.grad.data.unsqueeze(0)
        # print "float_weight", self.float_weight.data.unsqueeze(0)
        # print self.weight.grad.data
        # if self.bias is not None:
        #     print "bias", self.bias.grad.data.unsqueeze(0)
        self.float_weight.grad = Variable(self.binary_weight.grad.data)
        # set binary_weight_grad to zero is very very important
        self.binary_weight.grad = None
        # set gradient to None type
        # self.weight.grad = None
        # if self.bias is not None:
        #     self.bias.grad = None

    def forward(self, input):

        # normalize channel weight
        # min_w = self.float_weight.data.min()
        # delta = 0.9*(self.float_weight.data.mean()+self.rate*self.float_weight.data.std())
        # max_w = self.float_weight.data.max()
        '''min_w = self.float_weight.data.min()
        if max_w != min_w:
            self.float_weight.data.copy_((self.float_weight.data-min_w)/(max_w-min_w))'''
        # self.float_weight.data.copy_(self.float_weight.data/max_w)
        # compute mask
        # self.binary_weight.data.copy_(self.float_weight.data.ge(self.rate*max_w).float())
        # self.binary_weight.data.copy_(self.float_weight.data.ge(delta).float())
        # self.binary_weight.data.fill_(1)
        # self.binary_weight.data.copy_(
        #     self.float_weight.data.ge(self.rate).float())
        # get new weight
        # print self.binary_weight.size()
        # print self.weight.size()
        # print self.binary_weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).size()
        new_weight = self.binary_weight.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(self.weight) * self.weight
        return F.conv2d(input, new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class wcLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """

    def __init__(self, in_features, out_features, bias=True, rate=0):
        super(wcLinear, self).__init__(in_features=in_features,
                                       out_features=out_features, bias=bias)

        self.binary_weight = Parameter(
            torch.ones(self.weight.data.size(1)))
        # self.float_weight = Parameter(
        #     torch.ones(self.weight.data.size(1)))
        self.float_weight = Parameter(
            torch.zeros(self.weight.data.size(1)))
        self.rate = rate

    def compute_grad(self):
        self.float_weight.grad = Variable(self.binary_weight.grad.data)
        # set binary_weight_grad to zero is very very important
        self.binary_weight.grad = None
        # set gradient to None type
        # self.weight.grad = None
        # if self.bias is not None:
        #     self.bias.grad = None

    def forward(self, input):

        # normalize channel weight
        # max_w = self.float_weight.data.max()
        '''min_w = self.float_weight.data.min()
        if max_w != min_w:
            self.float_weight.data.copy_((self.float_weight.data-min_w)/(max_w-min_w))'''
        # self.float_weight.data.copy_(self.float_weight.data/max_w)
        # delta = 0.9*(self.float_weight.data.mean()+self.rate*self.float_weight.data.std())
        # self.binary_weight.data.copy_(self.float_weight.data.ge(delta).float())
        # self.binary_weight.data.copy_(
        #     self.float_weight.data.ge(self.rate).float())
        # self.binary_weight.data.fill_(1)
        # compute mask
        # self.binary_weight.data.copy_(self.float_weight.data.ge(self.rate*max_w).float())

        # get new weight
        new_weight = self.binary_weight.unsqueeze(0).expand_as(self.weight) * self.weight
        
        # channel selection for fc layer
        # new_weight = self.binary_weight.unsqueeze(1).expand(self.binary_weight.size(
        #     0), self.h * self.w).contiguous().view(-1).unsqueeze(0).expand_as(self.weight) * self.weight

        return F.linear(input, new_weight, self.bias)


class thinLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bw_size=0):
        super(thinLinear, self).__init__(in_features=in_features,
                                         out_features=out_features, bias=bias)

        self.binary_weight = Parameter(torch.ones(bw_size))

    def forward(self, input):
        
        # use torch.gather to generate new input 
        binary_mask = torch.nonzero(self.binary_weight.data).view(1, -1)
        thin_input = torch.gather(input, 1, Variable(binary_mask.expand(input.size(0), binary_mask.size(1)))).view(input.size(0), -1)
        # use torch.maksked_select to get new input
        # thin_input = torch.masked_select(
        #    input, self.binary_weight.unsqueeze(0).byte().data).view(input.size(0), -1)
        return F.linear(thin_input, self.weight, self.bias)

class thinConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bw_size=0):
        super(thinConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.binary_weight = Parameter(torch.ones(self.weight.data.size(bw_size)))

    def forward(self, input):
        
        binary_mask = torch.nonzero(self.binary_weight.data).view(1, -1, 1, 1)
        thin_input = torch.gather(input, 1, Variable(binary_mask.expand(input.size(0), binary_mask.size(1), input.size(2), input.size(3))))

        return F.conv2d(thin_input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)