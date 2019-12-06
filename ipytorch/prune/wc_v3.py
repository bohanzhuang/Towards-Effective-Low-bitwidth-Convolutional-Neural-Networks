import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F

__all__ = ["wcConv2d", "wcLinear", "thinLinear", "thinConv2d"]


# weight channels
# this code needs testing and can only used for fine-tuning
# compute cosine distance between feature maps
def compute_cosine(x, mask):
    x_view = x.view(x.size(0), x.size(1), -1)

    x_norm = (x_view*x_view).sum(2).pow(0.5).unsqueeze(2)
    
    x_xt = torch.matmul(x_view, x_view.transpose(1, 2))
    
    x_norm_t = torch.matmul(x_norm, x_norm.transpose(1, 2))
    
    cos_dist = (torch.clamp(x_xt, min=1e-6) / torch.clamp(x_norm_t, min=1e-6)).sum(0)
    mask_matrix = torch.matmul(mask.unsqueeze(1), mask.unsqueeze(0)) - Variable(torch.eye(mask.size(0)).type_as(cos_dist.data))
    cos_dist = (cos_dist * mask_matrix).sum(1) / (mask.sum()-1) + (1-mask)*x.size(0)

    return cos_dist

class wcConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, rate=0.):
        super(wcConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.binary_weight = Parameter(torch.ones(self.weight.size(1)))
        self.register_buffer("float_weight", torch.zeros(self.weight.size(1)))
        self.register_buffer('rate', torch.ones(1).fill_(rate))

    def update_mask(self):
        if self.rate[0] == 0:
            return
        self.binary_weight.data.copy_(
            self.float_weight.le(self.rate[0]).float())
        self.float_weight.fill_(0)
    
    def forward(self, input):
        if self.training:
            self.float_weight += compute_cosine(input, self.binary_weight).data

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

        self.binary_weight = Parameter(torch.ones(self.weight.size(1)))
        self.register_buffer("float_weight", torch.zeros(self.weight.size(1)))
        self.register_buffer('rate', torch.ones(1).fill_(rate))

    def update_mask(self):
        self.binary_weight.data.copy_(
            self.float_weight.le(self.rate[0]).float())
        self.float_weight.fill_(0)

    def forward(self, input):
        if self.training:
            self.float_weight += compute_cosine(input).data
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
