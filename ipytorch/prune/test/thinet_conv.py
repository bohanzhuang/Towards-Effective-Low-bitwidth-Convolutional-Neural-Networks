import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np 
import random
import torch.nn.functional as F
# ------------------method--------------------
# ThiNet: a filter level pruning method for deep neural network compression
# input layer_i and layer_{i+1}
# remove channels from layer_{i+1} according to the output of layer_{i+1}
# then remove filters from layer_i according to layer_{i+1}
# return pruned layrs
        

class qConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(qConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)
        self.filter_mask = torch.zeros(out_channels, in_channels)
        self.x_cache = None
        self.locations = 10
        self.ratio = 0.5
        self.remove_channels = None
        self.select_channels = None
        self.thin_weight = None

    def compute(self, x, y):
        print((x.size()))
        print((self.padding))
        padding = nn.ZeroPad2d((0, 0, self.padding[0], self.padding[1]))
        x = padding(x).data
        print((x.size()))

        y_h = torch.LongTensor(np.random.randint(y.size(2), size=self.locations*y.size(0))).cuda()
        y_w = torch.LongTensor(np.random.randint(y.size(3), size=self.locations*y.size(0))).cuda()
        y_d = torch.LongTensor(np.random.randint(y.size(1), size=self.locations*y.size(0))).cuda()

        w_select = self.weight.data[y_d]
        print((w_select.size()))
        
        x_h = y_h*self.stride[0]
        x_w = y_w*self.stride[1]

        temp_cache = tuple()
        for i in range(y_h.size(0)):
            x_select = x[i/self.locations, :, x_h[i]:x_h[i]+self.kernel_size[0], x_w[i]:x_w[i]+self.kernel_size[1]]*w_select[i]
            x_select = x_select.unsqueeze(0)
            temp_cache = temp_cache+(x_select, )
        
        temp_cache = torch.cat(temp_cache, 0)
        if self.x_cache is None:
            self.x_cache = temp_cache
        else:
            self.x_cache = torch.cat((self.x_cache, temp_cache), 0)
        print((self.x_cache.size()))

    def prune(self):
        I = list(range(self.in_channels))
        # print I
        T = []
        for i in range(int(self.in_channels*(1-self.ratio))):
            min_value = None
            for item in I:
                # print item
                tempT = T[:]
                tempT.append(item)
                # print tempT
                tempT = torch.LongTensor(tempT).cuda()
                value = self.x_cache.index_select(1, tempT).sum(1).pow(2).sum()
                if min_value is None or min_value > value:
                    min_value = value
                    min_i = item
            print(("min_i: ", min_i))
            I.remove(min_i)
            print(("len I: ", len(I)))
            T.append(min_i)
            print(("len T: ", len(T)))

        self.remove_channels = torch.LongTensor(sorted(T)).cuda()
        S = list(range(self.in_channels))
        for i in T:
            if i in S:
                S.remove(i)
        self.select_channels = torch.LongTensor(S).cuda()
        self.thin_weight = self.weight.select(1, self.select_channels)
        print((self.thin_weight.size()))
        
    def forward(self, input):        
        
        y = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        self.compute(input, y)
        return y
    
def main():
    old_conv = qConv2d(64, 3, 3)
    old_conv.cuda()
    new_conv = nn.Conv2d(32, 3, 3)
    new_conv.cuda()
    for i in range(20):
        rand_input = Variable(torch.randn(100, 64, 32, 32).cuda())
        y = old_conv(rand_input)
    old_conv.prune()
    new_conv.weight.data.copy_(old_conv.thin_weight.data)

if __name__ == '__main__':
    main()