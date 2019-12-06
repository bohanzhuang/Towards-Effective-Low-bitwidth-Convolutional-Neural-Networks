import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np 
import random

from torch.autograd import Variable
import torch.nn.functional as F
# ------------------method--------------------
# ThiNet: a filter level pruning method for deep neural network compression
# input layer_i and layer_{i+1}
# remove channels from layer_{i+1} according to the output of layer_{i+1}
# then remove filters from layer_i according to layer_{i+1}
# return pruned layrs
        

class qLinear(nn.Linear):
    """
    custom Linear layers for quantization
    """
    def __init__(self, in_features, out_features, bias=True):
        super(qLinear, self).__init__(in_features=in_features,
                                      out_features=out_features, bias=bias)

        self.x_cache = None
        self.locations = 10
        self.ratio = 0.5
        self.remove_channels = None
        self.select_channels = None
        self.thin_weight = None

    def compute(self, x, y):
        print((x.size()))
        x = x.data
        print((x.size()))

        y_d = torch.LongTensor(np.random.randint(y.size(1), size=self.locations*y.size(0))).cuda()

        w_select = self.weight.data[y_d]
        print((w_select.size()))
        
        temp_cache = tuple()
        x_n = torch.LongTensor(list(range(x.size(0)))).cuda().repeat(self.locations)
        temp_cache = x[x_n]*w_select
        
        if self.x_cache is None:
            self.x_cache = temp_cache
        else:
            self.x_cache = torch.cat((self.x_cache, temp_cache))
        print((self.x_cache.size()))

    def prune(self):
        I = list(range(self.in_features))
        # print I
        T = []
        for i in range(int(self.in_features*(1-self.ratio))):
            min_value = None
            for item in I:
                # print item
                tempT = T[:]
                tempT.append(item)
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
        S = list(range(self.in_features))
        for i in T:
            if i in S:
                S.remove(i)
        self.select_channels = torch.LongTensor(S).cuda()
        self.thin_weight = self.weight.select(1, self.select_channels)
        print((self.thin_weight.size()))
        
    def forward(self, input):        
        
        y = F.linear(input, self.weight, self.bias)
        self.compute(input, y)
        return y
    
def main():
    old_fc = qLinear(64, 10)
    old_fc.cuda()
    new_fc = nn.Linear(32, 10)
    new_fc.cuda()
    for i in range(10):
        rand_input = Variable(torch.randn(1000, 64).cuda())
        y = old_fc(rand_input)
    old_fc.prune()
    new_fc.weight.data.copy_(old_fc.thin_weight.data)

if __name__ == '__main__':
    main()