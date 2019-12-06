import numpy as np
from .ds_pre_resnet import *
from .ds_resnet import *
from .ds_vgg_cifar import *

__all__ = ['project_l1_ball', 'extract_d', 'extract_dp']
"""
given layer selection vector d \in \mmR^{n}, z denotes sparsity iterm of d
sort d into \mu and \mu_1 \le \mu_2 \dots \le \mu_n
find max index p = \max{j\in[n]: \mu_j-1/j*(\sum_{r=1}^j\mu-z)>0}
define \thta = 1/p(\sum_{i=1}^p -z)
output: d_i = max{d_i - \theta, 0}

reference code
function x = proximity_L1squared(x0, z)
d = length(x0);
s = sign(x0);
x0 = abs(x0);
[y0,ind_sort] = sort(x0, 'descend');
ycum = cumsum(y0);

val = (ycum -z)./ (1:d);
ind = find(y0 > val);
rho = ind(end);
tau = val(rho);
y = y0 - tau;
ind = find(y < 0);
y(ind) = 0;
x(ind_sort) = y;
x = s .* x;
"""

def project_l1_ball(model, z):
    d = extract_dp(model)
    # d = np.minimum(d, 1)
    # d = np.clip(d, 0, 1)
    if d.min() != d.max():
        d = (d-d.min())/(d.max()-d.min())
    # if d.min() != d.max():
    #     d = (d-d.mean())/np.power(d.std(), 2)

    d_sign = np.sign(d)
    d = np.abs(d)
    ind_sort = np.argsort(-d)
    y_0 = d[ind_sort]
    y_cum = np.cumsum(y_0)
    # print "y_cum", y_cum
    val = (y_cum-z)/np.array(list(range(1, d.shape[0]+1)))
    # print "val", val
    ind = np.where(y_0>val)[0]
    # print "ind", ind
    rho = ind[-1]
    # print "rho", rho
    tau = val[rho]
    # print "tau", tau
    y = y_0 - tau
    # print "y", y
    ind = np.where(y<0)[0]
    y[ind] = 0
    d = y[ind_sort]*d_sign
    # print d
    y_ind = np.argsort(-y)
    k = int(z)
    threashold = y[y_ind[k]]
    block_count = 0
    # print "update d"
    for block in model.modules():
        # print "modules", type(block)
        if isinstance(block, (dsPreBasicBlock, dsBottleneck, dsBasicBlock, dsVGGBlock)):
            # print "instance"
            if d[block_count] >= threashold:
                block.d.data.fill_(d[block_count])
            else:
                block.d.data.fill_(0) 
                # print "set to zero"
                # assert False
            # block.d.data.fill_(d[block_count])
            '''if d[block_count] >= threashold:
                block.dt.data.fill_(1)
            else:
                block.dt.data.fill_(0)'''
            block_count += 1
    # assert False

def extract_dp(model):
    d_list = []
    for block in model.modules():
        if isinstance(block, (dsPreBasicBlock, dsBottleneck, dsBasicBlock, dsVGGBlock)):
            # d_list.append(block.d.data[0])
            d_list.append(block.d_p.data[0])
    # print "get d:", d_list
    return np.array(d_list)

def extract_d(model):
    d_list = []
    for block in model.modules():
        if isinstance(block, (dsPreBasicBlock, dsBottleneck, dsBasicBlock, dsVGGBlock)):
            # d_list.append(block.d.data[0])
            d_list.append(block.d.data[0])
    # print "get d:", d_list
    return np.array(d_list)
 