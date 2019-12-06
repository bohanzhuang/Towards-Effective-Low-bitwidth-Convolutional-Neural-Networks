from ipytorch.models.lqnet.lqnet_quant import QConv2d, QReLU
import torch
import time

#test qrelu
qrelu = QReLU(k=2)
for i in range(100):
    x = torch.randn(2, 1, 3, 3, requires_grad=True)
    y = qrelu(x)
    y = y.reshape(-1)
    y = torch.sum(y)
    y.backward()
    print(y)
    print(x.grad)

#test conv
net = QConv2d(1, 32, kernel_size=3, k=2)
for i in range(100):
    x = torch.randn(2, 1, 3, 3, requires_grad=True)
    y = net(x)
    # y = qrelu(y)
    y = y.reshape(-1)
    y = torch.sum(y)
    y.backward()
