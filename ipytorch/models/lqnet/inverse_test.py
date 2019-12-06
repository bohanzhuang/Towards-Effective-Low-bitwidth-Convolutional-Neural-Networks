from ipytorch.models.lqnet.lqnet_quant import QConv2d, QReLU
import torch
import time

def inverse(A):
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    inv = []
    inv.append(A[1][1] / det)
    inv.append(-A[0][1] / det)
    inv.append(-A[1][0] / det)
    inv.append(A[0][0] / det)
    A_inv = torch.reshape(torch.stack(inv), (2, 2))
    return A_inv

a_cpu = torch.randn(2,2)
a_gpu = a_cpu.cuda()

time_start = time.perf_counter()
b = torch.inverse(a_cpu)
time_end = time.perf_counter()
print('CPU time is {:.02e}s'.format(time_end - time_start))

time_start = time.perf_counter()
b = inverse(a_cpu)
time_end = time.perf_counter()
print('CPU time is {:.02e}s'.format(time_end - time_start))

torch.cuda.synchronize()
time_start = time.perf_counter()
b = torch.inverse(a_gpu)
torch.cuda.synchronize()
time_end = time.perf_counter()
print('GPU time is {:.02e}s'.format(time_end - time_start))

torch.cuda.synchronize()
time_start = time.perf_counter()
b = inverse(a_gpu)
torch.cuda.synchronize()
time_end = time.perf_counter()
print('GPU time is {:.02e}s'.format(time_end - time_start))