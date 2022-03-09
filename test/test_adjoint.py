

import torch
from torch_nfft import nfft_adjoint, ndft_adjoint



# pos = torch.tensor(
#     [[0.1, 0.1],
#     [-0.1, -0.1],
#     [-0.2, 0.2]],
#     #[[0.0, 0.0], [0,0], [0,0]],
#     dtype=torch.float).cuda()
# x = torch.tensor(
#     [1,2,3],
#     # [1.0],
#     dtype=torch.float).cuda()
# N = 16
# m = 4

d = 2
b = 3
n = 1000
c = 10
pos = torch.rand((n*b,d), dtype=torch.float).cuda() - 0.5
pos /= 4*torch.linalg.norm(pos, dim=1, keepdim=True)
batch = None if b <= 1 else torch.div(torch.arange(n*b).cuda(), n, rounding_mode='trunc')
x = torch.rand((n*b, c), dtype=torch.float).cuda()
N = 16
m = 4

y_nfft = nfft_adjoint(x, pos, batch, N=N, m=m)
# print("Fast:")
# print(y_nfft[0])
print("y_nfft shape: ", y_nfft.shape)

# y_ndft = ndft_adjoint(x, pos, batch, N=N)
y_ndft = torch.cat([ndft_adjoint(x[:,i], pos, batch, N=N)[...,None] for i in range(c)], dim=-1)
# print("Exact:")
# print(y_ndft)
print("y_ndft shape: ", y_nfft.shape)

print("Difference between fast and exact:")
l1 = lambda v: torch.abs(v).sum().item()
print("L1 norm: {:.2e} absolute / {:.2e} relative".format(l1(y_nfft - y_ndft), l1(y_nfft - y_ndft) / l1(y_ndft)))
l2 = lambda v: torch.linalg.norm(v).item()
print("L2 norm: {:.2e} absolute / {:.2e} relative".format(l2(y_nfft - y_ndft), l2(y_nfft - y_ndft) / l2(y_ndft)))
linf = lambda v: torch.abs(v).max().item()
print("Linf norm: {:.2e} absolute / {:.2e} relative".format(linf(y_nfft - y_ndft), linf(y_nfft - y_ndft) / linf(y_ndft)))
