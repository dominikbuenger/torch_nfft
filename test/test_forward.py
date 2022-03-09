

import torch
from torch_nfft import nfft_forward, ndft_forward



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
b = 1
n = 10
c = 1
N = 16
m = 4

pos = torch.rand((n*b,d), dtype=torch.float).cuda() - 0.5
pos /= 4*torch.linalg.norm(pos, dim=1, keepdim=True)
batch = None if b <= 1 else torch.div(torch.arange(n*b).cuda(), n, rounding_mode='trunc')

x = torch.rand((b,) + (N,)*d + (c,), dtype=torch.float).cuda()

y_nfft = nfft_forward(x, pos, batch, m=m)
# print("Fast:")
# print(y_nfft[0])
print("y_nfft shape: ", y_nfft.shape)

# y_ndft = ndft_adjoint(x, pos, batch, N=N)
y_ndft = torch.cat([ndft_forward(x[...,i], pos, batch)[...,None] for i in range(c)], dim=-1)
# print("Exact:")
# print(y_ndft)
print("y_ndft shape: ", y_nfft.shape)

print("Difference between fast and exact:")
print("L1 norm:", torch.abs(y_nfft - y_ndft).sum().item())
print("L2 norm:", torch.linalg.norm(y_nfft - y_ndft).item())
print("Linf norm:", torch.abs(y_nfft - y_ndft).max().item())
