

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

d = 3
n = 1000
pos = torch.rand((n,d), dtype=torch.float).cuda() - 0.5
pos /= 4*torch.linalg.norm(pos, dim=1, keepdim=True)
x = torch.rand(n, dtype=torch.float).cuda()
N = 32
m = 6

y_nfft = nfft_adjoint(x, pos, N=N, m=m)
# print("Fast:")
# print(y_nfft[0])

y_ndft = ndft_adjoint(x, pos, N=N)
# print("Exact:")
# print(y_ndft)

print("Difference between fast and exact:")
print("L1 norm:", torch.abs(y_nfft[0] - y_ndft).sum().item())
print("L2 norm:", torch.linalg.norm(y_nfft[0] - y_ndft).item())
print("Linf norm:", torch.abs(y_nfft[0] - y_ndft).max().item())
