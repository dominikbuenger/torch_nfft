

import torch
import math

from torch_nfft import nfft_fastsum, ndft_fastsum, gaussian_analytical_coeffs


n = 5
dim = 2
c = 1
sigma = 0.1
N = 32
m = 3


pos = torch.rand((n, dim)).cuda() - 0.5
pos /= 4*torch.linalg.norm(pos, dim=1, keepdim=True)


# coeffs1d = sigma * math.sqrt(math.pi) * torch.exp(-(math.pi * sigma * torch.arange(-N//2, -N//2 + N).cuda()) ** 2)
# coeffs = torch.ones(*((N,)*dim)).cuda()
# for d in range(dim):
#     coeffs *= coeffs1d.view(N, *((1,)*d))

coeffs = gaussian_analytical_coeffs(sigma, dim=dim, N=N)


A_nfft = nfft_fastsum(torch.eye(n).cuda(), coeffs, pos, N=N, m=m)

print("Approximate A:")
print(A_nfft)
print()


grid1d = torch.arange(-N//2, N//2, dtype=torch.float).cuda()
grid = torch.cat([g[...,None] for g in torch.meshgrid(*((grid1d,)*dim), indexing='ij')], dim=dim)
tmp = pos.reshape(1,n,dim) - pos.reshape(n,1,dim)
tmp = torch.tensordot(grid, tmp, dims=([-1],[-1]))
tmp = torch.exp(2j * torch.pi * tmp)
tmp = torch.tensordot(coeffs.to(torch.cfloat), tmp, dims=dim)
A_trigon = tmp.real

print("Trigonometric A:")
print(A_trigon)
print()


A_true = torch.exp(- (pos.reshape(1,n,dim) - pos.reshape(n,1,dim)).pow(2).sum(-1) / (sigma**2))

print("True A:")
print(A_true)
