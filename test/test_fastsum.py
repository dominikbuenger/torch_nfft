

import torch
import math

from torch_nfft import nfft_fastsum


n = 5
dim = 2
c = 1

N = 16
m = 3


pos = torch.rand((n, dim)).cuda() - 0.5
pos /= torch.linalg.norm(pos, dim=1, keepdim=True)
x = torch.rand((n, c)).cuda()

sigma = 0.5


coeffs1d = (0.5 * sigma / math.sqrt(math.pi)) * torch.exp(-0.25 * sigma**2 * torch.arange(-N//2, -N//2 + N) ** 2).cuda()

coeffs = torch.ones(*((N,)*dim)).cuda()
for d in range(dim):
    coeffs *= coeffs1d.view(N, *((1,)*d))



print("Coeffs slice:")
print(coeffs[0])
print(coeffs[N//2])
print()


A_nfft = nfft_fastsum(torch.eye(n).cuda(), coeffs, pos, N=N, m=m)

y_nfft = nfft_fastsum(x, coeffs, pos, N=N, m=m)


print("Approximate A:")
print(A_nfft)
print("Approximate result:")
print(y_nfft)
print()


A_true = torch.exp(- (pos.reshape(1,n,dim) - pos.reshape(n,1,dim)).pow(2).sum(-1) / (sigma**2))
y_true = A_true @ x

print("True A:")
print(A_true)
print("True result:")
print(y_true)
print()
