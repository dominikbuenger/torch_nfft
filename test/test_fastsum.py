

import torch
import math

import torch_nfft


n = 200
dim = 2
c = 1
sigma = 0.2
N = 8
m = 3


pos = torch.rand((n, dim)).cuda() - 0.5
pos /= 4*torch.linalg.norm(pos, dim=1).max()

A_true = torch.exp(- (pos.reshape(1,n,dim) - pos.reshape(n,1,dim)).pow(2).sum(-1) / (sigma**2))

if n < 10:
    print("True A:")
    print(A_true)
    print()



coeffs = torch_nfft.gaussian_analytic_coeffs(sigma, dim=dim, N=N)

A_nfft = torch_nfft.nfft_fastsum(torch.eye(n).cuda(), coeffs, pos, cutoff=m)

print("Approximate A with analytic coeffs:")
if n < 10:
    print(A_nfft)
print("Maximum absolute entry error: ", torch.abs(A_nfft - A_true).max().item())
print()



A_trigon = torch_nfft.exact_trigonometric_matrix(coeffs, pos).real

print("Trigonometric A with analytic coeffs:")
if n < 10:
    print(A_trigon)
print("Maximum absolute entry error: ", torch.abs(A_trigon - A_true).max().item())
print()



coeffs = torch_nfft.gaussian_interpolated_coeffs(sigma, dim=dim, N=N, p=0)


A_nfft = torch_nfft.nfft_fastsum(torch.eye(n).cuda(), coeffs, pos, cutoff=m)

print("Approximate A with interpolated coeffs:")
if n < 10:
    print(A_nfft)
print("Maximum absolute entry error: ", torch.abs(A_nfft - A_true).max().item())
print()


A_trigon = torch_nfft.exact_trigonometric_matrix(coeffs, pos).real

print("Trigonometric A with interpolated coeffs:")
if n < 10:
    print(A_trigon)
print("Maximum absolute entry error: ", torch.abs(A_trigon - A_true).max().item())
