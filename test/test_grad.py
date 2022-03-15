

import torch
import torch_nfft
import numpy as np


n = 5
dim = 2
b = 2
c = 3
sigma = 0.2
N = 16
m = 3

pos = torch.rand((n*b, dim)).cuda() - 0.5
pos /= 4*torch.linalg.norm(pos, dim=1).max()

batch = None if b <= 1 else torch.div(torch.arange(n*b).cuda(), n, rounding_mode='trunc')



### ADJOINT

x = torch.rand((n*b, c), requires_grad=True).cuda()
x.retain_grad()

loss = torch_nfft.nfft_adjoint(x, pos, batch, N, m).abs().sum()
loss.backward()
dx = x.grad

x = x.detach()
perturbation = 1e-3
dx_fd = torch.zeros_like(x, requires_grad=False)
for i in np.ndindex(x.shape):
    x[i] += perturbation
    perturbed_loss = torch_nfft.nfft_adjoint(x, pos, batch, N, m).abs().sum()
    dx_fd[i] = (perturbed_loss - loss) / perturbation
    x[i] -= perturbation

print("ADJOINT OPERATION")
# print("Computed gradient:")
# print(dx)
# print("Finite differences:")
# print(dx_fd)
print("Maximum relative difference: ", (dx - dx_fd).abs().max().item() / dx_fd.abs().max().item())



### FORWARD

x = torch.rand((b, *((N,)*dim), c), requires_grad=True).cuda()
x.retain_grad()

loss = torch_nfft.nfft_forward(x, pos, batch, m).abs().sum()
loss.backward()
dx = x.grad

x = x.detach()
perturbation = 1e-3
dx_fd = torch.zeros_like(x, requires_grad=False)
for i in np.ndindex(x.shape):
    x[i] += perturbation
    perturbed_loss = torch_nfft.nfft_forward(x, pos, batch, m).abs().sum()
    dx_fd[i] = (perturbed_loss - loss) / perturbation
    x[i] -= perturbation

print("FORWARD OPERATION")
# print("Computed gradient:")
# print(dx[0,...,0])
# print("Finite differences:")
# print(dx_fd[0,...,0])
print("Maximum relative difference: ", (dx - dx_fd).abs().max().item() / dx_fd.abs().max().item())



### FASTSUM

x = torch.rand((n*b, c), requires_grad=True).cuda()
x.retain_grad()

coeffs = torch_nfft.gaussian_interpolated_coeffs(0.2, dim, N)

loss = torch_nfft.nfft_fastsum(x, coeffs, pos, batch=batch, cutoff=m).abs().sum()
loss.backward()
dx = x.grad

x = x.detach()
perturbation = 1e-3
dx_fd = torch.zeros_like(x, requires_grad=False)
for i in np.ndindex(x.shape):
    x[i] += perturbation
    perturbed_loss = torch_nfft.nfft_fastsum(x, coeffs, pos, batch=batch, cutoff=m).abs().sum()
    dx_fd[i] = (perturbed_loss - loss) / perturbation
    x[i] -= perturbation

print("FASTSUM OPERATION")
# print("Computed gradient:")
# print(dx[0,...,0])
# print("Finite differences:")
# print(dx_fd[0,...,0])
print("Maximum relative difference: ", (dx - dx_fd).abs().max().item() / dx_fd.abs().max().item())
