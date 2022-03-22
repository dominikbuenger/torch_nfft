

import torch
import torch_nfft


n = 4
dim = 2
b = 2
c = 3
diameter = 10
sigma = 0.2 * diameter
N = 16
m = 3
p = 0

pos = diameter*(torch.rand((n*b, dim)).cuda() - 0.5)

batch = None if b <= 1 else torch.div(torch.arange(n*b).cuda(), n, rounding_mode='trunc')

x = torch.rand((n*b, c)).cuda()


kernel = torch_nfft.GaussianKernel(sigma, dim, N, m, shift_by_center=True, max_infinity_norm=diameter/2, reg_degree=p)

y_approx = kernel(pos, batch=batch) @ torch.eye(n*b).cuda()

y_true = torch_nfft.exact_gaussian_matrix(sigma, pos, batch=batch)

print("Approximate result:")
print(y_approx)
print("True result:")
print(y_true)

print("Relative maximum absolute error:", torch.abs(y_approx - y_true).max().item() / y_true.abs().max().item())
