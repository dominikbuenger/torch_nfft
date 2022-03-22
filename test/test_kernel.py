

import torch
import torch_nfft


n = 4
dim = 2
b = 2
c = 3
diameter = 10
relative_sigma = 1
N = 16
m = 3
p = 0

pos = diameter*(torch.rand((n*b, dim)).cuda() - 0.5)

batch = None if b <= 1 else torch.div(torch.arange(n*b).cuda(), n, rounding_mode='trunc')

x = torch.rand((n*b, c)).cuda()


print("WITH ABSOLUTE SIGMA:")

kernel = torch_nfft.GaussianKernel(relative_sigma*diameter,
                                    dim, N, m, shift_by_center=True,
                                    max_infinity_norm=diameter/2, reg_degree=p)
y_approx = kernel(pos, batch=batch) @ torch.eye(n*b).cuda()

y_true = torch_nfft.exact_gaussian_matrix(relative_sigma*diameter, pos, batch=batch)

print("Approximate result:")
print(y_approx)
print("True result:")
print(y_true)
print("Relative maximum absolute error:", torch.abs(y_approx - y_true).max().item() / y_true.abs().max().item())


print()
print("WITH RELATIVE SIGMA:")

kernel = torch_nfft.GaussianKernel(relative_sigma, dim, N, m, shift_by_center=True, reg_degree=p)
y_approx = kernel(pos, batch=batch) @ torch.eye(n*b).cuda()

pos_shifted_scaled = torch_nfft.utils.scale_points_by_norm(
                        torch_nfft.utils.shift_points_by_center(
                            pos, batch=batch)[0],
                        batch=batch,
                        norm="infinity" if p<0 else "euclidean")[0]

y_true = torch_nfft.exact_gaussian_matrix(relative_sigma, pos_shifted_scaled, batch=batch)

print("Approximate result:")
print(y_approx)
print("True result:")
print(y_true)
print("Relative maximum absolute error:", torch.abs(y_approx - y_true).max().item() / y_true.abs().max().item())
