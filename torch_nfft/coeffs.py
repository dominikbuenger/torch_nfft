
import torch
_gaussian_analytical_coeffs = torch.ops.torch_nfft.gaussian_analytical_coeffs

def gaussian_analytical_coeffs(sigma, dim=3, N=16):
    return _gaussian_analytical_coeffs(sigma, N, dim)
