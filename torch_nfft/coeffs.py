
import torch
_gaussian_analytic_coeffs = torch.ops.torch_nfft.gaussian_analytic_coeffs
_gaussian_interpolated_coeffs = torch.ops.torch_nfft.gaussian_interpolated_coeffs
_interpolation_grid = torch.ops.torch_nfft.interpolation_grid
_radial_interpolation_grid = torch.ops.torch_nfft.radial_interpolation_grid
_interpolated_kernel_coeffs = torch.ops.torch_nfft.interpolated_kernel_coeffs


def gaussian_analytic_coeffs(sigma, dim=3, N=16):
    return _gaussian_analytic_coeffs(sigma, N, dim)


def gaussian_interpolated_coeffs(sigma, dim=3, N=16, p=-1, eps=0.0):
    return _gaussian_interpolated_coeffs(sigma, N, dim, p, eps)


def interpolation_grid(dim=3, N=16):
    return _interpolation_grid(N, dim)


def radial_interpolation_grid(dim=3, N=16):
    return _radial_interpolation_grid(N, dim)


def interpolated_kernel_coeffs(grid_values):
    return _interpolated_kernel_coeffs(grid_values)
