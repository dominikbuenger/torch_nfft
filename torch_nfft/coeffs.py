
import torch
_gaussian_analytical_coeffs = torch.ops.torch_nfft.gaussian_analytical_coeffs
_interpolation_grid = torch.ops.torch_nfft.interpolation_grid
_radial_interpolation_grid = torch.ops.torch_nfft.radial_interpolation_grid
_interpolated_kernel_coeffs = torch.ops.torch_nfft.interpolated_kernel_coeffs


def gaussian_analytical_coeffs(sigma, dim=3, N=16):
    return _gaussian_analytical_coeffs(sigma, N, dim)


def interpolation_grid(dim=3, N=16):
    return _interpolation_grid(N, dim)


def radial_interpolation_grid(dim=3, N=16):
    return _radial_interpolation_grid(N, dim)


def interpolated_kernel_coeffs(grid_values):
    return _interpolated_kernel_coeffs(grid_values)
