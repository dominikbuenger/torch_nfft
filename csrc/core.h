
#pragma once

#include <torch/extension.h>

torch::Tensor
nfft_adjoint(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t N,
    int64_t m,
    int64_t real_output);

torch::Tensor
nfft_forward(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t m,
    int64_t real_output);


torch::Tensor
nfft_fastsum(
    const torch::Tensor sources,
    const torch::Tensor targets,
    const torch::Tensor x,
    const torch::Tensor coeffs,
    const torch::optional<torch::Tensor> opt_source_batch,
    const torch::optional<torch::Tensor> opt_target_batch,
    const int64_t m);


torch::Tensor
gaussian_analytic_coeffs(
    const double sigma,
    const int64_t N,
    const int64_t dim);


torch::Tensor
gaussian_interpolated_coeffs(
    const double sigma,
    const int64_t N,
    const int64_t dim,
    const int64_t p,
    const double eps);


torch::Tensor
interpolation_grid(
    const int64_t N,
    const int64_t dim);


torch::Tensor
radial_interpolation_grid(
    const int64_t N,
    const int64_t dim);


torch::Tensor
interpolated_kernel_coeffs(
    const torch::Tensor grid_values);
