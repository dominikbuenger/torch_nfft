#pragma once

#include <torch/extension.h>


torch::Tensor
nfft_adjoint_cuda(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t N,
    int64_t m,
    int64_t real_output);


torch::Tensor
nfft_forward_cuda(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t m,
    int64_t real_output);


torch::Tensor
nfft_fastsum_cuda(
    const torch::Tensor sources,
    const torch::Tensor targets,
    const torch::Tensor x,
    const torch::Tensor coeffs,
    const torch::optional<torch::Tensor> opt_source_batch,
    const torch::optional<torch::Tensor> opt_target_batch,
    const int64_t N,
    const int64_t m);


torch::Tensor
gaussian_analytical_coeffs_cuda(
    const double sigma,
    const int64_t N,
    const int64_t dim);
