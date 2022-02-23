
#pragma once

#include <torch/extension.h>

torch::Tensor
nfft_adjoint(
    torch::Tensor sources,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_source_batch,
    int64_t N,
    int64_t m);


torch::Tensor
nfft_forward(
    torch::Tensor targets,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_target_ptr,
    double tol);
