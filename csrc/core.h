
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
