//#include <torch/extension.h>
#include <torch/script.h>

// #include "cpu/core_cpu.h"
#include "cuda/core_cuda.h"


/**
    Adjoint NFFT: Fast approximation of the adjoint NDFT (Non-equispaced
    Discrete Fourier Transform), i.e., the approximation of
        y[b, k[0], ..., k[dim-1], ...] =
            \sum_{i: batch[i]==b} exp(2j pi dot(k, pos[i])) * x[i, ...]
    for frequency indices k[d] = 0, ..., N-1 for all d=0,...,dim-1.
    where pos[i] is the dim-dimensional coordinate vector of point #i,
            given in [-1/2, 1/2)
          batch[i] is the batch index of point #i, 0 <= batch[i] < batch_size,
          x[i, ...] holds the real-valued coefficients for point #i,
          2j is two times the imaginary unit,
          dot(..., ...) refers to the scalar product of two vectors.

    The spatial dimension `dim` must be 1, 2, or 3.
    The trailing dimensions of x define separate independent NDFTs to be
        evaluated in parallel.

    pos, x, batch are given as PyTorch tensors with pos.shape == (n,dim),
        pos.dtype == torch.float, x.shape[0] == n, x.dtype == torch.float.
        batch can be None, which refers to all points belonging to batch 0.
        Otherwise it must have batch.shape == (n,) and dtype torch.long
        with entries ordered from 0 to batch_size-1.

    The output y is given as a tensor with y.dtype == torch.cfloat and
        y.dim == dim + x.dim. Its shape is given as
        y.shape[0] == batch_size, y.shape[1] == ... == y.shape[dim-1] == N,
        y.shape[dim] == N/2 + 1, and the remaining dimensions have the same
        sizes as the trailing dimensions of the input x.
    The final spatial dimension of y is only half as long due to redundancy for
    real-valued input, cf. the CUFFT documentation.

    The frequency bandwidth N must be a power of two.
    The cutoff parameter m controls the approximation quality and must be in
        1 <= m < N/2.
*/
torch::Tensor
nfft_adjoint(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t N,
    int64_t m,
    int64_t real_output)
{
    AT_ASSERTM(x.device().is_cuda(), "torch_nfft.nfft_adjoint is currently only implemented for GPU tensors");

    return nfft_adjoint_cuda(pos, x, opt_batch, N, m, real_output);
}


/**
    Forward NFFT: Fast approximation of the forward NDFT (Non-equispaced
    Discrete Fourier Transform), i.e., the approximation of
        y[i, ...] = \sum_{k[0]=0}^{N-1} ... \sum_{k[dim-1]=0}^{N-1}
            exp(-2j pi dot(k, pos[i])) x[batch[i], k[0],...,k[dim-1], ...]
    where pos[i] is the dim-dimensional coordinate vector of point #i,
            given in [-1/2, 1/2),
          batch[i] is the batch index of point #i, 0 <= batch[i] < batch_size,
          x[b, k[0], ..., k[dim-1], ...] holds the complex-valued coefficients
            for the frequency multiindex k for all points in batch #b,
          2j is two times the imaginary unit,
          dot(..., ...) refers to the scalar product of two vectors.
    The spatial dimension `dim` must be 1, 2, or 3.
    The trailing dimensions of x define separate independent NDFTs to be
        evaluated in parallel.

    pos, x, batch are given as PyTorch tensors with pos.shape == (n,dim),
        pos.dtype == torch.float, x.dtype == torch.cfloat,
        x.shape[0] == batch_size, x.shape[1] == ... == x.shape[dim-1] == N,
        x.shape[dim] == N/2 + 1. batch can be None, which refers to all points
        belonging to batch 0. Otherwise it must be 1d with batch.shape == (n,)
        and dtype torch.long with entries ordered from 0 to batch_size-1.

    The final spatial dimension of x is only half as long due to redundancy,
        as we expect the input to be "Hermitian" such that the output is
        real-valued, cf. the CUFFT documentation.

    The output y is given as a tensor with y.dtype == torch.float,
        y.dim == x.dim - dim, and y.shape[0] == n. The remaining dimensions have
        the same sizes as the trailing dimensions of the input x.

    The frequency bandwidth N must be a power of two.
    The cutoff parameter m controls the approximation quality and must be in
        1 <= m < N/2.
*/

torch::Tensor
nfft_forward(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t m,
    int64_t real_output)
{
    AT_ASSERTM(x.device().is_cuda(), "torch_nfft.nfft_forward is currently only implemented for GPU tensors");

    return nfft_forward_cuda(pos, x, opt_batch, m, real_output);
}


torch::Tensor
nfft_fastsum(
    const torch::Tensor sources,
    const torch::Tensor targets,
    const torch::Tensor x,
    const torch::Tensor coeffs,
    const torch::optional<torch::Tensor> opt_source_batch,
    const torch::optional<torch::Tensor> opt_target_batch,
    const int64_t m)
{
    AT_ASSERTM(x.device().is_cuda(), "torch_nfft.nfft_fastsum is currently only implemented for GPU tensors");

    return nfft_fastsum_cuda(sources, targets, x, coeffs, opt_source_batch, opt_target_batch, m);
}


torch::Tensor
gaussian_analytic_coeffs(
    const double sigma,
    const int64_t N,
    const int64_t dim)
{
    return gaussian_analytic_coeffs_cuda(sigma, N, dim);
}


torch::Tensor
gaussian_interpolated_coeffs(
    const double sigma,
    const int64_t N,
    const int64_t dim,
    const int64_t p,
    const double eps)
{
    return gaussian_interpolated_coeffs_cuda(sigma, N, dim, p, eps);
}


torch::Tensor
interpolation_grid(
    const int64_t N,
    const int64_t dim)
{
    return interpolation_grid_cuda(N, dim);
}


torch::Tensor
radial_interpolation_grid(
    const int64_t N,
    const int64_t dim)
{
    return radial_interpolation_grid_cuda(N, dim);
}


torch::Tensor
interpolated_kernel_coeffs(
    const torch::Tensor grid_values)
{
    AT_ASSERTM(grid_values.device().is_cuda(), "torch_nfft.interpolated_kernel_coeffs is currently only implemented for GPU tensors");

    return interpolated_kernel_coeffs_cuda(grid_values);
}



// Register operators for torch
static auto registry = torch::RegisterOperators()
    .op("torch_nfft::nfft_adjoint", &nfft_adjoint)
    .op("torch_nfft::nfft_forward", &nfft_forward)
    .op("torch_nfft::nfft_fastsum", &nfft_fastsum)
    .op("torch_nfft::gaussian_analytic_coeffs", &gaussian_analytic_coeffs)
    .op("torch_nfft::gaussian_interpolated_coeffs", &gaussian_interpolated_coeffs)
    .op("torch_nfft::interpolation_grid", &interpolation_grid)
    .op("torch_nfft::radial_interpolation_grid", &radial_interpolation_grid)
    .op("torch_nfft::interpolated_kernel_coeffs", &interpolated_kernel_coeffs);
