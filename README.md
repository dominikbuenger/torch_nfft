

# Non-equispaced Fast Fourier Transforms on GPU for PyTorch tensors

This is a GPU-based implementation of the NFFT method for fast approximation of one, two, or three-dimensional Non-equispaced Discrete Fourier Transform (NDFT) expressions.

Compared to other implementations, this package focuses on integration in the PyTorch tensor framework as well as enabling ''batched'' transforms, i.e., transforming multiple coefficient sets for multiple point sets at the same time.


## Restrictions

This package is work in progress.

* The point set must be `dim`-dimensional with `dim<=3`.
* The forward operation must be C2R and the adjoint operation must be R2C.
* The frequency bandwidth `N` must be the same in all dimensions, and $N$ is expected to be a power of two.

## Prerequesites

This package depends on PyTorch, Numpy setuptools, CUDA, and [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html).

## The adjoint operation

Fast approximation of the adjoint NDFT (Non-equispaced
Discrete Fourier Transform), i.e., the approximation of
```
y[b, k[0], ..., k[d-1], ...] =
    sum_{i: \mathrm{batch}[i] = b} exp(2j pi dot(k, pos[i])) * x[i, ...]
```
for frequency indices `k[d] = 0, ..., N-1` for all `d=0,...,dim-1`,
where `pos[i]` is the `dim`-dimensional coordinate vector of point #`i`,
        given in `[-1/2, 1/2)`
      `batch[i]` is the batch index of point #`i`, `0 <= batch[i] < batch_size`,
      `x[i, ...]` holds the real-valued coefficients for point #`i`,
      `2j` is two times the imaginary unit,
      `dot(..., ...)` refers to the scalar product of two vectors.

The spatial dimension `dim` must be 1, 2, or 3.
The trailing dimensions of `x` define separate independent NDFTs to be
    evaluated in parallel.

`pos`, `x`, `batch` are given as PyTorch tensors with `pos.shape == (n,dim)`,
    `pos.dtype == torch.float`, `x.shape[0] == n`, `x.dtype == torch.float`.
    `batch` can be `None`, which refers to all points belonging to batch `0`.
    Otherwise it must have `batch.shape == (n,)` and `batch.dtype == torch.long`
    with entries ordered from `0` to `batch_size-1`.

The output `y` is given as a tensor with `y.dtype == torch.cfloat` and
    `y.dim == dim + x.dim`. Its shape is given as
    `y.shape[0] == batch_size`, `y.shape[1] == ... == y.shape[dim-1] == N`,
    `y.shape[dim] == N/2 + 1`, and the remaining dimensions have the same
    sizes as the trailing dimensions of the input `x`.
The final spatial dimension of `y` is only half as long due to redundancy for
real-valued input, cf. the CUFFT documentation.

The frequency bandwidth `N` must be a power of two.
The cutoff parameter m controls the approximation quality and must be in
    `1 <= m < N/2`.
