
#define SQRT_PI 1.77245385090551602729816748334114518279754945f
#define SQUARE_PI 9.86960440108935861883449099987615113531369940f


__global__ void
fill_gaussian_analytic_coeffs_kernel(
    torch::PackedTensorAccessor64<float,1> coeffs_acc,
    const float sigma,
    const int dim,
    const int64_t N,
    const int64_t prod_N)
{
    int64_t idx, i;
    int d;
    float value;

    for (idx = blockIdx.x*blockDim.x + threadIdx.x; idx < prod_N; idx += gridDim.x*blockDim.x)
    {
        i = idx;
        value = 1.0f;
        for (d=0; d<dim; ++d) {
            // d-th component of multiindex l in [-N//2,...,N/2-1]^d is
            // given by (i % N - N/2)
            value *= SQRT_PI * sigma * expf(-sigma*sigma*SQUARE_PI*(i%N - N/2)*(i%N - N/2));
            i /= N;
        }
        coeffs_acc[idx] = value;
    }
}


__global__ void
fill_gaussian_regularized_values_kernel(
    cufftComplex *b,
    const float sigma2,
    const int dim,
    const int64_t N,
    const int64_t prod_N,
    const int64_t p,
    const float eps)
{
    int64_t idx, b_idx, i;
    int d;
    float r2, value;

    for (idx = blockIdx.x*blockDim.x + threadIdx.x; idx < prod_N; idx += gridDim.x*blockDim.x)
    {
        // idx has the form ((i[0]*N + i[1])*N + ...)*N + i[dim-1]
        // where i goes from 0 to N-1, which is shifted from l in [-N/2,...,N/2-1]
        i = idx;
        b_idx = 0;
        r2 = 0.0f;
        for (d=0; d<dim; ++d) {
            // i % N == i[dim-d-1] == l[dim-d-1] + N/2
            r2 += (((i % N) / (float)N) - 0.5f) * (((i % N) / (float)N) - 0.5f);
            b_idx = N*b_idx + ((i + N/2) % N);
            i /= N;
        }

        if (p < 0 || r2 <= (0.5f - eps)*(0.5f - eps)) {
            value = expf(-r2 / sigma2);
        }
        else if (r2 >= 0.25f) {
            value = expf(-0.25 / sigma2);
        }
        else {
            // regularized part: not implemented yet
        }

        b[b_idx] = make_cuFloatComplex(value, 0.0f);
    }
}


__global__ void
fill_interpolation_grid_kernel(
    torch::PackedTensorAccessor64<float,2> grid_acc,
    const int dim,
    const int64_t N,
    const int64_t prod_N)
{
    int64_t idx, i;
    int d;

    for (idx = blockIdx.x*blockDim.x + threadIdx.x; idx < prod_N; idx += gridDim.x*blockDim.x)
    {
        // idx has the form ((i[0]*N + i[1])*N + ...)*N + i[dim-1]
        // where i goes from 0 to N-1, which is shifted from k in [-N/2,...,N/2-1]
        i = idx;
        for (d=0; d<dim; ++d) {
            // i % N == i[dim-d-1] == k[dim-d-1] + N/2 == N*grid[idx]
            grid_acc[idx][dim-d-1] = ((i % N) / (float)N) - 0.5f;
            i /= N;
        }
    }
}

__global__ void
fill_radial_interpolation_grid_kernel(
    torch::PackedTensorAccessor64<float,1> grid_acc,
    const int dim,
    const int64_t N,
    const int64_t prod_N)
{
    int64_t idx, i;
    int d;
    float r;

    for (idx = blockIdx.x*blockDim.x + threadIdx.x; idx < prod_N; idx += gridDim.x*blockDim.x)
    {
        // idx has the form ((i[0]*N + i[1])*N + ...)*N + i[dim-1]
        // where i goes from 0 to N-1, which is shifted from l in [-N/2,...,N/2-1]
        i = idx;
        r = 0;
        for (d=0; d<dim; ++d) {
            // i % N == i[dim-d-1] == l[dim-d-1] + N/2
            r += (((i % N) / (float)N) - 0.5f) * (((i % N) / (float)N) - 0.5f);
            i /= N;
        }
        grid_acc[idx] = sqrtf(r);
    }
}


__global__ void
copy_complex_grid_kernel_values_kernel(
    const torch::PackedTensorAccessor64<c10::complex<float>,1> grid_values_acc,
    cufftComplex *b,
    const int dim,
    const int64_t N,
    const int64_t prod_N)
{
    int64_t reverse_idx, i, idx, b_idx;
    int d;
    c10::complex<float> value;

    for (reverse_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_idx < prod_N; reverse_idx += gridDim.x*blockDim.x)
    {
        i = reverse_idx;
        idx = 0;
        b_idx = 0;
        for (d=0; d<dim; ++d) {
            idx = N*idx + (i % N);
            b_idx = N*b_idx + ((i + N/2) % N);
            i /= N;
        }
        value = grid_values_acc[idx];
        b[b_idx] = make_cuFloatComplex(value.real(), value.imag());
    }
}

__global__ void
copy_real_grid_kernel_values_kernel(
    const torch::PackedTensorAccessor64<float,1> grid_values_acc,
    cufftComplex *b,
    const int dim,
    const int64_t N,
    const int64_t prod_N)
{
    int64_t reverse_idx, i, idx, b_idx;
    int d;

    for (reverse_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_idx < prod_N; reverse_idx += gridDim.x*blockDim.x)
    {
        i = reverse_idx;
        idx = 0;
        b_idx = 0;
        for (d=0; d<dim; ++d) {
            idx = N*idx + (i % N);
            b_idx = N*b_idx + ((i + N/2) % N);
            i /= N;
        }
        b[b_idx] = make_cuFloatComplex(grid_values_acc[idx], 0.0f);
    }
}


__global__ void
copy_interpolated_kernel_coeffs_kernel(
    torch::PackedTensorAccessor64<c10::complex<float>,1> coeffs_acc,
    cufftComplex *b,
    const int dim,
    const int64_t N,
    const int64_t prod_N)
{
    int64_t reverse_idx, i, idx, b_idx;
    int d;

    for (reverse_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_idx < prod_N; reverse_idx += gridDim.x*blockDim.x)
    {
        i = reverse_idx;
        idx = 0;
        b_idx = 0;
        for (d=0; d<dim; ++d) {
            idx = N*idx + (i % N);
            b_idx = N*b_idx + ((i + N/2) % N);
            i /= N;
        }
        coeffs_acc[idx] = c10::complex<float>(cuCrealf(b[b_idx]) / prod_N, cuCimagf(b[b_idx]) / prod_N);
    }
}
