#include <ATen/cuda/CUDAContext.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <complex.h>
#include <cufft.h>

#define THREADS 1024

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
    Auxiliary function to setup two dim3 variables to be used in kernel calls.

    nx, ny, nz: desired number of threads in each dimension. If possible, the
      resulting grid will satisfy gridDim.x*blockDim.x >= nx (etc.). This can
      not always be achieved because the maximum gridDim is 65535 in each
      dimension.
    ty, tz: number of threads per block in y and z dimension. Because the
      maximum number of threads per block in all dimensions combined is 1024,
      the used number of threads in x dimension is simply set to 1024/(ty*tz).
*/
__host__ void
setupGrid(dim3 *gridDim, dim3 *blockDim,
        int64_t nx, int64_t ny=1, int64_t nz=1,
        int64_t ty=1, int64_t tz=1) {
    blockDim->z = tz <= 64 ? tz : 64;
    blockDim->y = (ty*blockDim->z > 1024) ? (1024 / blockDim->z) : ty;
    blockDim->x = 1024 / (blockDim->y * blockDim->z);
    gridDim->x = (nx + blockDim->x - 1) / blockDim->x;
    gridDim->y = (ny + blockDim->y - 1) / blockDim->y;
    gridDim->z = (nz + blockDim->z - 1) / blockDim->z;
    if (gridDim->x > 65535) gridDim->x = 65535;
    if (gridDim->y > 65535) gridDim->y = 65535;
    if (gridDim->z > 65535) gridDim->z = 65535;
}



#define PI                  3.141592653589793115997963468544185161590576171875f
#define THREE_QUARTER_PI    2.356194490192344928846982537459627163147877049531f
#define PI_THIRD            1.047197551196597746154214461093167628065723133125f


// The actual window function phi_act(x) is defined as
//   phi_act(x) = exp(-M^2/b * x^2) / sqrt(pi * b)
// with b = (2 sigma m) / ((2 sigma - 1) pi), where sigma is the oversampling
// rate. We assume that M is large enough that phi_act(x) vanishes outside
// of [-1/2,1/2], so the series-based periodicization does not change the
// function inside this interval.
// We do not evaluate phi_act directly but with a scaled argument on [-N, N],
//   phi(x) = phi_act(x/M) = exp(-x^2 / b) / sqrt(pi * b)
//          = exp(-x^2 * inv_b) * inv_sqrt_b_pi
// with inv_b = 1 / b = pi * (2 sigma - 1) / (2 sigma m)
//      inv_sqrt_b_pi = sqrt((2 sigma - 1) / (2 sigma m)).
// In our case with sigma=2:
//      b = (4 * m) / (3 * pi)
//      inv_b = 0.75 * pi / m
//      inv_sqrt_b_pi = sqrt(0.75 / m)
__device__ __forceinline__ float
eval_phi(const float x, const float window_inv_b, const float window_inv_sqrt_b_pi)
{
    return expf(-(x*x)*window_inv_b) * window_inv_sqrt_b_pi;
}

// The Fourier coefficients of the actual window function phi_act are
//   phi_act_hat(k) = 1/M * exp(-b pi^2 / M^2 k^2).
// Because we need to take the inverse and multiply with M later on, we
// instead evaluate
//   phi_hat_inv(k) = 1 / (M phi_hat_act(k)) = exp(b_square_pi_over_M * k^2)
// with b_square_pi_over_M = b * (pi / M)^2.
// In our case with oversampling rate 2:
//   b_square_pi_over_M = (4 * m * pi) / (3 * M^2) = (m * pi) / (3 * N^2)
__device__ __forceinline__ float
eval_phi_hat_inv(const int64_t k, const float window_b_square_pi_over_M)
{
    return expf(float(k*k) * window_b_square_pi_over_M);
}


// Fill array point_shifts such that
//   point_shifts[point_idx*dim + d] = floor(point_acc[point_idx][d]*M - m).
// For a fixed point_idx and dimension d, this is the smallest element of the
// index set I_Mm.
// Because the point coordinates are supposed to be in [-1/2, 1/2), the shifts
// are guaranteed to be in [-N-m,...,N-m-1].
__global__ void
compute_shifts_kernel(
    const torch::PackedTensorAccessor64<float, 2> point_acc,
    int *point_shifts,
    const int M, const int m,
    const int64_t dim, const int64_t num_points)
{
    int64_t point_idx, d;
    for (point_idx = blockDim.x*blockIdx.x + threadIdx.x; point_idx < num_points; point_idx += gridDim.x*blockDim.x)
    {
        for (d=blockDim.y*blockIdx.y + threadIdx.y; d < dim; d += gridDim.y*blockDim.y)
        {
            point_shifts[point_idx*dim + d] = (int)floorf(point_acc[point_idx][d]*M) - m;

            if (point_idx == 0)
                printf(" - point coordinate x[%ld,%ld] = %f:  mid frequency %d, shift %d, I_Mm = {%d, ..., %d}\n",
                    point_idx, d, point_acc[point_idx][d],
                    point_shifts[point_idx*dim + d] + m, point_shifts[point_idx*dim + d],
                    point_shifts[point_idx*dim + d], point_shifts[point_idx*dim + d] + (int)(2*m+1));
        }
    }
}

// Fill array point_psi such that
//   point_psi[(point_idx*dim + d)*window_length + window_idx]
//     = phi(M*x[d] - floor(M*x[d]) - l) = phi_act(x[d] - (floor(M*x[d]) + l)/M)
//     = phi(M*x[d] - point_shifts[point_idx*dim + d] - window_idx)
//   for all l in [-m,...,m] shifted to window_idx = l + m in [0,...,2*m]
__global__ void
compute_psi_kernel(
    const torch::PackedTensorAccessor64<float, 2> point_acc,
    const int *point_shifts,
    float *psi,
    const int64_t dim, const int64_t num_points,
    const int64_t N, const int64_t window_length,
    const float window_inv_b, const float window_inv_sqrt_b_pi)
{
    int64_t point_idx, window_idx, d;
    for (point_idx = blockDim.x*blockIdx.x + threadIdx.x; point_idx < num_points; point_idx += gridDim.x*blockDim.x)
    {
        for (window_idx = blockDim.y*blockIdx.y + threadIdx.y; window_idx < window_length; window_idx += gridDim.y*blockDim.y)
        {
            for (d=blockDim.z*blockIdx.z + threadIdx.z; d < dim; d += gridDim.z*blockDim.z)
            {
                psi[(point_idx*dim + d)*window_length + window_idx] =
                    eval_phi(point_acc[point_idx][d]*2.0*N - point_shifts[point_idx*dim + d] - window_idx,
                        window_inv_b, window_inv_sqrt_b_pi);
                if (point_idx == 0)
                    printf(" - psi for point pos[%ld,%ld] at  l=%ld: %f\n",
                        point_idx, d, window_idx - (window_length-2)/2,
                        psi[(point_idx*dim + d)*window_length + window_idx]);
            }
        }
    }
}

// Fill array phi_hat_inv with the inverse N-periodic Fourier coefficients of the Gaussian window function,
//   phi_hat_inv[freq_idx] = 1/(M*phi_act_hat(freq_idx))
// for all freq_idx in [0,...,N/2].
// Because of symmetricity, we can later obtain the remaining freq_idx in [N/2+1,...,N-1] via
//   1/phi_hat(freq_idx) = 1/phi_hat(-freq_idx) = phi_hat_inv[N-freq_idx].
__global__ void
compute_phi_hat_inv_kernel(
    float* phi_hat_inv,
    const int64_t N,
    const float window_b_square_pi_over_M)
{
    int64_t freq_idx;

    for (freq_idx=blockDim.x*blockIdx.x + threadIdx.x; freq_idx < N; freq_idx += gridDim.x*blockDim.x)
    {
        phi_hat_inv[freq_idx] = eval_phi_hat_inv(freq_idx, window_b_square_pi_over_M);
        printf(" - phi_hat_inv at frequency %ld:  %f\n", freq_idx, phi_hat_inv[freq_idx]);
    }
}

// Fill array g such that the sum of the impacts of all points in batch #batch_idx
// for column #column_idx on the frequency multiindex i={i[0],...,i[dim-1]} in [0,...,M-1]^d
// is stored in
//   g[ (batch_idx*num_columns + column_idx)*signal_dist + ((i[0]*M + i[1])*M + ...)*M + i[dim-1] ]
__global__ void
adjoint_window_convolution_kernel(
    const torch::PackedTensorAccessor64<float,2> x_acc,
    const torch::PackedTensorAccessor64<int64_t,1> batch_acc,
    const int *point_shifts,
    const float *point_psi,
    cufftReal *g,
    const int64_t dim, const int64_t num_points, const int64_t num_columns,
    const int64_t M, const int64_t window_length, const int64_t window_volume,
    const int64_t signal_dist)
{
    int64_t point_idx, batch_idx, column_idx, window_idx, w, freq_idx, d;
    float value;

    for (point_idx=blockDim.x*blockIdx.x + threadIdx.x; point_idx < num_points; point_idx += gridDim.x*blockDim.x)
    {
        batch_idx = batch_acc[point_idx];
        for (column_idx=blockDim.y*blockIdx.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (window_idx=blockDim.z*blockIdx.z + threadIdx.z; window_idx < window_volume; window_idx += gridDim.z*blockDim.z)
            {
                value = x_acc[point_idx][column_idx];

                // window_idx: multiindex l={l[0],...,l[dim-1]} with 0 <= l[d] < window_length = 2*m+2,
                // shifted from original shifts (l[d] - m) in [-m,...,m+1]
                // stored in reverse order compared to normal:
                //   window_idx = ((l[dim-1]*window_length + l[dim-2])*window_length + ...)*window_length + l[0]
                // so we can iteratively obtain the current component l[d] = window_idx % window_length
                // and then shave it off via window_idx /= window_length.
                // However, we need to copy window_idx to a new variable w because of the outer for loop.
                w = window_idx;

                // freq_idx: multiindex i={i[0],...,i[dim-1]} in [0,...,M-1]^dim
                // build freq_idx iteratively in normal order: (needs to be this way for cuFFT)
                //   freq_idx = ((i[0]*M + i[1])*M + ...)*M + i[dim-1]
                //   with i[d] = floor(M*x[d]) + l[d] - m  from I_Mm = [floor(M*x[d])-m, ..., ceil(M*x[d])+m] (assuming ceil() = floor()+1)
                //   cropped and continued periodically to [0,...,M-1].
                // We have point_shifts[point_idx*dim + d] = floor(M*x[d]) - m
                //   -->  point_shifts[point_idx*dim + d] + l[d] is in [-N-m,...,N-1+m] and can be cropped modulo M

                // Layout of g: value for batch #batch_idx, column #column_idx, frequency multiindex i = {i[0],...,i[dim-1]}:
                // If signal_dist == M^dim, the signal component would be stored at
                //     g[ (batch_idx*num_columns + column_idx)*M^dim + freq_idx ] =
                //     g[ ((((batch_idx*num_columns + column_idx)*M + i[0])*M + i[1])*M + ...)*M + i[dim-1] ]
                // Unfortunately, signal_dist is not M^dim because the complex output data needs more space,
                // so we cannot already include the first part in freq_idx.
                freq_idx = 0;

                for (d=0; d<dim; ++d) {
                    // thanks to reverse storage: l[d] = w % window_length
                    freq_idx = M*freq_idx + ((point_shifts[point_idx*dim + d] + (w % window_length) + M) % M);
                    value *= point_psi[(point_idx*dim + d)*window_length + (w % window_length)];
                    // shave off smallest dimension l[d]
                    w /= window_length;
                }

                atomicAdd(g + ((batch_idx*num_columns + column_idx)*signal_dist + freq_idx), (cufftReal)value);

                if (point_idx == 0 && column_idx == 0)
                    printf(" - Contribution of x[%ld, %ld] = %f to frequency %ld = [%ld, %ld, %ld]: %f\n",
                        point_idx, column_idx, x_acc[point_idx][column_idx], freq_idx,
                        (point_shifts[point_idx*dim + 0] + (window_idx % window_length) + M) % M,
                        dim > 1 ? (point_shifts[point_idx*dim + 1] + ((window_idx / window_length) % window_length) + M) % M : 0,
                        dim > 2 ? (point_shifts[point_idx*dim + 2] + (window_idx / (window_length * window_length)) + M) % M : 0,
                        value);
            }
        }
    }
}


// Fill tensor y_hat such that for all batch_idx, column_idx, and
//   freq_idx = ((i[0]*N + i[1])*N + ...)*(N/2+1) + i[dim-1]
//   (where all i[d] are in [0,...,N-1] except i[dim-1] in [0,...,N/2])
// the output is stored in y[batch_idx][freq_idx][column_idx],
// which is essentially an entry of g_hat divided by the product
//   phi_hat(i[0]) * ... * phi_hat(i[dim-1]).
// Because g_hat has twice the number of entries, we need to shift the indices
// to find the correct entry at
//   g_hat[(batch_idx*num_columns + column_idx)*half_prod_M + shifted_freq_idx]
//   with   shifted_freq_idx = ((j[0]*M + i[1])*M + ...)*(N+1) + j[dim-1]
//   where  j[d] = i[d] + (i[d] <= N/2 ? 0 : N)
__global__ void
adjoint_rolloff_correction_kernel(
    torch::PackedTensorAccessor64<c10::complex<float>,3> y_acc, // size batch_size x (halfN+1)*N*...*N x num_columns
    const cufftComplex *g_hat, // size batch_size * num_columns * (N+1)*2N*...*2N
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t N, const int64_t halfN, const int64_t half_prod_N)
{
    int64_t batch_idx, freq_idx, reverse_freq_idx, f, column_idx, g_hat_idx, d;
    float factor;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < half_prod_N; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                // we actually iterate over the frequency indices in reverse order:
                //   reverse_freq_idx = ((i[dim-1]*N + i[dim-2])*N + ...)*N + i[0]
                // so we can obtain the current i[d] as f % N if we always shave off f /= N in each iteration.
                f = reverse_freq_idx;
                // But for the index in y, we need to also build the original freq_idx
                freq_idx = 0;
                // Because g_hat is contiguous, the correct index can be computed iteratively in one go:
                //   g_hat_idx = ((((batch_idx*num_columns + column_idx)*M + i[0])*M + i[0])*M + ...)*(N+1) * i[dim-1])
                g_hat_idx = batch_idx*num_columns + column_idx;
                for (d=0; d<dim-1; ++d)
                {
                    freq_idx = freq_idx*N + (f % N);
                    g_hat_idx = g_hat_idx*2*N + (f % N);
                    factor *= phi_hat_inv[f % N];
                    if (f % N > halfN)
                        // second half: j[d] = i[d]+N
                        g_hat_idx += N;
                    f /= N;
                }

                // Last dimension is different because i[dim-1] only goes to N/2
                freq_idx = (halfN + 1)*freq_idx + f;
                g_hat_idx = (N + 1)*g_hat_idx + f;
                factor *= phi_hat_inv[f];

                y_acc[batch_idx][freq_idx][column_idx] =
                    c10::complex<float>(cuCrealf(g_hat[g_hat_idx])*factor, cuCimagf(g_hat[g_hat_idx])*factor);

                if (batch_idx == 0 && column_idx == 0)
                    printf(" - Output in frequency %ld = [%ld, %ld, %ld], oversampled frequency %ld = [%ld, %ld, %ld]:  g_hat=%f + %fi, factor=%f\n",
                            freq_idx, reverse_freq_idx % N, (reverse_freq_idx / N) % N, reverse_freq_idx / (N * N),
                            g_hat_idx, g_hat_idx / (2*N*(N+1)), (g_hat_idx / (N+1)) % (2*N), g_hat_idx % (N+1),
                            cuCrealf(g_hat[g_hat_idx]), cuCimagf(g_hat[g_hat_idx]), factor);
            }
        }
    }
}

// for debugging purposes only
__global__ void
print_g_slice_2d_kernel(
    const float* g,
    const int64_t M)
{
    int64_t f;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (f=0; f < M*M; ++f) {
            printf(" - g[0,0,%ld,%ld,0] = %f\n",
                f / M, f % M, g[f]);
        }
    }
}

// for debugging purposes only
__global__ void
print_g_hat_slice_2d_kernel(
    const cufftComplex* g_hat,
    const int64_t N)
{
    int64_t f;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (f=0; f < 2*N*(N+1); ++f) {
            printf(" - g_hat[0,0,%ld,%ld,0] = %f + %fi\n",
                f / (N+1), f % (N+1), cuCrealf(g_hat[f]), cuCimagf(g_hat[f]));
        }
    }
}


/**
    Main function for Adjoint NFFT on GPU
    See csrc/core.cpp for documentation of parameters
*/
torch::Tensor
nfft_adjoint_cuda(
    const torch::Tensor sources,
    const torch::Tensor x,
    const torch::optional<torch::Tensor> opt_source_batch,
    const int64_t N,
    const int64_t m)
{

    CHECK_CUDA(sources);
    CHECK_CUDA(x);
    CHECK_INPUT(x.scalar_type() == at::ScalarType::Float);
    cudaSetDevice(x.get_device());

    CHECK_INPUT(sources.dim() == 2);
    CHECK_INPUT(sources.scalar_type() == at::ScalarType::Float);

    int dim = sources.size(1);
    printf("Point dimension: %d\n", dim);
    CHECK_INPUT(dim >= 1 && dim <= 3);

    int64_t num_sources_total = sources.size(0);
    printf("Total number of source points: %ld\n", num_sources_total);

    int batch_size = 1;
    torch::Tensor source_batch;
    if (opt_source_batch.has_value()) {
        source_batch = opt_source_batch.value();
        CHECK_CUDA(source_batch);
        CHECK_INPUT(source_batch.dim() == 1);
        CHECK_INPUT(source_batch.scalar_type() == at::ScalarType::Long);
        batch_size = source_batch.index({-1}).item().to<int>() + 1;
    } else {
        source_batch = torch::zeros({num_sources_total}, sources.options().dtype(torch::kLong));
    }
    printf("Batch size: %d\n", batch_size);

    // Check size of x
    CHECK_INPUT(x.dim() >= 1);
    CHECK_INPUT(x.size(0) == num_sources_total);
    int64_t num_columns = x.numel() / num_sources_total;
    printf("Number of columns: %ld\n", num_columns);

    // Prepare frequency domain size parameters
    int M_array[3] = {1,1,1};
    int prod_N = 1;
    int64_t window_length = 2*m+2;
    int64_t window_volume = 1;
    for (int d=0; d<dim; ++d) {
        M_array[d] = 2*N;
        prod_N *= N;
        window_volume *= window_length;
    }
    printf("M_array: %d %d %d\n", M_array[0], M_array[1], M_array[2]);

    int complex_size_array[3] = {1,1,1};
    int64_t half_prod_M = N + 1;
    complex_size_array[dim-1] = N + 1;
    for (int d=0; d<dim-1; ++d) {
        complex_size_array[d] = 2*N;
        half_prod_M *= 2*N;
    }
    int64_t half_prod_N = (prod_N / N) * (N/2 + 1);


    // Prepare parameters for Gaussian window function
    const float window_inv_b = THREE_QUARTER_PI / m;
    const float window_inv_sqrt_b_pi = sqrtf(0.75f / m);
    const float window_b_square_pi_over_M = PI_THIRD * m / (N*N);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 gridDim, blockDim;

/// PREPARE SOURCES

    int *source_shifts;
    cudaMalloc(&source_shifts, num_sources_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_sources_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        source_shifts,
        2*(int)N, (int)m,
        dim, num_sources_total);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    float *source_psi;
    cudaMalloc(&source_psi, num_sources_total*dim*window_volume*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_sources_total, window_length, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        source_shifts,
        source_psi,
        dim, num_sources_total, N, window_length,
        window_inv_b, window_inv_sqrt_b_pi);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

/// COMPUTE g    (convolution with window function)

    // real input data:
    //   g[(batch_idx*num_columns + column_idx)*2*half_prod_M + freq_idx]
    //   with freq_idx = ((i[0]*M + i[1])*M + ...)*M + i[dim-1]
    //   where all i[d] are in [0,...,M-1]
    //   (zero padding between the signals because the maximum freq_idx is smaller than 2*half_prod_M - 1)

    // complex output data:
    //   g_hat[(batch_idx*num_columns + column_idx)*half_prod_M + freq_idx]
    //   with freq_idx = ((i[0]*M + i[1])*M + ...)*(N+1) + i[dim-1]
    //   where all i[d] are in [0,...,M-1] except i[dim-1] in [0,...,N]
    //   (no zero padding between the signals because the maximum freq_idx is half_prod_M - 1)

    // An alternative would be to have only zero padding at the end of the real input data,
    // so we don't need to pass signal_dist to the window convolution,
    // which would only require idist == M^dim for cuFFT

    cufftReal *g;
    cudaMalloc(&g, batch_size*num_columns*half_prod_M*sizeof(cufftComplex));
    cudaMemset(g, 0, batch_size*num_columns*half_prod_M*sizeof(cufftComplex));
    cufftComplex *g_hat = (cufftComplex*)g;

    const torch::Tensor x_reshaped = x.view({num_sources_total, num_columns});

    setupGrid(&gridDim, &blockDim, num_sources_total, num_columns, window_volume, 32);
    adjoint_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
        x_reshaped.packed_accessor64<float,2>(),
        source_batch.packed_accessor64<int64_t,1>(),
        source_shifts,
        source_psi,
        g,
        dim, num_sources_total, num_columns,
        2*N, window_length, window_volume,
        2*half_prod_N);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

/// EXECUTE FFT

    // if (dim == 2) {
    //     print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
    //     gpuErrchk( cudaPeekAtLastError() );
    //     gpuErrchk( cudaDeviceSynchronize() );
    // }

    cufftHandle plan;
    AT_ASSERTM(cufftPlanMany(&plan, dim,
                M_array,                        // shape of the transform (n)
                M_array,                        // shape of the real input data (inembed)
                1,                              // stride of the real input data (istride)
                2*half_prod_M,                  // distance between consecutive input batch signals (idist)
                complex_size_array,             // shape of the complex output data (onembed)
                1,                              // stride of the complex output data (ostride)
                half_prod_M,                    // distance between consecutive output batch signals (odist)
                CUFFT_R2C,                      // transform type
                batch_size*num_columns)         // total number of signals
            == CUFFT_SUCCESS, "Failed to create CUFFT plan");

    // cufftExecR2C is always "implicitely forward", which is exactly what we need.
    // If we wanted the adjoint, we needed to fill g in reverse order to emulate
    //   g_hat[l] = \sum_{k=0}^{M-1} g[k] exp(2j pi k l / M)
    //            = \sum_{k=0}^{M-1} g[k] exp(-2j pi (M - k) l / M)
    //            = \sum_{k=0}^{M-1} g[k == 0 ? 0 : M-k] exp(-2j pi k l / M)
    AT_ASSERTM(cufftExecR2C(plan, g, g_hat)
                == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cufftDestroy(plan);

    // if (dim == 2) {
    //     print_g_hat_slice_2d_kernel<<<1,1,0,stream>>>(g_hat, N);
    //     gpuErrchk( cudaPeekAtLastError() );
    //     gpuErrchk( cudaDeviceSynchronize() );
    // }

/// PREPARE PHI_HAT

    float *phi_hat_inv;
    cudaMalloc(&phi_hat_inv, N*sizeof(float));

    setupGrid(&gridDim, &blockDim, N);
    compute_phi_hat_inv_kernel<<<gridDim, blockDim, 0, stream>>>(
        phi_hat_inv, N, window_b_square_pi_over_M);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

/// PREPARE OUTPUT

    // Prepare shape of y:
    //   - batch_size
    //   - N, repeated (dim-1) times
    //   - N/2 + 1 for the final spatial dimension
    //   - trailing dimensions of x (factors of num_columns)
    std::vector<int64_t> y_sizes(x.dim()+dim);
    y_sizes[0] = batch_size;
    for (int d=1; d<dim; ++d)
        y_sizes[d] = N;
    y_sizes[dim] = N/2 + 1;
    for (int d=0; d<x.dim()-1; ++d)
        y_sizes[d+dim+1] = x.size(d+1);


    auto y = torch::zeros(y_sizes, x.options().dtype(torch::Dtype::ComplexFloat));
    auto y_reshaped = y.view({batch_size, half_prod_N, num_columns});

    setupGrid(&gridDim, &blockDim, half_prod_N, num_columns, batch_size, 32);
    adjoint_rolloff_correction_kernel<<<gridDim, blockDim, 0, stream>>>(
        y_reshaped.packed_accessor64<c10::complex<float>,3>(),
        g_hat,
        phi_hat_inv,
        dim, batch_size, num_columns,
        N, N/2, half_prod_N);

/// CLEANUP

    cudaFree(source_shifts);
    cudaFree(source_psi);
    cudaFree(g);
    cudaFree(phi_hat_inv);

    return y;
}




torch::Tensor
nfft_forward_cuda(
    torch::Tensor targets,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_target_ptr,
    double tol)
{
    return x;
}
