#include <ATen/cuda/CUDAContext.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <complex.h>
#include <cufft.h>

#include "cuda_utils.cu"
#include "window_operations.cu"
#include "adjoint_window_operations.cu"

#define PI 3.141592653589793115997963468544185161590576171875f




#ifdef NFFT_PRINT_DEBUG

// for debugging purposes only
__global__ void
print_g_slice_2d_kernel(
    const cufftComplex* g,
    const int64_t M)
{
    int64_t f;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (f=0; f < M*M; ++f) {
            printf(" - g[0,0,%ld,%ld,0] = %f + %fi\n",
                f / M, f % M, cuCrealf(g[f]), cuCimagf(g[f]));
        }
    }
}

// for debugging purposes only
__global__ void
print_g_hat_slice_2d_kernel(
    const cufftComplex* g_hat,
    const int64_t M)
{
    int64_t f;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (f=0; f < M*M; ++f) {
            printf(" - g_hat[0,0,%ld,%ld,0] = %f + %fi\n",
                f / M, f % M, cuCrealf(g_hat[f]), cuCimagf(g_hat[f]));
        }
    }
}

#endif // NFFT_PRINT_DEBUG


__host__ void
check_point_input(
    const torch::Tensor pos,
    const torch::optional<torch::Tensor> optional_batch,
    int *out_dim,
    int64_t *out_num_points,
    int64_t *out_batch_size,
    torch::Tensor *out_batch)
{
    CHECK_CUDA(pos);
    CHECK_INPUT(pos.dim() == 2);
    CHECK_INPUT(pos.scalar_type() == at::ScalarType::Float);

    *out_num_points = pos.size(0);
    *out_dim = pos.size(1);
    CHECK_INPUT(*out_dim >= 1 && *out_dim <= 3);

    if (optional_batch.has_value()) {
        *out_batch = optional_batch.value();
        CHECK_CUDA((*out_batch));
        CHECK_INPUT(out_batch->dim() == 1);
        CHECK_INPUT(out_batch->scalar_type() == at::ScalarType::Long);
        *out_batch_size = out_batch->index({-1}).item().to<int64_t>() + 1;
    }
    else {
        *out_batch = torch::zeros({*out_num_points}, pos.options().dtype(torch::kLong));
        *out_batch_size = 1;
    }
}


__host__ void
check_spatial_coeffs_input(
    const torch::Tensor x,
    const int64_t num_points,
    int *out_real_input,
    int64_t *out_num_columns)
{
    CHECK_CUDA(x);

    *out_real_input = (x.scalar_type() == at::ScalarType::Float);
    if (!(*out_real_input))
        CHECK_INPUT(x.scalar_type() == at::ScalarType::ComplexFloat);

    CHECK_INPUT(x.dim() >= 1);
    CHECK_INPUT(x.size(0) == num_points);
    *out_num_columns = x.numel() / num_points;

}


__host__ void
check_spectral_coeffs_input(
    const torch::Tensor x,
    const int dim,
    const int64_t batch_size,
    int *out_real_input,
    int64_t *out_N,
    int64_t *out_num_columns)
{
    CHECK_CUDA(x);

    *out_real_input = (x.scalar_type() == at::ScalarType::Float);
    if (!(*out_real_input))
        CHECK_INPUT(x.scalar_type() == at::ScalarType::ComplexFloat);

    CHECK_INPUT(x.dim() >= dim + 1);
    CHECK_INPUT(x.size(0) == batch_size);

    *out_N = x.size(1);
    CHECK_INPUT(*out_N >= 2);

    *out_num_columns = x.numel() / (batch_size * (*out_N));
    for (int d=2; d<dim+1; ++d) {
        CHECK_INPUT(x.size(d) == *out_N);
        *out_num_columns /= *out_N;
    }
}


__host__ void
setup_spectral_dimensions(
    int64_t N,
    int64_t m,
    int dim,
    int *out_M_array,
    int64_t *out_prod_N,
    int64_t *out_prod_M,
    int64_t *out_window_volume)
{
    *out_prod_M = 1;
    *out_prod_N = 1;
    *out_window_volume = 1;
    for (int d=0; d<dim; ++d) {
        out_M_array[d] = 2*N;
        *out_prod_M *= 2*N;
        *out_prod_N *= N;
        *out_window_volume *= 2*m+2;
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
    const int64_t m,
    const int64_t real_output)
{
    int dim;
    int64_t batch_size;
    int64_t num_sources_total;
    torch::Tensor source_batch;
    check_point_input(sources, opt_source_batch,
        &dim, &num_sources_total, &batch_size, &source_batch);

    int real_input;
    int64_t num_columns;
    check_spatial_coeffs_input(x, num_sources_total,
        &real_input, &num_columns);

    cudaSetDevice(x.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 gridDim, blockDim;

    int M_array[3];
    int64_t prod_N;
    int64_t prod_M;
    int64_t window_volume;
    setup_spectral_dimensions(N, m, dim,
        M_array, &prod_N, &prod_M, &window_volume);


#ifdef NFFT_PRINT_DEBUG
    printf("Point dimension: %d\n", dim);
    printf("Total number of source points: %ld\n", num_sources_total);
    printf("Number of columns: %ld\n", num_columns);
    printf("Batch size: %d\n", batch_size);
    printf("M_array: %d %d %d\n", M_array[0], M_array[1], M_array[2]);
#endif


/// PREPARE SOURCES

    int *source_shifts;
    cudaMalloc(&source_shifts, num_sources_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_sources_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        source_shifts,
        2*N, m,
        dim, num_sources_total);

    CHECK_ERRORS();

    float *source_psi;
    cudaMalloc(&source_psi, num_sources_total*dim*(2*m+2)*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_sources_total, 2*m+2, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        source_shifts,
        source_psi,
        dim, num_sources_total, N, 2*m+2,
        WINDOW_FORWARD_PARAM1(N, m), WINDOW_FORWARD_PARAM2(N, m));

    CHECK_ERRORS();

/// COMPUTE g    (convolution with window function)

    cufftComplex *g;
    cudaMalloc(&g, batch_size*num_columns*prod_M*sizeof(cufftComplex));
    cudaMemset(g, 0, batch_size*num_columns*prod_M*sizeof(cufftComplex));

    const torch::Tensor x_reshaped = x.view({num_sources_total, num_columns});

    setupGrid(&gridDim, &blockDim, num_sources_total, num_columns, window_volume, 32);
    if (real_input) {
        real_adjoint_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<float,2>(),
            source_batch.packed_accessor64<int64_t,1>(),
            source_shifts,
            source_psi,
            g,
            dim, num_sources_total, num_columns,
            2*N, 2*m+2, window_volume);
    }
    else {
        complex_adjoint_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<c10::complex<float>,2>(),
            source_batch.packed_accessor64<int64_t,1>(),
            source_shifts,
            source_psi,
            g,
            dim, num_sources_total, num_columns,
            2*N, 2*m+2, window_volume);
    }

    CHECK_ERRORS();

/// EXECUTE FFT

#ifdef NFFT_PRINT_DEBUG
    if (dim == 2) {
        print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
#endif

    cufftHandle plan;
    AT_ASSERTM(cufftPlanMany(&plan, dim,
                M_array,                        // shape of the transform (n)
                M_array,                        // shape of the real input data (inembed)
                1,                              // stride of the real input data (istride)
                prod_M,                         // distance between consecutive input batch signals (idist)
                M_array,                        // shape of the complex output data (onembed)
                1,                              // stride of the complex output data (ostride)
                prod_M,                         // distance between consecutive output batch signals (odist)
                CUFFT_C2C,                      // transform type
                batch_size*num_columns)         // total number of signals
            == CUFFT_SUCCESS, "Failed to create CUFFT plan");

    AT_ASSERTM(cufftExecC2C(plan, g, g, CUFFT_INVERSE)
                == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cufftDestroy(plan);

#ifdef NFFT_PRINT_DEBUG
    if (dim == 2) {
        print_g_hat_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
#endif

/// PREPARE PHI_HAT

    float *phi_hat_inv;
    cudaMalloc(&phi_hat_inv, (N/2+1)*sizeof(float));

    setupGrid(&gridDim, &blockDim, N/2+1);
    compute_phi_hat_inv_kernel<<<gridDim, blockDim, 0, stream>>>(
        phi_hat_inv, N/2, WINDOW_ADJOINT_PARAM(N, m));

    CHECK_ERRORS();

/// PREPARE OUTPUT

    // Prepare shape of y:
    //   - batch_size
    //   - N, repeated dim times
    //   - trailing dimensions of x (factors of num_columns)
    std::vector<int64_t> y_sizes(x.dim()+dim);
    y_sizes[0] = batch_size;
    for (int d=0; d<dim; ++d)
        y_sizes[1+d] = N;
    for (int d=0; d<x.dim()-1; ++d)
        y_sizes[1+dim+d] = x.size(d+1);


    auto y = torch::zeros(y_sizes, x.options().dtype(
        real_output ? torch::Dtype::Float : torch::Dtype::ComplexFloat));
    auto y_reshaped = y.view({batch_size, prod_N, num_columns});

    setupGrid(&gridDim, &blockDim, prod_N, num_columns, batch_size, 32);
    if (real_output) {
        real_adjoint_rolloff_correction_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<float,3>(),
            g,
            phi_hat_inv,
            dim, batch_size, num_columns,
            N, N/2, prod_N);
    }
    else {
        complex_adjoint_rolloff_correction_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<c10::complex<float>,3>(),
            g,
            phi_hat_inv,
            dim, batch_size, num_columns,
            N, N/2, prod_N);
     }

/// CLEANUP

    cudaFree(source_shifts);
    cudaFree(source_psi);
    cudaFree(g);
    cudaFree(phi_hat_inv);

    return y;
}


/*

__global__ void
forward_rolloff_correction_kernel(
    const torch::PackedTensorAccessor64<c10::complex<float>,3> x_acc, // size batch_size x (halfN+1)*N*...*N x num_columns
    cufftComplex *g_hat, // size batch_size * num_columns * (N+1)*2N*...*2N
    const float *phi_hat_inv, // size N
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
                // so we can obtain the current i[d] = f % N if we always shave off f /= N in each iteration.
                f = reverse_freq_idx;
                // But for the index in x, we need to also build the original
                //   freq_idx = ((i[0]*N + i[1])*N + ...)*N + i[dim-1]
                freq_idx = 0;
                // Because g_hat is contiguous, the correct index can be computed iteratively in one go:
                //   g_hat_idx = ((((batch_idx*num_columns + column_idx)*M + j[0])*M + j[0])*M + ...)*(N+1) * j[dim-1])
                g_hat_idx = batch_idx*num_columns + column_idx;
                for (d=0; d<dim-1; ++d)
                {
                    freq_idx = freq_idx*N + (f % N);
                    g_hat_idx = g_hat_idx*2*N + (f % N);
                    if (f % N >= halfN) {
                        // second half: j[d] = i[d]+N
                        g_hat_idx += N;
                        // the positive frequency i[d] actually refers to the negative frequency i[d]-N,
                        // which can be retrieved via symmetry: phi_hat(i[d]-N) = phi_hat(N-i[d])
                        factor *= phi_hat_inv[N - (f % N)]
                    }
                    else {
                        factor *= phi_hat_inv[f % N];
                    }
                    f /= N;
                }

                // Last dimension is different because i[dim-1] only goes to N/2
                // Even if f == N/2, the corresponding g_hat_idx is not increased by N
                freq_idx = (halfN + 1)*freq_idx + f;
                g_hat_idx = (N + 1)*g_hat_idx + f;
                factor *= phi_hat_inv[f];

                g_hat[g_hat_idx] = make_cuFloatComplex(x_acc[batch_idx][freq_idx][column_idx].real * factor,
                                                        x_acc[batch_idx][freq_idx][column_idx].imag * factor);

                if (batch_idx == 0 && column_idx == 0)
                    printf(" - g_hat in oversampled frequency [%ld, %ld, %ld], original frequency [%ld, %ld, %ld]:  g_hat=%f + %fi, factor=%f\n",
                            g_hat_idx / (2*N*(N+1)), (g_hat_idx / (N+1)) % (2*N), g_hat_idx % (N+1),
                            reverse_freq_idx % N, (reverse_freq_idx / N) % N, reverse_freq_idx / (N * N),
                            cuCrealf(g_hat[g_hat_idx]), cuCimagf(g_hat[g_hat_idx]), factor);
            }
        }
    }
}



__global__ void
forward_window_convolution_kernel(
    torch::PackedTensorAccessor64<float,2> y_acc,
    const torch::PackedTensorAccessor64<int64_t,1> batch_acc,
    const int *point_shifts,
    const float *point_psi,
    const cufftReal *g,
    const int64_t dim, const int64_t num_points, const int64_t num_columns,
    const int64_t M, const int64_t 2*m+2, const int64_t window_volume,
    const int64_t signal_dist)
{
    int64_t point_idx, batch_idx, column_idx, window_idx, w, freq_idx, d;
    float factor;

    for (point_idx=blockDim.x*blockIdx.x + threadIdx.x; point_idx < num_points; point_idx += gridDim.x*blockDim.x)
    {
        batch_idx = batch_acc[point_idx];
        for (column_idx=blockDim.y*blockIdx.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (window_idx=blockDim.z*blockIdx.z + threadIdx.z; window_idx < window_volume; window_idx += gridDim.z*blockDim.z)
            {
                factor = 1.0f;

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
                    factor *= point_psi[(point_idx*dim + d)*window_length + (w % window_length)];
                    // shave off smallest dimension l[d]
                    w /= window_length;
                }

                atomicAdd(&(y_acc[point_idx][column_idx]),
                            factor*g[(batch_idx*num_columns + column_idx)*signal_dist + freq_idx])
                //atomicAdd(g + ((batch_idx*num_columns + column_idx)*signal_dist + freq_idx), (cufftReal)value);

                //if (point_idx == 0 && column_idx == 0)
                if (freq_idx == 0)
                    printf(" - Contribution of g[%ld, %ld, %ld, %ld, %ld]=%f to output y[%ld,%ld]: %f\n",
                        batch_idx, column_idx,
                        (point_shifts[point_idx*dim + 0] + (window_idx % window_length) + M) % M,
                        dim > 1 ? (point_shifts[point_idx*dim + 1] + ((window_idx / window_length) % window_length) + M) % M : 0,
                        dim > 2 ? (point_shifts[point_idx*dim + 2] + (window_idx / (window_length * window_length)) + M) % M : 0,
                        g[(batch_idx*num_columns + column_idx)*signal_dist + freq_idx],
                        point_idx, column_idx, x_acc[point_idx][column_idx], freq_idx,
                        value);
            }
        }
    }
}



torch::Tensor
nfft_forward_cuda(
    torch::Tensor targets,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_target_batch,
    int m)
{
    CHECK_CUDA(targets);
    CHECK_CUDA(x);
    CHECK_INPUT(x.scalar_type() == at::ScalarType::Float);
    cudaSetDevice(x.get_device());

    CHECK_INPUT(targets.dim() == 2);
    CHECK_INPUT(targets.scalar_type() == at::ScalarType::ComplexFloat);

    int dim = sources.size(1);
    printf("Point dimension: %d\n", dim);
    CHECK_INPUT(dim >= 1 && dim <= 3);

    int64_t num_targets_total = targets.size(0);
    printf("Total number of target points: %ld\n", num_targets_total);

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
    CHECK_INPUT(x.dim() >= 1 + dim);
    CHECK_INPUT(x.size(0) == batch_size);
    int N = 2*(x.size(dim)-1);
    int64_t num_columns = x.numel() / (batch_size * (N/2 + 1));
    for (int d=1; d<dim; ++d) {
        CHECK_INPUT(x.size(d) == N);
        num_columns /= N;
    }
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

/// PREPARE PHI_HAT

    float *phi_hat_inv;
    cudaMalloc(&phi_hat_inv, N*sizeof(float));

    setupGrid(&gridDim, &blockDim, N);
    compute_phi_hat_inv_kernel<<<gridDim, blockDim, 0, stream>>>(
        phi_hat_inv, N, window_b_square_pi_over_M);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


/// COPY INPUT TO G_HAT
    cufftReal *g;
    cudaMalloc(&g, batch_size*num_columns*half_prod_M*sizeof(cufftComplex));
    cudaMemset(g, 0, batch_size*num_columns*half_prod_M*sizeof(cufftComplex));
    cufftComplex *g_hat = (cufftComplex*)g;

    auto x_reshaped = x.view({batch_size, half_prod_N, num_columns});

    setupGrid(&gridDim, &blockDim, half_prod_N, num_columns, batch_size, 32);
    forward_rolloff_correction_kernel<<<gridDim, blockDim, 0, stream>>>(
        x_reshaped.packed_accessor64<c10::complex<float>,3>(),
        g_hat,
        phi_hat_inv,
        dim, batch_size, num_columns,
        N, N/2, half_prod_N);


/// EXECUTE FFT

    if (dim == 2) {
        print_g_hat_slice_2d_kernel<<<1,1,0,stream>>>(g_hat, N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    cufftHandle plan;
    AT_ASSERTM(cufftPlanMany(&plan, dim,
                M_array,                        // shape of the transform (n)
                complex_size_array,             // shape of the complex output data (onembed)
                1,                              // stride of the complex output data (ostride)
                half_prod_M,                    // distance between consecutive output batch signals (odist)
                M_array,                        // shape of the real input data (inembed)
                1,                              // stride of the real input data (istride)
                2*half_prod_M,                  // distance between consecutive input batch signals (idist)
                CUFFT_C2R,                      // transform type
                batch_size*num_columns)         // total number of signals
            == CUFFT_SUCCESS, "Failed to create CUFFT plan");

    // cufftExecC2R is always "implicitely adjoint", which is exactly what we need.
    AT_ASSERTM(cufftExecC2R(plan, g_hat, g)
                == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cufftDestroy(plan);

    if (dim == 2) {
        print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

/// PREPARE TARGETS

    int *target_shifts;
    cudaMalloc(&target_shifts, num_targets_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_targets_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        targets.packed_accessor64<float,2>(),
        target_shifts,
        2*(int)N, (int)m,
        dim, num_targets_total);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    float *target_psi;
    cudaMalloc(&target_psi, num_targets_total*dim*window_length*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_targets_total, window_length, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        targets.packed_accessor64<float,2>(),
        target_shifts,
        target_psi,
        dim, num_targets_total, N, window_length,
        window_inv_b, window_inv_sqrt_b_pi);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

/// PREPARE OUTPUT


    setupGrid(&gridDim, &blockDim, num_sources_total, num_columns, window_volume, 32);
    forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
        x_reshaped.packed_accessor64<float,2>(),
        source_batch.packed_accessor64<int64_t,1>(),
        source_shifts,
        source_psi,
        g,
        dim, num_sources_total, num_columns,
        2*N, window_length, window_volume,
        2*half_prod_M);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

/// CLEANUP

    cudaFree(source_shifts);
    cudaFree(source_psi);
    cudaFree(g);
    cudaFree(phi_hat_inv);

    return y;
}
*/

torch::Tensor
nfft_forward_cuda(
    torch::Tensor pos,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_batch,
    int64_t m,
    int64_t real_output)
{
    return x;
}
