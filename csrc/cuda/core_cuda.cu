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



torch::Tensor
nfft_forward_cuda(
    torch::Tensor targets,
    torch::Tensor x,
    torch::optional<torch::Tensor> opt_target_batch,
    int64_t m,
    int64_t real_output)
{
    int dim;
    int64_t batch_size;
    int64_t num_targets_total;
    torch::Tensor target_batch;
    check_point_input(targets, opt_target_batch,
        &dim, &num_targets_total, &batch_size, &target_batch);

    int real_input;
    int64_t N;
    int64_t num_columns;
    check_spectral_coeffs_input(x, dim, batch_size,
        &real_input, &N, &num_columns);

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
    printf("Total number of target points: %ld\n", num_targets_total);
    printf("Number of columns: %ld\n", num_columns);
    printf("Batch size: %d\n", batch_size);
    printf("M_array: %d %d %d\n", M_array[0], M_array[1], M_array[2]);
#endif




/// PREPARE PHI_HAT

    float *phi_hat_inv;
    cudaMalloc(&phi_hat_inv, (N/2+1)*sizeof(float));

    setupGrid(&gridDim, &blockDim, N/2+1);
    compute_phi_hat_inv_kernel<<<gridDim, blockDim, 0, stream>>>(
        phi_hat_inv, N/2, WINDOW_ADJOINT_PARAM(N, m));

    CHECK_ERRORS();


/// COPY INPUT TO G
    cufftComplex *g;
    cudaMalloc(&g, batch_size*num_columns*prod_M*sizeof(cufftComplex));
    cudaMemset(g, 0, batch_size*num_columns*prod_M*sizeof(cufftComplex));

    auto x_reshaped = x.view({batch_size, prod_N, num_columns});

    setupGrid(&gridDim, &blockDim, prod_N, num_columns, batch_size, 32);
    if (real_input) {
        real_forward_rolloff_correction_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<float,3>(),
            g,
            phi_hat_inv,
            dim, batch_size, num_columns,
            N, N/2, prod_N);
    }
    else {
        complex_forward_rolloff_correction_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<c10::complex<float>,3>(),
            g,
            phi_hat_inv,
            dim, batch_size, num_columns,
            N, N/2, prod_N);
    }
    CHECK_ERRORS();


/// EXECUTE FFT

#ifdef NFFT_PRINT_DEBUG
    if (dim == 2) {
        print_g_hat_slice_2d_kernel<<<1,1,0,stream>>>(g_hat, N);
        CHECK_ERRORS();
    }
#endif

    cufftHandle plan;
    AT_ASSERTM(cufftPlanMany(&plan, dim,
                M_array,                        // shape of the transform (n)
                M_array,             // shape of the complex output data (onembed)
                1,                              // stride of the complex output data (ostride)
                prod_M,                    // distance between consecutive output batch signals (odist)
                M_array,                        // shape of the complex input data (inembed)
                1,                              // stride of the complex input data (istride)
                prod_M,                  // distance between consecutive input batch signals (idist)
                CUFFT_C2C,                      // transform type
                batch_size*num_columns)         // total number of signals
            == CUFFT_SUCCESS, "Failed to create CUFFT plan");

    // cufftExecC2R is always "implicitely adjoint", which is exactly what we need.
    AT_ASSERTM(cufftExecC2C(plan, g, g, CUFFT_FORWARD)
                == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    CHECK_ERRORS();

    cufftDestroy(plan);

#ifdef NFFT_PRINT_DEBUG
    if (dim == 2) {
        print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
        CHECK_ERRORS();
    }
#endif

/// PREPARE TARGETS

    int *target_shifts;
    cudaMalloc(&target_shifts, num_targets_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_targets_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        targets.packed_accessor64<float,2>(),
        target_shifts,
        2*N, m,
        dim, num_targets_total);

    CHECK_ERRORS();

    float *target_psi;
    cudaMalloc(&target_psi, num_targets_total*dim*(2*m+2)*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_targets_total, 2*m+2, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        targets.packed_accessor64<float,2>(),
        target_shifts,
        target_psi,
        dim, num_targets_total, N, 2*m+2,
        WINDOW_FORWARD_PARAM1(N, m), WINDOW_FORWARD_PARAM2(N, m));

    CHECK_ERRORS();

/// PREPARE OUTPUT

    std::vector<int64_t> y_sizes(x.dim()-dim);
    y_sizes[0] = num_targets_total;
    for (int d=0; d<x.dim()-dim-1; ++d)
        y_sizes[1+dim+d] = x.size(d+1);


    auto y = torch::zeros(y_sizes, x.options().dtype(
        real_output ? torch::Dtype::Float : torch::Dtype::ComplexFloat));
    auto y_reshaped = y.view({num_targets_total, num_columns});


    setupGrid(&gridDim, &blockDim, num_targets_total, num_columns, window_volume, 32);
    if (real_output) {
        real_forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<float,2>(),
            target_batch.packed_accessor64<int64_t,1>(),
            target_shifts,
            target_psi,
            g,
            dim, num_targets_total, num_columns,
            2*N, 2*m+2, window_volume);
    }
    else {
        complex_forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<c10::complex<float>,2>(),
            target_batch.packed_accessor64<int64_t,1>(),
            target_shifts,
            target_psi,
            g,
            dim, num_targets_total, num_columns,
            2*N, 2*m+2, window_volume);
    }

    CHECK_ERRORS();

/// CLEANUP

    cudaFree(target_shifts);
    cudaFree(target_psi);
    cudaFree(g);
    cudaFree(phi_hat_inv);

    return y;
}



/*




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

// torch::Tensor
// nfft_forward_cuda(
//     torch::Tensor pos,
//     torch::Tensor x,
//     torch::optional<torch::Tensor> opt_batch,
//     int64_t m,
//     int64_t real_output)
// {
//     return x;
// }
