#include <ATen/cuda/CUDAContext.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <complex.h>
#include <cufft.h>

// #define NFFT_PRINT_DEBUG

#include "cuda_utils.cu"
#include "spatial_window_operations.cu"
#include "spectral_window_operations.cu"
#include "kernel_coeffs.cu"


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
    printf("Batch size: %ld\n", batch_size);
    printf("M_array: %d %d %d\n", M_array[0], M_array[1], M_array[2]);
#endif


/// PREPARE SOURCES

    int *point_shifts;
    cudaMalloc(&point_shifts, num_sources_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_sources_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        point_shifts,
        2*N, m,
        dim, num_sources_total);

    CHECK_ERRORS();

    float *point_psi;
    cudaMalloc(&point_psi, num_sources_total*dim*(2*m+2)*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_sources_total, 2*m+2, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        point_shifts,
        point_psi,
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
            point_shifts,
            point_psi,
            g,
            dim, num_sources_total, num_columns,
            2*N, 2*m+2, window_volume);
    }
    else {
        complex_adjoint_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<c10::complex<float>,2>(),
            source_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_sources_total, num_columns,
            2*N, 2*m+2, window_volume);
    }

    CHECK_ERRORS();

/// EXECUTE FFT

#ifdef NFFT_PRINT_DEBUG
    if (dim == 2) {
        print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
        CHECK_ERRORS();
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

    CHECK_ERRORS();

    cufftDestroy(plan);

#ifdef NFFT_PRINT_DEBUG
    if (dim == 2) {
        print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
        CHECK_ERRORS();
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

    cudaFree(point_shifts);
    cudaFree(point_psi);
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
    printf("Batch size: %ld\n", batch_size);
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
        print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
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

    int *point_shifts;
    cudaMalloc(&point_shifts, num_targets_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_targets_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        targets.packed_accessor64<float,2>(),
        point_shifts,
        2*N, m,
        dim, num_targets_total);

    CHECK_ERRORS();

    float *point_psi;
    cudaMalloc(&point_psi, num_targets_total*dim*(2*m+2)*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_targets_total, 2*m+2, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        targets.packed_accessor64<float,2>(),
        point_shifts,
        point_psi,
        dim, num_targets_total, N, 2*m+2,
        WINDOW_FORWARD_PARAM1(N, m), WINDOW_FORWARD_PARAM2(N, m));

    CHECK_ERRORS();

/// PREPARE OUTPUT

    std::vector<int64_t> y_sizes(x.dim()-dim);
    y_sizes[0] = num_targets_total;
    for (int d=0; d<x.dim()-dim-1; ++d)
        y_sizes[1+d] = x.size(1+dim+d);


    auto y = torch::zeros(y_sizes, x.options().dtype(
        real_output ? torch::Dtype::Float : torch::Dtype::ComplexFloat));
    auto y_reshaped = y.view({num_targets_total, num_columns});


    setupGrid(&gridDim, &blockDim, num_targets_total, num_columns, window_volume, 32);
    if (real_output) {
        real_forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<float,2>(),
            target_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_targets_total, num_columns,
            2*N, 2*m+2, window_volume);
    }
    else {
        complex_forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<c10::complex<float>,2>(),
            target_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_targets_total, num_columns,
            2*N, 2*m+2, window_volume);
    }

    CHECK_ERRORS();

/// CLEANUP

    cudaFree(point_shifts);
    cudaFree(point_psi);
    cudaFree(g);
    cudaFree(phi_hat_inv);

    return y;
}



torch::Tensor
nfft_fastsum_cuda(
    const torch::Tensor sources,
    const torch::Tensor targets,
    const torch::Tensor x,
    const torch::Tensor coeffs,
    const torch::optional<torch::Tensor> opt_source_batch,
    const torch::optional<torch::Tensor> opt_target_batch,
    const int64_t N,
    const int64_t m)
{
    int dim;
    int64_t batch_size;
    int64_t num_sources_total;
    torch::Tensor source_batch;
    check_point_input(sources, opt_source_batch,
        &dim, &num_sources_total, &batch_size, &source_batch);

    int is_symmetric = sources.is_same(targets);
#ifdef NFFT_PRINT_DEBUG
    printf("is_symmetric = %d\n", is_symmetric);
#endif

    int target_dim;
    int64_t target_batch_size;
    int64_t num_targets_total;
    torch::Tensor target_batch;
    if (is_symmetric) {
        num_targets_total = num_sources_total;
        target_batch = source_batch;
    }
    else {
        check_point_input(targets, opt_target_batch,
            &target_dim, &num_targets_total, &target_batch_size, &target_batch);
        CHECK_INPUT(dim == target_dim);
        CHECK_INPUT(target_batch_size == batch_size);
    }

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

    CHECK_CUDA(coeffs);
    CHECK_INPUT(coeffs.dim() == dim);
    CHECK_INPUT(coeffs.numel() == prod_N);
    int real_coeffs = (coeffs.scalar_type() == at::ScalarType::Float);
    if (!real_coeffs)
        CHECK_INPUT(coeffs.scalar_type() == at::ScalarType::ComplexFloat);


#ifdef NFFT_PRINT_DEBUG
    printf("Point dimension: %d\n", dim);
    printf("Total number of source points: %ld\n", num_sources_total);
    printf("Total number of target points: %ld\n", num_targets_total);
    printf("Number of columns: %ld\n", num_columns);
    printf("Batch size: %ld\n", batch_size);
    printf("M_array: %d %d %d\n", M_array[0], M_array[1], M_array[2]);
#endif


/// PREPARE SOURCES
#ifdef NFFT_PRINT_DEBUG
    printf("Preparing sources\n");
#endif

    int *point_shifts;
    cudaMalloc(&point_shifts, num_sources_total*dim*sizeof(int));

    setupGrid(&gridDim, &blockDim, num_sources_total, dim);
    compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        point_shifts,
        2*N, m,
        dim, num_sources_total);
    CHECK_ERRORS();

    float *point_psi;
    cudaMalloc(&point_psi, num_sources_total*dim*(2*m+2)*sizeof(float));

    setupGrid(&gridDim, &blockDim, num_sources_total, 2*m+2, dim);
    compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
        sources.packed_accessor64<float,2>(),
        point_shifts,
        point_psi,
        dim, num_sources_total, N, 2*m+2,
        WINDOW_FORWARD_PARAM1(N, m), WINDOW_FORWARD_PARAM2(N, m));
    CHECK_ERRORS();

/// COMPUTE g    (convolution with window function)

#ifdef NFFT_PRINT_DEBUG
    printf("Computing g\n");
#endif

    cufftComplex *g;
    cudaMalloc(&g, batch_size*num_columns*prod_M*sizeof(cufftComplex));
    cudaMemset(g, 0, batch_size*num_columns*prod_M*sizeof(cufftComplex));

    const torch::Tensor x_reshaped = x.view({num_sources_total, num_columns});

    setupGrid(&gridDim, &blockDim, num_sources_total, num_columns, window_volume, 32);
    if (real_input) {
        real_adjoint_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<float,2>(),
            source_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_sources_total, num_columns,
            2*N, 2*m+2, window_volume);
    }
    else {
        complex_adjoint_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            x_reshaped.packed_accessor64<c10::complex<float>,2>(),
            source_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_sources_total, num_columns,
            2*N, 2*m+2, window_volume);
    }

    CHECK_ERRORS();

/// EXECUTE ADJOINT FFT
#ifdef NFFT_PRINT_DEBUG
    printf("Executing adjoint FFT\n");
#endif

// #ifdef NFFT_PRINT_DEBUG
//     if (dim == 2) {
//         print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
//         CHECK_ERRORS();
//     }
// #endif

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

    CHECK_ERRORS();

// #ifdef NFFT_PRINT_DEBUG
//     if (dim == 2) {
//         print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
//         CHECK_ERRORS();
//     }
// #endif

/// PREPARE PHI_HAT
#ifdef NFFT_PRINT_DEBUG
    printf("Preparing phi_hat\n");
#endif

    float *phi_hat_inv;
    cudaMalloc(&phi_hat_inv, (N/2+1)*sizeof(float));

    setupGrid(&gridDim, &blockDim, N/2+1);
    compute_phi_hat_inv_kernel<<<gridDim, blockDim, 0, stream>>>(
        phi_hat_inv, N/2, WINDOW_ADJOINT_PARAM(N, m));

    CHECK_ERRORS();

/// CONVOLUTION WITH THE KERNEL
#ifdef NFFT_PRINT_DEBUG
    printf("Convolution with the kernel\n");
#endif

    const torch::Tensor coeffs_reshaped = coeffs.view({prod_N});

    setupGrid(&gridDim, &blockDim, num_targets_total, num_columns, window_volume, 32);
    if (real_coeffs) {
        real_kernel_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            coeffs_reshaped.packed_accessor64<float,1>(),
            g,
            phi_hat_inv,
            dim, batch_size, num_columns,
            2*N, N/2, prod_M);
    }
    else {
        complex_kernel_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            coeffs_reshaped.packed_accessor64<c10::complex<float>,1>(),
            g,
            phi_hat_inv,
            dim, batch_size, num_columns,
            2*N, N/2, prod_M);
    }


    /// EXECUTE FORWARD FFT
#ifdef NFFT_PRINT_DEBUG
    printf("Executing forward FFT\n");
#endif

// #ifdef NFFT_PRINT_DEBUG
//     if (dim == 2) {
//         print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
//         CHECK_ERRORS();
//     }
// #endif

    AT_ASSERTM(cufftExecC2C(plan, g, g, CUFFT_FORWARD)
                == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    CHECK_ERRORS();

    cufftDestroy(plan);

// #ifdef NFFT_PRINT_DEBUG
//     if (dim == 2) {
//         print_g_slice_2d_kernel<<<1,1,0,stream>>>(g, 2*N);
//         CHECK_ERRORS();
//     }
// #endif

    /// PREPARE TARGETS

    if (!is_symmetric) {

#ifdef NFFT_PRINT_DEBUG
        printf("Preparing targets\n");
#endif

        cudaFree(&point_shifts);
        cudaFree(&point_psi);

        cudaMalloc(&point_shifts, num_targets_total*dim*sizeof(int));

        setupGrid(&gridDim, &blockDim, num_targets_total, dim);
        compute_shifts_kernel<<<gridDim, blockDim, 0, stream>>>(
            targets.packed_accessor64<float,2>(),
            point_shifts,
            2*N, m,
            dim, num_targets_total);

        CHECK_ERRORS();

        cudaMalloc(&point_psi, num_targets_total*dim*(2*m+2)*sizeof(float));

        setupGrid(&gridDim, &blockDim, num_targets_total, 2*m+2, dim);
        compute_psi_kernel<<<gridDim, blockDim, 0, stream>>>(
            targets.packed_accessor64<float,2>(),
            point_shifts,
            point_psi,
            dim, num_targets_total, N, 2*m+2,
            WINDOW_FORWARD_PARAM1(N, m), WINDOW_FORWARD_PARAM2(N, m));

        CHECK_ERRORS();
    }

/// PREPARE OUTPUT
#ifdef NFFT_PRINT_DEBUG
    printf("Preparing outputs\n");
#endif

    std::vector<int64_t> y_sizes(x.sizes().vec());
    y_sizes[0] = num_targets_total;

    auto y = torch::zeros(y_sizes, x.options());
    auto y_reshaped = y.view({num_targets_total, num_columns});

    setupGrid(&gridDim, &blockDim, num_targets_total, num_columns, window_volume, 32);
    if (real_input) {
        real_forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<float,2>(),
            target_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_targets_total, num_columns,
            2*N, 2*m+2, window_volume);
    }
    else {
        complex_forward_window_convolution_kernel<<<gridDim, blockDim, 0, stream>>>(
            y_reshaped.packed_accessor64<c10::complex<float>,2>(),
            target_batch.packed_accessor64<int64_t,1>(),
            point_shifts,
            point_psi,
            g,
            dim, num_targets_total, num_columns,
            2*N, 2*m+2, window_volume);
    }

    CHECK_ERRORS();

/// CLEANUP

    cudaFree(point_shifts);
    cudaFree(point_psi);
    cudaFree(g);
    cudaFree(phi_hat_inv);

    return y;
}


torch::Tensor
gaussian_analytical_coeffs_cuda(
    const double sigma,
    const int64_t N,
    const int64_t dim)
{
    int64_t prod_N = N;
    for (int d=1; d<dim; ++d)
        prod_N *= N;

    std::vector<int64_t> coeffs_sizes = std::vector<int64_t>(dim, N);
    torch::Tensor coeffs = torch::zeros(coeffs_sizes,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor coeffs_reshaped = coeffs.view({prod_N});

    dim3 gridDim, blockDim;
    setupGrid(&gridDim, &blockDim, prod_N);

    fill_gaussian_analytical_coeffs_kernel<<<gridDim, blockDim>>>(
        coeffs_reshaped.packed_accessor64<float,1>(),
        (float)sigma, dim, N, prod_N);
    CHECK_ERRORS();

    return coeffs;
}


torch::Tensor
gaussian_interpolated_coeffs_cuda(
    const double sigma,
    const int64_t N,
    const int64_t dim,
    const int64_t p,
    const double eps)
{
    AT_ASSERTM(p <= 0, "Gaussian interpolated coeffs are currently only implemented for p<=0");
    AT_ASSERTM(eps == 0.0, "Gaussian interpolated coeffs are currently only implemented for eps=0");

    int N_array[3] = {(int)N,(int)N,(int)N};
    int64_t prod_N = N;
    for (int d=1; d<dim; ++d)
        prod_N *= N;

    cufftComplex *b;
    cudaMalloc(&b, prod_N*sizeof(cufftComplex));

    dim3 gridDim, blockDim;
    setupGrid(&gridDim, &blockDim, prod_N);

    fill_gaussian_regularized_values_kernel<<<gridDim, blockDim>>>(
        b, (float)(sigma*sigma), dim, N, prod_N, p, (float)eps);
    CHECK_ERRORS();

    cufftHandle plan;
    AT_ASSERTM(cufftPlanMany(&plan, dim,
                N_array,                        // shape of the transform (n)
                N_array,                        // shape of the real input data (inembed)
                1,                              // stride of the real input data (istride)
                prod_N,                         // distance between consecutive input batch signals (idist)
                N_array,                        // shape of the complex output data (onembed)
                1,                              // stride of the complex output data (ostride)
                prod_N,                         // distance between consecutive output batch signals (odist)
                CUFFT_C2C,                      // transform type
                1)                              // total number of signals
            == CUFFT_SUCCESS, "Failed to create CUFFT plan");

    AT_ASSERTM(cufftExecC2C(plan, b, b, CUFFT_FORWARD)
            == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    CHECK_ERRORS();
    cufftDestroy(plan);

    std::vector<int64_t> coeffs_sizes = std::vector<int64_t>(dim, N);
    torch::Tensor coeffs = torch::zeros(coeffs_sizes,
        torch::TensorOptions().dtype(at::ScalarType::ComplexFloat).device(torch::kCUDA));
    torch::Tensor coeffs_reshaped = coeffs.view(prod_N);

    copy_interpolated_kernel_coeffs_kernel<<<gridDim, blockDim>>>(
        coeffs_reshaped.packed_accessor64<c10::complex<float>, 1>(),
        b,
        dim, N, prod_N);
    CHECK_ERRORS();

    cudaFree(b);
    return coeffs;
}


torch::Tensor
interpolation_grid_cuda(
    const int64_t N,
    const int64_t dim)
{
    int64_t prod_N = N;
    for (int d=1; d<dim; ++d)
        prod_N *= N;

    std::vector<int64_t> grid_sizes = std::vector<int64_t>(dim+1, N);
    grid_sizes[dim] = dim;
    torch::Tensor grid = torch::zeros(grid_sizes,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor grid_reshaped = grid.view({prod_N, dim});

    dim3 gridDim, blockDim;
    setupGrid(&gridDim, &blockDim, prod_N);

    fill_interpolation_grid_kernel<<<gridDim, blockDim>>>(
        grid_reshaped.packed_accessor64<float,2>(),
        dim, N, prod_N);
    CHECK_ERRORS();

    return grid;
}

torch::Tensor
radial_interpolation_grid_cuda(
    const int64_t N,
    const int64_t dim)
{
    int64_t prod_N = N;
    for (int d=1; d<dim; ++d)
        prod_N *= N;

    std::vector<int64_t> grid_sizes = std::vector<int64_t>(dim, N);
    torch::Tensor grid = torch::zeros(grid_sizes,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor grid_reshaped = grid.view({prod_N});

    dim3 gridDim, blockDim;
    setupGrid(&gridDim, &blockDim, prod_N);

    fill_radial_interpolation_grid_kernel<<<gridDim, blockDim>>>(
        grid_reshaped.packed_accessor64<float,1>(),
        dim, N, prod_N);
    CHECK_ERRORS();

    return grid;
}


torch::Tensor
interpolated_kernel_coeffs_cuda(
    const torch::Tensor grid_values)
{
    CHECK_CUDA(grid_values);
    int dim = grid_values.dim();
    CHECK_INPUT(dim >= 1 && dim <= 3);
    int N = grid_values.size(0);

    int N_array[3] = {N,N,N};
    int64_t prod_N = N;
    for (int d=1; d<dim; ++d) {
        CHECK_INPUT(grid_values.size(d) == N);
        prod_N *= N;
    }

    cufftComplex *b;
    cudaMalloc(&b, prod_N*sizeof(cufftComplex));

    dim3 gridDim, blockDim;
    setupGrid(&gridDim, &blockDim, prod_N);

    const torch::Tensor grid_values_reshaped = grid_values.view(prod_N);

    if (grid_values.scalar_type() == at::ScalarType::Float) {
        copy_real_grid_kernel_values_kernel<<<gridDim, blockDim>>>(
            grid_values_reshaped.packed_accessor64<float,1>(),
            b, dim, N, prod_N);
    }
    else {
        CHECK_INPUT(grid_values.scalar_type() == at::ScalarType::ComplexFloat);
        copy_complex_grid_kernel_values_kernel<<<gridDim, blockDim>>>(
            grid_values_reshaped.packed_accessor64<c10::complex<float>,1>(),
            b, dim, N, prod_N);
    }
    CHECK_ERRORS();


    cufftHandle plan;
    AT_ASSERTM(cufftPlanMany(&plan, dim,
                N_array,                        // shape of the transform (n)
                N_array,                        // shape of the real input data (inembed)
                1,                              // stride of the real input data (istride)
                prod_N,                         // distance between consecutive input batch signals (idist)
                N_array,                        // shape of the complex output data (onembed)
                1,                              // stride of the complex output data (ostride)
                prod_N,                         // distance between consecutive output batch signals (odist)
                CUFFT_C2C,                      // transform type
                1)                              // total number of signals
            == CUFFT_SUCCESS, "Failed to create CUFFT plan");

    AT_ASSERTM(cufftExecC2C(plan, b, b, CUFFT_FORWARD)
            == CUFFT_SUCCESS, "Failed to execute CUFFT plan");

    CHECK_ERRORS();
    cufftDestroy(plan);

    torch::Tensor coeffs = torch::zeros(grid_values.sizes(),
                                        grid_values.options().dtype(at::ScalarType::ComplexFloat));
    torch::Tensor coeffs_reshaped = coeffs.view(prod_N);

    copy_interpolated_kernel_coeffs_kernel<<<gridDim, blockDim>>>(
        coeffs_reshaped.packed_accessor64<c10::complex<float>, 1>(),
        b,
        dim, N, prod_N);
    CHECK_ERRORS();

    cudaFree(b);
    return coeffs;
}
