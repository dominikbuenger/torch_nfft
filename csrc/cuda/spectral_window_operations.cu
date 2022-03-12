

#define PI_THIRD 1.047197551196597746154214461093167628065723133125f
#define WINDOW_ADJOINT_PARAM(N,m) (PI_THIRD * m / (N*N))


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



// Fill array phi_hat_inv with the inverse N-periodic Fourier coefficients of the Gaussian window function,
//   phi_hat_inv[freq_idx] = 1/(M*phi_act_hat(freq_idx))
// for all freq_idx in [0,...,N/2].
// Because of symmetry, we can later obtain the remaining freq_idx in [-N/2,...,1] via
//   1/phi_hat(freq_idx) = 1/phi_hat(-freq_idx) = phi_hat_inv[-freq_idx].
__global__ void
compute_phi_hat_inv_kernel(
    float* phi_hat_inv,
    const int64_t halfN,
    const float window_b_square_pi_over_M)
{
    int64_t freq_idx;

    for (freq_idx=blockDim.x*blockIdx.x + threadIdx.x; freq_idx <= halfN; freq_idx += gridDim.x*blockDim.x)
    {
        phi_hat_inv[freq_idx] = eval_phi_hat_inv(freq_idx, window_b_square_pi_over_M);

#ifdef NFFT_PRINT_DEBUG
        printf(" - phi_hat_inv at frequency %ld:  %f\n", freq_idx, phi_hat_inv[freq_idx]);
#endif
    }
}



// Fill tensor y such that for all batch_idx, column_idx, and
//   freq_idx = ((i[0]*N + i[1])*N + ...)*N + i[dim-1],
//   (referring to the actual frequency multiindex l with l[d] = i[d] - N/2)
// the output is stored in y[batch_idx][freq_idx][column_idx]
__global__ void
complex_adjoint_rolloff_correction_kernel(
    torch::PackedTensorAccessor64<c10::complex<float>,3> y_acc, // size batch_size x N^dim x num_columns
    const cufftComplex *g_hat, // size batch_size * num_columns * M^dim
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t N, const int64_t halfN, const int64_t prod_N)
{
    int64_t batch_idx, freq_idx, reverse_freq_idx, f, column_idx, g_hat_idx, d;
    float factor;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < prod_N; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                // we actually iterate over the frequency indices in reverse order:
                //   reverse_freq_idx = ((i[dim-1]*N + i[dim-2])*N + ...)*N + i[0]
                // so we can obtain the current i[d] as f % N if we always shave off f /= N in each iteration.
                f = reverse_freq_idx;
                // But for the index in y, we need to also build the original freq_idx
                freq_idx = 0;
                // The correct index can be computed iteratively:
                //   g_hat_idx = ((((batch_idx*num_columns + column_idx)*M + i[0])*M + i[0])*M + ...)*(N+1) * i[dim-1])
                g_hat_idx = batch_idx*num_columns + column_idx;
                for (d=0; d<dim; ++d)
                {
                    freq_idx = freq_idx*N + (f % N);
                    if (f % N < halfN) {
                        // first half: actual frequency (f % N) - halfN is negative
                        // g_hat value is stored in the end part
                        g_hat_idx = g_hat_idx*2*N + 2*N + (f % N) - halfN;
                        // phi_hat value is stored at absolute value of actual frequency
                        factor *= phi_hat_inv[halfN - (f % N)];
                    }
                    else {
                        // second half: actual frequency (f % N) - halfN is non-negative
                        // g_hat value is stored in the first part
                        g_hat_idx = g_hat_idx*2*N + (f % N) - halfN;
                        // phi_hat value is stored at actual frequency
                        factor *= phi_hat_inv[(f % N) - halfN];
                    }
                    f /= N;
                }

                y_acc[batch_idx][freq_idx][column_idx] =
                    c10::complex<float>(cuCrealf(g_hat[g_hat_idx])*factor, cuCimagf(g_hat[g_hat_idx])*factor);

#ifdef NFFT_PRINT_DEBUG
                if (batch_idx == 0 && column_idx == 0)
                    printf(" - Output at freq index %ld = [%ld, %ld, %ld], oversampled frequency %ld = [%ld, %ld, %ld]:  g_hat=%f + %fi, factor=%f\n",
                            freq_idx, reverse_freq_idx % N, (reverse_freq_idx / N) % N, reverse_freq_idx / (N * N),
                            g_hat_idx, g_hat_idx / (4*N*N), (g_hat_idx / (2*N)) % (2*N), g_hat_idx % (2*N),
                            cuCrealf(g_hat[g_hat_idx]), cuCimagf(g_hat[g_hat_idx]), factor);
#endif
            }
        }
    }
}

// real version of complex_adjoint_rolloff_correction_kernel
__global__ void
real_adjoint_rolloff_correction_kernel(
    torch::PackedTensorAccessor64<float,3> y_acc, // size batch_size x N^dim x num_columns
    const cufftComplex *g_hat, // size batch_size * num_columns * M^dim
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t N, const int64_t halfN, const int64_t prod_N)
{
    int64_t batch_idx, freq_idx, reverse_freq_idx, f, column_idx, g_hat_idx, d;
    float factor;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < prod_N; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                f = reverse_freq_idx;
                freq_idx = 0;
                g_hat_idx = batch_idx*num_columns + column_idx;
                for (d=0; d<dim; ++d)
                {
                    freq_idx = freq_idx*N + (f % N);
                    if (f % N < halfN) {
                        g_hat_idx = g_hat_idx*2*N + 2*N + (f % N) - halfN;
                        factor *= phi_hat_inv[halfN - (f % N)];
                    }
                    else {
                        g_hat_idx = g_hat_idx*2*N + (f % N) - halfN;
                        factor *= phi_hat_inv[(f % N) - halfN];
                    }
                    f /= N;
                }

                y_acc[batch_idx][freq_idx][column_idx] = cuCrealf(g_hat[g_hat_idx])*factor;
            }
        }
    }
}




__global__ void
complex_forward_rolloff_correction_kernel(
    const torch::PackedTensorAccessor64<c10::complex<float>,3> x_acc, // size batch_size x prod_N x num_columns
    cufftComplex *g_hat, // size batch_size * num_columns * prod_M
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t N, const int64_t halfN, const int64_t prod_N)
{
    int64_t batch_idx, freq_idx, reverse_freq_idx, f, column_idx, g_hat_idx, d;
    float factor;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < prod_N; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                // we actually iterate over the frequency indices in reverse order:
                //   reverse_freq_idx = ((i[dim-1]*N + i[dim-2])*N + ...)*N + i[0]
                // so we can obtain the current i[d] as f % N if we always shave off f /= N in each iteration.
                // Here i[d] in [0,N-1] corresponds to the frequency i[d]-N/2 in [-N/2,N/2-1].
                f = reverse_freq_idx;
                // But for the index in x, we need to also build the original
                //   freq_idx = ((i[0]*N + i[1])*N + ...)*N + i[dim-1]
                freq_idx = 0;
                // The correct index can be computed iteratively:
                //   g_hat_idx = ((((batch_idx*num_columns + column_idx)*M + i[0])*M + i[0])*M + ...)*(N+1) * i[dim-1])
                g_hat_idx = batch_idx*num_columns + column_idx;
                for (d=0; d<dim; ++d)
                {
                    freq_idx = freq_idx*N + (f % N);
                    if (f % N < halfN) {
                        // first half: actual frequency (f % N) - halfN is negative
                        // g_hat value is stored in the end part
                        g_hat_idx = g_hat_idx*2*N + 2*N + (f % N) - halfN;
                        // phi_hat value is stored at absolute value of actual frequency
                        factor *= phi_hat_inv[halfN - (f % N)];
                    }
                    else {
                        // second half: actual frequency (f % N) - halfN is non-negative
                        // g_hat value is stored in the first part
                        g_hat_idx = g_hat_idx*2*N + (f % N) - halfN;
                        // phi_hat value is stored at actual frequency
                        factor *= phi_hat_inv[(f % N) - halfN];
                    }
                    f /= N;
                }


                g_hat[g_hat_idx] = make_cuFloatComplex(x_acc[batch_idx][freq_idx][column_idx].real() * factor,
                                                        x_acc[batch_idx][freq_idx][column_idx].imag() * factor);

#ifdef NFFT_PRINT_DEBUG
                if (batch_idx == 0 && column_idx == 0)
                    printf(" - g_hat in oversampled frequency [%ld, %ld, %ld], original frequency [%ld, %ld, %ld]:  g_hat=%f + %fi, factor=%f\n",
                            g_hat_idx / (2*N*(N+1)), (g_hat_idx / (N+1)) % (2*N), g_hat_idx % (N+1),
                            reverse_freq_idx % N, (reverse_freq_idx / N) % N, reverse_freq_idx / (N * N),
                            cuCrealf(g_hat[g_hat_idx]), cuCimagf(g_hat[g_hat_idx]), factor);
#endif
            }
        }
    }
}


// Variant of complex_forward_rolloff_correction_kernel for real-valued tensor x
__global__ void
real_forward_rolloff_correction_kernel(
    const torch::PackedTensorAccessor64<float,3> x_acc, // size batch_size x prod_N x num_columns
    cufftComplex *g_hat, // size batch_size * num_columns * prod_M
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t N, const int64_t halfN, const int64_t prod_N)
{
    int64_t batch_idx, freq_idx, reverse_freq_idx, f, column_idx, g_hat_idx, d;
    float factor;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < prod_N; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                f = reverse_freq_idx;
                freq_idx = 0;
                g_hat_idx = batch_idx*num_columns + column_idx;
                for (d=0; d<dim; ++d)
                {
                    freq_idx = freq_idx*N + (f % N);
                    if (f % N < halfN) {
                        g_hat_idx = g_hat_idx*2*N + 2*N + (f % N) - halfN;
                        factor *= phi_hat_inv[halfN - (f % N)];
                    }
                    else {
                        g_hat_idx = g_hat_idx*2*N + (f % N) - halfN;
                        factor *= phi_hat_inv[(f % N) - halfN];
                    }
                    f /= N;
                }


                g_hat[g_hat_idx] = make_cuFloatComplex(x_acc[batch_idx][freq_idx][column_idx] * factor, 0.0f);
            }
        }
    }
}



__global__ void
complex_kernel_convolution_kernel(
    const torch::PackedTensorAccessor64<c10::complex<float>,1> coeffs_acc,
    cufftComplex *g_hat, // size batch_size * num_columns * prod_M
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t M, const int64_t halfN, const int64_t prod_M)
{
    int64_t batch_idx, column_idx, reverse_freq_idx, coeff_idx, g_hat_idx, f, d;
    float factor;
    c10::complex<float> coeff;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < prod_M; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                coeff_idx = 0;
                g_hat_idx = batch_idx*num_columns + column_idx;

                for (d=0; d<dim; ++d) {
                    // a coeff_idx below zero indicates that this part of g_hat
                    // is set to zero and there is no fitting coefficient
                    if (coeff_idx >= 0) {
                        if (f % M < halfN) {
                            // First quarter of g_hat: refers to positive actual
                            // frequency (f % M) in [0,...,halfN-1]
                            // coefficient is stored in second half of coeffs
                            coeff_idx = 2*halfN*coeff_idx + halfN + (f % M);
                            // phi_hat value is stored in the actual frequency index
                            factor *= phi_hat_inv[f % M];
                        }
                        else if (f % M >= 3*halfN) {
                            // Last quarter of g_hat: refers to negative actual
                            // frequency (f % M) - M in [-halfN,...,-1]
                            // coefficient stored in first half of coeffs
                            coeff_idx = 2*halfN*coeff_idx + (f % M) - 3*halfN;
                            // phi_hat value is stored in the absolute of the actual
                            // frequency index, M - (f % M) in [1,...,halfN]
                            factor *= phi_hat_inv[M - (f % M)];
                        }
                        else {
                            coeff_idx = -1;
                        }
                    }

                    g_hat_idx = M*g_hat_idx + (f % M);
                }

                if (coeff_idx < 0) {
                    g_hat[g_hat_idx] = make_cuFloatComplex(0.0f, 0.0f);
                }
                else {
                    coeff = coeffs_acc[coeff_idx];
                    g_hat[g_hat_idx] = make_cuFloatComplex(
                        factor * (cuCrealf(g_hat[g_hat_idx])*coeff.real() - cuCimagf(g_hat[g_hat_idx])*coeff.imag()),
                        factor * (cuCrealf(g_hat[g_hat_idx])*coeff.imag() + cuCimagf(g_hat[g_hat_idx])*coeff.real()));
                }
            }
        }
    }
}


__global__ void
real_kernel_convolution_kernel(
    const torch::PackedTensorAccessor64<float,1> coeffs_acc,
    cufftComplex *g_hat, // size batch_size * num_columns * prod_M
    const float *phi_hat_inv, // size N/2 + 1
    const int64_t dim, const int64_t batch_size, const int64_t num_columns,
    const int64_t M, const int64_t halfN, const int64_t prod_M)
{
    int64_t batch_idx, column_idx, reverse_freq_idx, coeff_idx, g_hat_idx, f, d;
    float factor;

    for (batch_idx=blockIdx.z*blockDim.z + threadIdx.z; batch_idx < batch_size; batch_idx += gridDim.z*blockDim.z)
    {
        for (column_idx = blockIdx.y*blockDim.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (reverse_freq_idx = blockIdx.x*blockDim.x + threadIdx.x; reverse_freq_idx < prod_M; reverse_freq_idx += gridDim.x*blockDim.x)
            {
                factor = 1.0f;
                coeff_idx = 0;
                g_hat_idx = batch_idx*num_columns + column_idx;

                for (d=0; d<dim; ++d) {
                    // a coeff_idx below zero indicates that this part of g_hat
                    // is set to zero and there is no fitting coefficient
                    if (coeff_idx >= 0) {
                        if (f % M < halfN) {
                            // First quarter of g_hat: refers to positive actual
                            // frequency (f % (2*N)) in [0,...,halfN-1]
                            // coefficient is stored in second half of coeffs
                            coeff_idx = 2*halfN*coeff_idx + halfN + (f % M);
                            // phi_hat value is stored in the actual frequency index
                            factor *= phi_hat_inv[f % M];
                        }
                        else if (f % M >= 3*halfN) {
                            // Last quarter of g_hat: refers to negative actual
                            // frequency (f % (2*N)) - 2*N in [-halfN,...,-1]
                            // coefficient stored in first half of coeffs
                            coeff_idx = 2*halfN*coeff_idx + (f % M) - 3*halfN;
                            // phi_hat value is stored in the absolute of the actual
                            // frequency index, 2*N - (f % (2*N)) in [1,...,halfN]
                            factor *= phi_hat_inv[M - (f % M)];
                        }
                        else {
                            coeff_idx = -1;
                        }
                    }

                    g_hat_idx = M*g_hat_idx + (f % M);
                }

                if (coeff_idx < 0) {
                    g_hat[g_hat_idx] = make_cuFloatComplex(0.0f, 0.0f);
                }
                else {
                    factor *= coeffs_acc[coeff_idx];
                    g_hat[g_hat_idx] = make_cuFloatComplex(
                        factor * cuCrealf(g_hat[g_hat_idx]),
                        factor * cuCimagf(g_hat[g_hat_idx]));
                }
            }
        }
    }
}
