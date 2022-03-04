

#define THREE_QUARTER_PI 2.356194490192344928846982537459627163147877049531f

#define WINDOW_FORWARD_PARAM1(N,m) (THREE_QUARTER_PI / m)
#define WINDOW_FORWARD_PARAM2(N,m) (sqrtf(0.75f / m))


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

#ifdef NFFT_PRINT_DEBUG
            if (point_idx == 0)
                printf(" - point coordinate x[%ld,%ld] = %f:  mid frequency %d, shift %d, I_Mm = {%d, ..., %d}\n",
                    point_idx, d, point_acc[point_idx][d],
                    point_shifts[point_idx*dim + d] + m, point_shifts[point_idx*dim + d],
                    point_shifts[point_idx*dim + d], point_shifts[point_idx*dim + d] + (int)(2*m+1));
#endif
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

#ifdef NFFT_PRINT_DEBUG
                if (point_idx == 0)
                    printf(" - psi for point pos[%ld,%ld] at  l=%ld: %f\n",
                        point_idx, d, window_idx - (window_length-2)/2,
                        psi[(point_idx*dim + d)*window_length + window_idx]);
#endif
            }
        }
    }
}

// Fill array g such that the sum of the impacts of all points in batch #batch_idx
// for column #column_idx on the frequency multiindex i={i[0],...,i[dim-1]} in [0,...,M-1]^d
// is stored in
//   g[ ((((batch_idx*num_columns + column_idx)*M + i[0])*M + i[1])*M + ...)*M + i[dim-1] ]
__global__ void
real_adjoint_window_convolution_kernel(
    const torch::PackedTensorAccessor64<float,2> x_acc,
    const torch::PackedTensorAccessor64<int64_t,1> batch_acc,
    const int *point_shifts,
    const float *point_psi,
    cufftComplex *g,
    const int64_t dim, const int64_t num_points, const int64_t num_columns,
    const int64_t M, const int64_t window_length, const int64_t window_volume)
{
    int64_t point_idx, batch_idx, column_idx, window_idx, w, g_idx, d;
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
                //     g[ (batch_idx*num_columns + column_idx)*M^dim + freq_idx ] =
                //     g[ ((((batch_idx*num_columns + column_idx)*M + i[0])*M + i[1])*M + ...)*M + i[dim-1] ]
                // so we can already include the first part in g_idx.
                g_idx = batch_idx*num_columns + column_idx;

                for (d=0; d<dim; ++d) {
                    // thanks to reverse storage: l[d] = w % window_length
                    g_idx = M*g_idx + ((point_shifts[point_idx*dim + d] + (w % window_length) + M) % M);
                    value *= point_psi[(point_idx*dim + d)*window_length + (w % window_length)];
                    // shave off smallest dimension l[d]
                    w /= window_length;
                }

                atomicAddComplex(g + g_idx, value);

#ifdef NFFT_PRINT_DEBUG
                //if (point_idx == 0 && column_idx == 0)
                if (window_idx == 0)
                    printf(" - Contribution of x[%ld, %ld] = %f to frequency [%ld, %ld, %ld]: %f\n",
                        point_idx, column_idx, x_acc[point_idx][column_idx],
                        (point_shifts[point_idx*dim + 0] + (window_idx % window_length) + M) % M,
                        dim > 1 ? (point_shifts[point_idx*dim + 1] + ((window_idx / window_length) % window_length) + M) % M : 0,
                        dim > 2 ? (point_shifts[point_idx*dim + 2] + (window_idx / (window_length * window_length)) + M) % M : 0,
                        value);
#endif
            }
        }
    }
}

// Version of real_adjoint_window_convolution_kernel for complex data
__global__ void
complex_adjoint_window_convolution_kernel(
    const torch::PackedTensorAccessor64<c10::complex<float>,2> x_acc,
    const torch::PackedTensorAccessor64<int64_t,1> batch_acc,
    const int *point_shifts,
    const float *point_psi,
    cufftComplex *g,
    const int64_t dim, const int64_t num_points, const int64_t num_columns,
    const int64_t M, const int64_t window_length, const int64_t window_volume)
{
    int64_t point_idx, batch_idx, column_idx, window_idx, w, g_idx, d;
    c10::complex<float> value;

    for (point_idx=blockDim.x*blockIdx.x + threadIdx.x; point_idx < num_points; point_idx += gridDim.x*blockDim.x)
    {
        batch_idx = batch_acc[point_idx];
        for (column_idx=blockDim.y*blockIdx.y + threadIdx.y; column_idx < num_columns; column_idx += gridDim.y*blockDim.y)
        {
            for (window_idx=blockDim.z*blockIdx.z + threadIdx.z; window_idx < window_volume; window_idx += gridDim.z*blockDim.z)
            {
                value = x_acc[point_idx][column_idx];
                w = window_idx;
                g_idx = batch_idx*num_columns + column_idx;

                for (d=0; d<dim; ++d) {
                    // thanks to reverse storage: l[d] = w % window_length
                    g_idx = M*g_idx + ((point_shifts[point_idx*dim + d] + (w % window_length) + M) % M);
                    value *= point_psi[(point_idx*dim + d)*window_length + (w % window_length)];
                    // shave off smallest dimension l[d]
                    w /= window_length;
                }

                atomicAddComplex(g + g_idx, value.real(), value.imag());

            }
        }
    }
}
