
#define SQRT_PI 1.77245385090551602729816748334114518279754945f
#define SQUARE_PI 9.86960440108935861883449099987615113531369940f


__global__ void
fill_gaussian_analytical_coeffs_kernel(
    torch::PackedTensorAccessor64<float,1> coeffs_acc,
    float sigma,
    int64_t dim,
    int64_t N,
    int64_t prod_N)
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
