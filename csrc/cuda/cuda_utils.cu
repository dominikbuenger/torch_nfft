
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

#define CHECK_ERRORS() { gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize()); }

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


__device__ __forceinline__ void
atomicAddComplex(cufftComplex *target, float re)
{
    atomicAdd((float*)target, re);
}

__device__ __forceinline__ void
atomicAddComplex(cufftComplex *target, float re, float im)
{
    atomicAdd((float*)target, re);
    atomicAdd(((float*)target) + 1, im);
}

__device__ __forceinline__ void
atomicAddComplex(cufftComplex *target, cufftComplex value)
{
    atomicAdd((float*)target, cuCrealf(value));
    atomicAdd(((float*)target) + 1, cuCimagf(value));
}
