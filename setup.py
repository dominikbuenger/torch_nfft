
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# import os

core_sources = ['csrc/core.cpp',
                # 'csrc/cpu/core_cpu.cpp',
                'csrc/cuda/core_cuda.cu']

include_dirs = ['csrc', '/usr/local/cuda/include']
library_dirs = ['/usr/local/cuda/lib64']

core_extension = CUDAExtension('torch_nfft.core',
                               core_sources,
                               include_dirs = include_dirs,
                               libraries = ['cufft'],
                               library_dirs = library_dirs,
                               extra_link_args=['-s'])

setup(
    name='torch_nfft',
    packages=['torch_nfft'],
    py_modules = [],
    ext_modules=[core_extension],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
