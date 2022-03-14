


import torch
import os

_nfft_adjoint = torch.ops.torch_nfft.nfft_adjoint
_nfft_forward = torch.ops.torch_nfft.nfft_forward
_nfft_fastsum = torch.ops.torch_nfft.nfft_fastsum

# See the documentation in csrc/core.cpp
def nfft_adjoint(x, pos, batch=None, N=16, m=3, real_output=False):
    return _nfft_adjoint(pos, x, batch, N, m, 1 if real_output else 0)


def nfft_forward(x, pos, batch=None, m=3, real_output=False):
    return _nfft_forward(pos, x, batch, m, 1 if real_output else 0)


def nfft_fastsum(x, coeffs, sources, targets=None, source_batch=None, target_batch=None, batch=None, N=16, m=3):
    if targets is None:
        targets = sources
        target_batch = source_batch
    if batch is not None:
        source_batch = batch
        target_batch = batch

    return _nfft_fastsum(sources, targets, x, coeffs, source_batch, target_batch, N, m)
