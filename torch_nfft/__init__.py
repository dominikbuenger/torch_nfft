#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:27:14 2022

@author: dbunger
"""



import torch
import os

torch.ops.load_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'core.so'))
_nfft_adjoint = torch.ops.torch_nfft.nfft_adjoint

# See the documentation in csrc/core.cpp
def nfft_adjoint(x, pos, batch=None, N=16, m=3):
    return _nfft_adjoint(pos, x, batch, N, m)



# Non-approximatory reference implementation of NDFT
def ndft_adjoint(x, pos, N=16):
    n, d = sources.shape
    device = sources.device

    grid1d = torch.arange(N, dtype=torch.float, device=device) / N
    grid = torch.cat([g[...,None] for g in torch.meshgrid(*((grid1d,)*(d-1)), grid1d[:(N//2+1)], indexing='ij')], dim=d)

    fourier_tensor = torch.exp(2j * torch.pi * torch.tensordot(grid, sources, dims=([-1],[-1])))

    return torch.tensordot(fourier_tensor, x.to(torch.cfloat), dims=([-1],[0]))
