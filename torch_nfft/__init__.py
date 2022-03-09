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
_nfft_forward = torch.ops.torch_nfft.nfft_forward

# See the documentation in csrc/core.cpp
def nfft_adjoint(x, pos, batch=None, N=16, m=3, real_output=False):
    return _nfft_adjoint(pos, x, batch, N, m, 1 if real_output else 0)


def nfft_forward(x, pos, batch=None, m=3, real_output=False):
    return _nfft_forward(pos, x, batch, m, 1 if real_output else 0)


# Non-approximatory reference implementation of NDFT
def ndft_adjoint(x, pos, batch=None, N=16):
    n, d = pos.shape
    device = pos.device
    x = x.to(torch.cfloat)

    grid1d = torch.arange(-N/2, N/2, dtype=torch.float, device=device)
    grid = torch.cat([g[...,None] for g in torch.meshgrid(*((grid1d,)*d), indexing='ij')], dim=d)

    def single_ndft(x_part, pos_part):
        fourier_tensor = torch.exp(2j * torch.pi * torch.tensordot(grid, pos_part, dims=([-1],[-1])))
        y = torch.tensordot(fourier_tensor, x_part, dims=1)
        return y[None,...]

    if batch is None:
        return single_ndft(x, pos)

    else:
        batch_size = batch.max().item() + 1
        return torch.cat([single_ndft(x[batch == idx], pos[batch == idx]) for idx in range(batch_size)])


def ndft_forward(x, pos, batch=None):
    n, d = pos.shape
    device = pos.device
    x = x.to(torch.cfloat)
    N = x.shape[1]

    grid1d = torch.arange(-N/2, N/2, dtype=torch.float, device=device)
    grid = torch.cat([g[...,None] for g in torch.meshgrid(*((grid1d,)*d), indexing='ij')], dim=d)

    def single_ndft(x_part, pos_part):
        fourier_tensor = torch.exp(-2j * torch.pi * torch.tensordot(pos_part, grid, dims=([-1],[-1])))
        return torch.tensordot(fourier_tensor, x_part, dims=d)

    if batch is None:
        return single_ndft(x[0], pos)

    else:
        batch_size = batch.max().item() + 1
        return torch.cat([single_ndft(x[idx], pos[batch == idx]) for idx in range(batch_size)])
