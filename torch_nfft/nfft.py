


import torch

_nfft_adjoint = torch.ops.torch_nfft.nfft_adjoint
_nfft_forward = torch.ops.torch_nfft.nfft_forward
_nfft_fastsum = torch.ops.torch_nfft.nfft_fastsum

# See the documentation in csrc/core.cpp
def nfft_adjoint(x, pos, batch=None, N=16, m=3, real_output=False):
    return _nfft_adjoint(pos, x, batch, N, m, 1 if real_output else 0)



def nfft_forward(x, pos, batch=None, m=3, real_output=False):
    return _nfft_forward(pos, x, batch, m, 1 if real_output else 0)



class NfftFastsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeffs, sources, targets, source_batch, target_batch, cutoff_param):

        assert not coeffs.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. coefficients is not possible"
        assert not sources.requires_grad and not targets.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. sources and targets is not possible"
        assert source_batch is None or not source_batch.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. batches is not possible"
        assert target_batch is None or not target_batch.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. batches is not possible"

        y = _nfft_fastsum(sources, targets, x, coeffs, source_batch, target_batch, cutoff_param)

        ctx.save_for_backward(sources, targets, coeffs, source_batch, target_batch)
        ctx.m = cutoff_param

        return y

    @staticmethod
    def backward(ctx, dy):
        sources, targets, coeffs, source_batch, target_batch = ctx.saved_variables

        dx = _nfft_fastsum(targets, sources, dy, coeffs, target_batch, source_batch, ctx.m)

        return dx, None, None, None, None, None, None


def nfft_fastsum(x, coeffs, sources, targets=None, source_batch=None, target_batch=None, batch=None, N=16, m=3):
    if targets is None:
        targets = sources
        target_batch = source_batch
    if batch is not None:
        source_batch = batch
        target_batch = batch

    return NfftFastsumFunction.apply(x, coeffs, sources, targets, source_batch, target_batch, m)
