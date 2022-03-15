


import torch

_nfft_adjoint = torch.ops.torch_nfft.nfft_adjoint
_nfft_forward = torch.ops.torch_nfft.nfft_forward
_nfft_fastsum = torch.ops.torch_nfft.nfft_fastsum


class NfftAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos, batch, bandwidth, cutoff, real_output):
        y = _nfft_adjoint(pos, x, batch, bandwidth, cutoff, 1 if real_output else 0)

        ctx.save_for_backward(pos, batch)
        ctx.cutoff = cutoff
        ctx.real_input = not x.is_complex()

        return y

    @staticmethod
    def backward(ctx, dy):
        pos, batch = ctx.saved_variables

        dx = _nfft_forward(pos, dy, batch, ctx.cutoff, 1 if ctx.real_input else 0)

        return dx, None, None, None, None, None


def nfft_adjoint(x, pos, batch=None, bandwidth=16, cutoff=3, real_output=False):
    return NfftAdjointFunction.apply(x, pos, batch, bandwidth, cutoff, real_output)



class NfftForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos, batch, cutoff, real_output):
        y = _nfft_forward(pos, x, batch, cutoff, 1 if real_output else 0)

        ctx.save_for_backward(pos, batch)
        ctx.cutoff = cutoff
        ctx.bandwidth = x.size(1)
        ctx.real_input = not x.is_complex()

        return y

    @staticmethod
    def backward(ctx, dy):
        pos, batch = ctx.saved_variables

        dx = _nfft_adjoint(pos, dy, batch, ctx.bandwidth, ctx.cutoff, 1 if ctx.real_input else 0)

        return dx, None, None, None, None


def nfft_forward(x, pos, batch=None, cutoff=3, real_output=False):
    return NfftForwardFunction.apply(x, pos, batch, cutoff, real_output)



class NfftFastsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeffs, sources, targets, source_batch, target_batch, cutoff):

        assert not coeffs.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. coefficients is not possible"
        assert not sources.requires_grad and not targets.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. sources and targets is not possible"
        assert source_batch is None or not source_batch.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. batches is not possible"
        assert target_batch is None or not target_batch.requires_grad, \
            "NfftFastsum: Gradient computation w.r.t. batches is not possible"

        y = _nfft_fastsum(sources, targets, x, coeffs, source_batch, target_batch, cutoff)

        ctx.save_for_backward(sources, targets, coeffs, source_batch, target_batch)
        ctx.cutoff = cutoff

        return y

    @staticmethod
    def backward(ctx, dy):
        sources, targets, coeffs, source_batch, target_batch = ctx.saved_variables

        dx = _nfft_fastsum(targets, sources, dy, coeffs, target_batch, source_batch, ctx.cutoff)

        return dx, None, None, None, None, None, None


def nfft_fastsum(x, coeffs, sources, targets=None, source_batch=None, target_batch=None, batch=None, cutoff=3):
    if targets is None:
        targets = sources
        target_batch = source_batch
    if batch is not None:
        source_batch = batch
        target_batch = batch

    return NfftFastsumFunction.apply(x, coeffs, sources, targets, source_batch, target_batch, cutoff)
