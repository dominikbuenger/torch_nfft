


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


def nfft_fastsum(x, coeffs, sources, targets=None, source_batch=None, target_batch=None, /, batch=None, cutoff=3):
    r"""
        Approximates multiplication with a trigonometric kernel matrix.
        

        The following variants of this function can be used, all of which have
        an additional optional `cutoff` keyword argument to control the
        approximation quality:

        .. code:: python

            y = nfft_fastsum(x, coeffs, sources)
            y = nfft_fastsum(x, coeffs, sources, targets)
            y = nfft_fastsum(x, coeffs, sources, batch=batch)
            y = nfft_fastsum(x, coeffs, sources, targets, batch=batch)
            y = nfft_fastsum(x, coeffs, sources, targets, source_batch, target_batch)


        Parameters
        ----------
        `x` (Tensor): The tensor to be multiplied from the left by the
        trigonometric kernel matrix. `x.size(0)` must be equal to the
        number of source points.

        `coeffs` (Tensor): The tensor holding the trigonometric coefficients
        :math:`b_{\boldsymbol{\ell}}`. It must be :math:`d`-dimensional
        and its size in every dimension must be the same, which is
        used as the bandwidth :math:`N`. Compared to :math:`b`, its
        contents are shifted such that :math:`b_{(l_1,\ldots,l_d)}` is
        stored in :code:`coeffs[l_1+N/2,...,l_d+N/2]` for indices with
        :math:`-N/2 \leq l_t \leq N/2-1`.

        `sources`, `targets` (Tensor): The source and target
        points, given as 2D tensors with shape `[n, d]`, where `n` is
        the number of sources or targets, respectively. If the `targets` are
        omitted, the `sources` are used for both point sets.

        `source_batch`, `target_batch` (Tensor, optional): The batch indices
        of the source and target points. For `i in range(num_sources)`,
        `source_batch[i]` must be an integer from `range(batch_size)`
        indicating to which point set in the batch the point
        `sources[i]` belongs. The batch vectors must be ordered such
        that `source_batch[-1] == target_batch[-1] == batch_size-1`.
        Default: `None`, indicating that there is only one point set.

        `batch` (Tensor, optional): If only one batch vector is given, it
        is used as both `source_batch` and `target_batch`. Must be
        given as a keyword argument.

        `cutoff` (int, optional): The cutoff parameter of the NFFT.
        Higher values improve accuracy but greatly increase runtime.
        Default: `3`.

        :rtype: `Tensor`

        If the input tensor `x` is real-valued, then the output `y` is made
        real-valued as well by returning only the real part.


        The algorithm is targeted at the following use-case:

        * the spatial dimension :math:`d` is at most 3,
        * the number of source and target points per point set is large,
        * the bandwidth :math:`N` is a comparably small power of two (e.g., 16,
          32, or 64), and the `cutoff` parameter is small (:math:`\leq 8` and
          :math:`< N/2`).


        The computational cost for a single input vector `x` is in

        .. math::
            \mathcal{O}(m^d \max(n_{\mathrm{sources}},\, n_{\mathrm{targets}}) + N^d \log(N)),

        where :math:`m` is the `cutoff` parameter. Under the above assumptions,
        this may be much more efficient than the general cost
        :math:`\mathcal{O}(n_{\mathrm{sources}} n_{\mathrm{targets}})` of
        multiplication with a dense kernel matrix. The algorithm needs to
        allocate essentially :math:`2^{d+1} N^d` floats.

    """

    if targets is None:
        targets = sources
        target_batch = source_batch
    if batch is not None:
        source_batch = batch
        target_batch = batch

    return NfftFastsumFunction.apply(x, coeffs, sources, targets, source_batch, target_batch, cutoff)
