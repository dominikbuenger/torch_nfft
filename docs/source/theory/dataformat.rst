

===============
Data format
===============


Point sets
===========

Each :math:`d`-dimensional point set of :math:`n` points is expected to be
given in a 2D tensor with shape `[n,d]`, such that :code:`points[i]` gives the
`i`-th point.

For functions that need a source and target point sets, providing only a
source tensor results in using that same tensor for the targets as well.


Batches of point sets
=====================

In batched mode, it is possible to collect multiple point sets in a batch.
The points are expected to be given in one large 2D tensor with shape `[n, d]`,
where `n` is the total number of points in all batches combined. The tensor
can be thought of as the vertical concatenation of the independent point set
tensors. As opposed to a 3D tensor whose slices hold the point sets, this
format allows for the different point sets to have independent sizes.

In order to specify which point in the tensor belongs to which point set, you
need to pass a `batch` vector with as many entries as the total number of
points. Its `dtype` must be `torch.long` and its entries must be in
`range(batch_size)`, where `batch_size` is the number of point sets in the
batch.

The batch vector should be sorted such that its last element is the largest
point set index, `batch_size - 1`, as this element is used to deduce the
batch size.


Spatial input and output
========================

The output of :py:func:`nfft_forward`, the input of :py:func:`nfft_adjoint`,
and the input and output of :py:func:`nfft_fastsum` all contain *spatial*
signals where the `i`-th slice refers to the `i`-th point in the point set (or
batch of point sets). For that reason, :code:`x.size(0)` must be equal to the
total number of points. The other dimensions can be of arbitrary size, which
means that multiple signals are transformed simultaneously.


Spectral input and output
=========================

The input of :py:func:`nfft_forward` and the output of :py:func:`nfft_adjoint`
contain *spectral* signals. For each frequency multiindex
:math:`\mathbf{k} = (k_1,\, \ldots,\, k_d) \in \mathcal{I}_N^d`, the slice
:code:`x[b, k_1, ..., k_d]` contains the values of all signals belonging to
point set `#b` in the batch (`b` is in `range(batch_size)`) at frequency
:math:`\mathbf{k}`. For that reason, the tensor must satisfy
:code:`x.size(0) == batch_size` and
:code:`x.size(1) == ... == x.size(d) == N`, where `N` is the frequency
bandwidth. The remaining dimensions can be of arbitrary size, which
means that multiple signals are transformed simultaneously.


Trigonometric coefficients
==========================

For a trigonometric kernel `K_b`,
the coefficients :math:`b_{\boldsymbol{\ell}}` are given in the input tensor
`coeffs` which must be :math:`d`-dimensional with shape `[N,...,N]`. The
index set is shifted such that :math:`b_{(l_1,\ldots,l_d)}` is stored in
`coeffs[l_1,...,l_d]`.
