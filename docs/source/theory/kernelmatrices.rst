

=================
Kernel Matrices
=================


General definition
==================

A *kernel* is a mathematical function that maps spatial vectors to scalars,
i.e., in general

.. math::
    K: \mathbb{R}^d \to \mathbb{C},

where :math:`d` is the spatial dimension. Given two sets of :math:`d`-dimensional
`source` and `target` points with :math:`n_{\mathrm{sources}}` and
:math:`n_{\mathrm{targets}}` elements, respectively, the kernel matrix is
defined as

.. math::
    A \in \mathbb{C}^{n_{\mathrm{targets}} \times n_{\mathrm{sources}}},
    \qquad
    A_{ij} = K\big( \mathrm{source}_j - \mathrm{target}_i \big).

If :math:`K(\mathbf{z})` is quickly decaying with increasing
:math:`\|\mathbf{z}\|`, then a large value in :math:`A_{ij}` means
that target :math:`i` is close to source :math:`j`.


Batch kernel matrices
---------------------

In the case that the given sources and target consist of multiple independent
data sets, the kernel matrix `A` is the block diagonal matrix whose blocks are
the kernel matrices of all the individual point/target sets. This is equivalent
to the same formula as above but with :math:`A_{ij}` set to zero if source
:math:`j` and target :math:`i` do not belong to the same set.



Trigonometric kernels
=====================

A special class of kernels are *trigonometric* functions of the form

.. math::
    K(\mathbf{z}) = \sum_{\boldsymbol{\ell} \in \mathcal{I}_N^d} b_{\boldsymbol{\ell}}
        \exp\big(2 \pi \mathrm{i} \boldsymbol{\ell}^T \mathbf{z}\big)

with the :math:`d`-dimensional multiindex set

.. math::
    \mathcal{I}_N^d = \big\{\boldsymbol{\ell} = (l_1,\ldots,l_d) \in
        \mathbb{Z}^d: -N/2 \leq l_t \leq N/2-1 \forall t=1,\ldots,d \big\}.

:math:`N` is called the *bandwidth* of the kernel.

In this case, a single matrix-vector product of the kernel matrix :math:`A` and
a vector :math:`x \in \mathbb{C}^{n_{\mathrm{sources}}}` can be written
as

.. math::
    \begin{aligned}
    (A x)_i &= \sum_{j=1}^{n_{\mathrm{sources}}} A_{ij} x_j
    = \sum_{j=1}^{n_{\mathrm{sources}}} x_j \sum_{\boldsymbol{\ell} \in \mathcal{I}_N^d}
        b_{\boldsymbol{\ell}} \exp\big(2 \pi \mathrm{i} \boldsymbol{\ell}^T
            (\mathrm{source}_j - \mathrm{target}_i) \big) \\
    &= \sum_{\boldsymbol{\ell} \in \mathcal{I}_N^d} b_{\boldsymbol{\ell}}
        \left(\sum_{j=1}^{n_{\mathrm{sources}}} x_j \exp\big(2 \pi \mathrm{i}
            \boldsymbol{\ell}^T \mathrm{source}_j \big) \right)
        \exp\big( -2 \pi \mathrm{i} \boldsymbol{\ell}^T \mathrm{target}_i \big).
    \end{aligned}

The inner sum is an adjoint NDFT and the outer sum is a forward NDFT.
Both of these can be approximated efficiently by NFFTs. The resulting
algorithm is implemented in :py:func:`nfft_fastsum`.


.. note::
    Each trigonometric kernel is always a `1`-periodic function, i.e.,

    .. math::
        K(\mathbf{z} + \mathbf{k}) = K(\mathbf{z}) \quad \forall\ \mathbf{k} \in \mathbb{Z}^d.

    Hence modifying any component of any source or target point by an integer
    does not change the result.
    In many use cases, the sources and targets are intended to be restricted to
    a small subdomain of :math:`\mathbb{R}^d`, e.g., in order to ensure that all
    differences :math:`\mathrm{source}_j - \mathrm{target}_i` are within
    :math:`[-1/2, 1/2]^d`.



Approximating general kernels
=============================

General kernels can be approximated by trigonometric kernels in order to
leverage the fast approximation of matrix products. In that case, the bandwidth
:math:`N` controls the approximation quality, where larger values improve
accuracy but also increase runtime.

A specific kernel that is of practical importance is the Gaussian kernel,

.. math::
    K(\mathbf{z}) = \exp\left( - \frac{\|\mathbf{z}\|^2}{\sigma^2} \right),

where :math:`\|\cdot\|` denotes the Euclidean norm and :math:`\sigma` is a
decay parameter.


Analytic coefficients
---------------------

Some kernels can be approximated by truncating their Fourier series. This often
requires that the kernel is itself a `1`-periodic function. In the case of the
Gaussian kernel, for example, this can be achieved by replacing :math:`K_\sigma`
by

.. math::
    K_{\sigma,P}(\mathbf{z}) = \sum_{\mathbf{k} \in \mathbb{Z}^d} K_\sigma(\mathbf{z} - \mathbf{k}),

whose Fourier coefficients are known and can be evaluated cheaply.
Their setup is implemented in :py:func:`gaussian_analytic_coeffs` or more
conveniently in :py:class:`GaussianKernel` with argument :code:`analytic=True`.

However, this approach is only practical if the decay parameter :math:`\sigma`
is small enough that :math:`K_\sigma(1/2)` is negligible. Otherwise, the
approximation of :math:`K_\sigma` by :math:`K_{\sigma,P}` will introduce a
considerable error.


Interpolated coefficients
-------------------------

A practical alternative is to choose the trigonometric kernel :math:`K_I` with
coefficients :math:`b_{\boldsymbol{\ell}}` in such a way that it interpolates a
given kernel :math:`K` on the uniform grid

.. math::
    \left\{ \frac{\mathbf{k}}{N} : \ \mathbf{k} \in \mathcal{I}_N^d \right\}
        \subset [-1/2, 1/2]^d.

The interpolation condition :math:`K_I(\mathbf{k}/N) = K(\mathbf{k}/N)` leads to

.. math::
    \sum_{\boldsymbol{\ell} \in \mathcal{I}_N^d} b_{\boldsymbol{\ell}}
    \exp\left(2 \pi \mathrm{i} \frac{\boldsymbol{\ell}^T \mathbf{k}}{N} \right)
    = K(\mathbf{k}/N),

which can be solved for the coefficients with a forward FFT.

For the Gaussian kernel, this procedure is implemented in
:py:func:`gaussian_interpolated_coeffs` or more conveniently in
:py:class:`GaussianKernel`.

For general kernels, the standard workflow consists of three steps:

* setup the interpolation grid with :py:func:`interpolation_grid` (or for radially
  symmetric kernels, :py:func:`radial_interpolation_grid`)
* evaluate the kernel on the grid,
* pass the tensor of kernel values to :py:func:`interpolated_kernel_coeffs`.
