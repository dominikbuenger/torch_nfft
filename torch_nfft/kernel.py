
import torch
import math

from .coeffs import gaussian_analytic_coeffs, gaussian_interpolated_coeffs
from .utils import shift_points_by_center, scale_points_by_norm
from .matrices import GramMatrix, AdjacencyMatrix


class GaussianKernel:
    r"""
        An approximation of a Gaussian kernel function that allows fast multiplication
        with its Gram matrices.

        A typical workflow looks like this:

        * Setup a :code:`kernel` instance of this class.
        * Call :code:`matrix = kernel(...)` to obtain its :py:class:`GramMatrix`
        object w.r.t. given source and target points.
        * Use :code:`y = matrix @ x` to evaluate matrix products with the Gram
        matrix in a fast and approximative manner.


        This class has two major modes of operation. In the first case, all
        future source and target points are already known to be located in a
        ball with radius :math:`\rho`. This radius should be passed in the
        `max_euclidean_norm` or `max_infinity_norm` arguments of this class,
        depending on whether it is the :math:`L_2` or :math:`L_{\inf}` norm
        that is known in advance. Then the radius will be used to scale the
        points accordingly and setup the Gram matrix for the following kernel:

        .. math::
            K(\mathbf{z}) = \exp\left(-\frac{\|\mathbf{z}\|^2}{\sigma^2}\right)

        The second mode of operation does not require the radius a priori, but
        will scale each set of source and target points independently by its
        own radius :math:`\rho`. This results in the Gram matrix for the kernel:

        .. math::
            K(\mathbf{z}) = \exp\left(-\frac{\|\mathbf{z}\|^2}{\rho^2 \sigma^2}\right)


        Parameters
        -----------

        `sigma` (float): The parameter :math:`\sigma` of the Gaussian kernel.

        `dim` (int): The spatial dimension. Default: 3.

        `bandwidth` (int): The bandwidth of the `nfft_fastsum` method. Should
        be a small power of two, e.g., 16, 32, or 64. Default: 16.

        `cutoff` (int): The cutoff parameter of the underlying NFFT. Default: 3.

        `shift_by_center` (bool): If True, the points will be shifted to the
        origin *before* scaling and computing the gram matrix. Set to False if
        you already know that the points will be located around the origin.
        Default: True.

        `max_euclidean_norm` (float): The maximum Euclidean :math:`L_2` norm of
        the points that will be passed, if known. Default: None.

        `max_infinity_norm` (float): The maximum Euclidean :math:`L_{\inf}`
        norm of the points that will be passed, if known. Default: None.


    """

    def __init__(self, sigma, dim=3, bandwidth=16, cutoff=3,
                shift_by_center=True, max_euclidean_norm=None, max_infinity_norm=None,
                analytic=False, reg_degree=-1, reg_width=0.0):

        self.cutoff = cutoff
        self.shift_by_center = shift_by_center
        self.scale_by_norm = None
        self.factor = 0.25 - 0.5*reg_width

        if reg_degree < 0:
            radius = max_infinity_norm or max_euclidean_norm
            if radius is None:
                self.scale_by_norm = "infinity"
            else:
                self.factor /= radius
        else:
            radius = max_euclidean_norm
            if radius is None and max_infinity_norm is not None:
                radius = max_infinity_norm / math.sqrt(dim)
            if radius is None:
                self.scale_by_norm = "euclidean"
            else:
                self.factor /= radius

        if analytic:
            self.coeffs = gaussian_analytic_coeffs(self.factor * sigma, dim, bandwidth)
        else:
            self.coeffs = gaussian_interpolated_coeffs(self.factor * sigma, dim, bandwidth, reg_degree, reg_width)


    def gram_matrix(self, sources, targets=None, source_batch=None, target_batch=None, /, batch=None):
        if batch is not None:
            source_batch = batch
            target_batch = batch

        if self.shift_by_center:
            sources, targets = shift_points_by_center(sources, targets, source_batch, target_batch)

        if self.scale_by_norm is not None:
            sources, targets = scale_points_by_norm(sources, targets,
                                    source_batch, target_batch,
                                    factor=self.factor, norm=self.scale_by_norm)
        else:
            sources = self.factor * sources
            if targets is not None:
                targets = self.factor * targets

        return GramMatrix(self.coeffs, sources, targets, source_batch, target_batch, cutoff=self.cutoff)


    def __call__(self, *args, **kwargs):
        return self.gram_matrix(*args, **kwargs)


    def adjacency_matrix(self, sources, batch=None, loop_weight=1, normalization=None):
        return AdjacencyMatrix(self.gram_matrix(sources, batch=batch),
                                diagonal_offset=loop_weight-1,
                                normalization=normalization)
