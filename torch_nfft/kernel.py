
import torch

from .nfft import nfft_fastsum
from .coeffs import gaussian_analytic_coeffs, gaussian_interpolated_coeffs

class GramMatrix:
    def __init__(self, coeffs, sources, targets=None, source_batch=None, target_batch=None, /, batch=None, cutoff=3):
        if targets is None:
            targets = sources
            target_batch = source_batch

        if batch is not None:
            source_batch = batch
            target_batch = batch

        self.coeffs = coeffs
        self.sources = sources
        self.targets = targets
        self.source_batch = source_batch
        self.target_batch = target_batch
        self.cutoff = cutoff


    def apply(self, x):
        return nfft_fastsum(x, self.coeffs, self.sources, self.targets,
                            self.source_batch, self.target_batch, cutoff=self.cutoff)

    def __matmul__(self, x):
        return self.apply(x)


    def is_symmetric(self):
        return self.sources is self.sources and self.source_batch is self.target_batch

    def transpose(self):
        if self.is_symmetric():
            return self
        return GramMatrix(self.coeffs, self.targets, self.sources, self.target_batch, self.source_batch, cutoff=self.cutoff)

    @property
    def T(self):
        return self.transpose()


    def row_sums(self):
        return self.apply(torch.ones(self.sources.size(0), device=self.sources.device))

    def column_sums(self):
        return self.T.row_sums()


# If uniform_radius is given, then all points are assumed to have norm of at
# most uniform_radius and the GramMatrix approximates
#    A[j, i] = K(dist(sources[i], targets[j]))
# Otherwise, the maximum radius of each input point set is computed independently
# and the GramMatrix approximates
#    A[j, i] = K(dist(sources[i], targets[j]) / radius)
class GaussianKernel:
    def __init__(self, sigma, dim=3, bandwidth=16, cutoff=3,
                shift_to_center=False, uniform_radius=None,
                analytic=False, reg_degree=-1, reg_width=0.0):

        self.cutoff = cutoff
        self.shift_to_center = shift_to_center
        self.scale_by_radius = uniform_radius is None

        self.factor = 0.25 - 0.5*reg_width
        if uniform_radius is not None:
            self.factor /= uniform_radius

        if analytic:
            self.coeffs = gaussian_analytic_coeffs(self.factor * sigma, dim, bandwidth)
        else:
            self.coeffs = gaussian_interpolated_coeffs(self.factor * sigma, dim, bandwidth, reg_degree, reg_width)


    def gram_matrix(self, sources, targets=None, source_batch=None, target_batch=None, /, batch=None):
        if self.shift_to_center:
            if targets is None:
                sources = sources - sources.mean(dim=0, keepdim=True)
            else:
                center = (sources.sum(dim=0, keepdim=True) + targets.sum(dim=0, keepdim=True)) / (sources.size(0) + targets.size(0))
                sources = sources - center
                targets = targets - center

        if self.scale_by_radius:
            if source_batch is None and batch is None:
                radius = torch.sum(sources ** 2, dim=1).max().sqrt().item()
                if targets is not None:
                    radius = max(radius, torch.sum(targets ** 2, dim=1).max().sqrt().item())

                sources = sources * (self.factor / radius)
                if targets is not None:
                    targets = targets * (self.factor / radius)
            else:
                try:
                    from torch_scatter import scatter_max
                except ImportError as e:
                    raise RuntimeError("GaussianKernel with batched input without uniform_radius requires torch_scatter") from e

                radius = scatter_max(torch.sum(sources ** 2, dim=1), batch or source_batch).sqrt()
                if targets is not None:
                    radius = torch.maximum(radius, scatter_max(torch.sum(targets ** 2, dim=1), batch or target_batch).sqrt())

                radius /= self.factor
                sources = sources / radius[batch or source_batch, None]
                if targets is not None:
                    targets = targets / radius[batch or target_batch, None]

        else:
            sources = self.factor * sources
            if targets is not None:
                targets = self.factor * targets

        return GramMatrix(self.coeffs, sources, targets, source_batch, batch=batch, cutoff=self.cutoff)


    def __call__(self, *args, **kwargs):
        return self.gram_matrix(*args, **kwargs)
