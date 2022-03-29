
import torch
from .nfft import nfft_fastsum

class AbstractMatrix:
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device

    def apply(self, x):
        raise NotImplementedError()

    def __matmul__(self, x):
        return self.apply(x)


    def is_symmetric(self):
        return False

    def transpose(self):
        if self.is_symmetric():
            return self
        raise NotImplementedError()

    @property
    def T(self):
        return self.transpose()


    def row_sums(self):
        return self.apply(torch.ones(self.shape[0], device=self.device))

    def column_sums(self):
        return self.T.row_sums()

    def to_dense(self):
        return self.apply(torch.eye(self.shape[0], device=self.device))


class GramMatrix(AbstractMatrix):
    def __init__(self, coeffs, sources, targets=None, source_batch=None, target_batch=None, /, batch=None, cutoff=3):
        if targets is None:
            targets = sources
            target_batch = source_batch

        if batch is not None:
            source_batch = batch
            target_batch = batch

        super().__init__((sources.size(0), targets.size(0)), sources.device)

        self.coeffs = coeffs
        self.sources = sources
        self.targets = targets
        self.source_batch = source_batch
        self.target_batch = target_batch
        self.cutoff = cutoff


    def apply(self, x):
        return nfft_fastsum(x, self.coeffs, self.sources, self.targets,
                            self.source_batch, self.target_batch, cutoff=self.cutoff)

    def is_symmetric(self):
        return self.sources is self.sources and self.source_batch is self.target_batch

    def transpose(self):
        if self.is_symmetric():
            return self
        return GramMatrix(self.coeffs, self.targets, self.sources, self.target_batch, self.source_batch, cutoff=self.cutoff)



class AdjacencyMatrix(AbstractMatrix):

    def __init__(self, gram_matrix, diagonal_offset=0, normalization=None, degree_threshold=0):

        if not gram_matrix.is_symmetric():
            raise ValueError("The underlying Gram matrix of an AdjacencyMatrix must be symmetric")

        super().__init__(gram_matrix.shape, gram_matrix.device)

        self.gram_matrix = gram_matrix
        self.diagonal_offset = diagonal_offset
        self.normalization = normalization

        if normalization is not None and normalization != "none":
            degrees = gram_matrix.row_sums()
            if diagonal_offset != 0:
                degrees += diagonal_offset

            negative_nodes = degrees < degree_threshold
            if torch.any(negative_nodes):
                import warnings
                # Force warnings.warn() to omit the source code line in the message
                formatwarning_orig = warnings.formatwarning
                warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
                    formatwarning_orig(message, category, filename, lineno, line='')
                warnings.warn("AdjacencyMatrix with normalization: {} out of {} node degrees are smaller than the threshold {:.4g}".format(
                    torch.sum(negative_nodes), degrees.numel(), degree_threshold),
                    RuntimeWarning, stacklevel=1)
                warnings.formatwarning = formatwarning_orig

                degrees[negative_nodes] = torch.inf

            # normalization == "rw" is a synonym for "left"
            if normalization == "rw":
                normalization = "left"

            if normalization == "sym":
                self.d_inv_sqrt = torch.rsqrt(degrees)
            elif normalization == "left" or normalization == "right":
                self.d_inv = 1 / degrees
            else:
                raise ValueError(f"Unknown AdjacencyMatrix normalization type: {normalization}")

    def apply_left_normalization(self, x):
        if self.normalization == "sym":
            return self.d_inv_sqrt[(...,) + (None,)*(x.dim()-1)] * x
        elif self.normalization == "left":
            return self.d_inv[(...,) + (None,)*(x.dim()-1)] * x
        return x

    def apply_right_normalization(self, x):
        if self.normalization == "sym":
            return self.d_inv_sqrt[(...,) + (None,)*(x.dim()-1)] * x
        elif self.normalization == "right":
            return self.d_inv[(...,) + (None,)*(x.dim()-1)] * x
        return x

    def apply(self, x):
        x = self.apply_right_normalization(x)
        y = self.gram_matrix @ x
        if self.diagonal_offset != 0:
            y += self.diagonal_offset * x
        return self.apply_left_normalization(y)


    def is_symmetric(self):
        return self.normalization != "left" and self.normalization != "right"

    def transpose(self):
        if self.normalization == "left" or self.normalization == "right":
            transposed = AdjacencyMatrix(self.gram_matrix, self.diagonal_offset, normalization=None)
            transposed.normalization = "right" if self.normalization == "left" else "left"
            transposed.d_inv = self.d_inv
            return transposed
        return self
