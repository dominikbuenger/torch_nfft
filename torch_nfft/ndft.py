
import torch

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



def ndft_fastsum(x, coeffs, sources, targets=None, source_batch=None, target_batch=None, batch=None, N=16):
    if targets is None:
        targets = sources
        target_batch = source_batch
    if batch is not None:
        source_batch = batch
        target_batch = batch

    y = ndft_adjoint(x, sources, source_batch, N=N)

    y *= coeffs[None,...,None]

    y = ndft_forward(y, targets, target_batch)

    return y if x.is_complex() else y.real


def exact_gaussian_matrix(sigma, sources, targets=None, source_batch=None, target_batch=None, batch=None):
    if targets is None:
        targets = sources
        target_batch = source_batch
    if batch is not None:
        source_batch = batch
        target_batch = batch

    def single_gaussian_matrix(source_part, target_part):
        source_sq_norms = torch.sum(source_part**2, dim=1, keepdim=True)
        target_sq_norms = torch.sum(target_part**2, dim=1, keepdim=True)
        return torch.exp(-(target_sq_norms - 2*target_part @ source_part.T + source_sq_norms.T) / (sigma**2))

    if source_batch is None:
        return single_gaussian_matrix(sources, targets)

    batch_size = source_batch.max().item() + 1
    blocks = [single_gaussian_matrix(sources[source_batch == b], targets[target_batch == b])
                for b in range(batch_size)]
    return torch.block_diag(*blocks)
