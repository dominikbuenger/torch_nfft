
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
