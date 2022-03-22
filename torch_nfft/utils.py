

import torch


def shift_points_by_center(sources, targets=None, source_batch=None, target_batch=None):

    if source_batch is None:
        min_coords = sources.min(dim=0).values
        max_coords = sources.max(dim=0).values
        if targets is not None:
            min_coords = torch.minimum(min_coords, targets.min(dim=0).values)
            max_coords = torch.maximum(max_coords, targets.max(dim=0).values)

        center = 0.5 * (min_coords + max_coords)
        sources = sources - center
        if targets is not None:
            targets = targets - center

    else:
        try:
            from torch_scatter import scatter_max, scatter_min
        except ImportError as e:
            raise RuntimeError("Shifting a batch of points by their centers requires torch_scatter") from e

        min_coords = scatter_min(sources, source_batch, dim=0)[0]
        max_coords = scatter_max(sources, source_batch, dim=0)[0]
        if targets is not None:
            min_coords = torch.minimum(min_coords, scatter_min(targets, target_batch, dim=0)[0])
            max_coords = torch.maximum(max_coords, scatter_max(targets, target_batch, dim=0)[0])

        centers = 0.5 * (min_coords + max_coords)
        sources = sources - centers[source_batch]
        if targets is not None:
            targets = targets - centers[target_batch]

    return sources, targets


def scale_points_by_norm(sources, targets=None, source_batch=None, target_batch=None, factor=1, norm="euclidean"):

    if source_batch is None:
        if norm == "euclidean":
            norm = lambda points: torch.sum(points ** 2, dim=1).max().sqrt().item()
        elif norm == "infinity":
            norm = lambda points: torch.abs(points).max().item()
        else:
            raise ValueError(f"scale_points_by_norm received unknown norm: {norm}")

        radius = norm(sources)
        if targets is not None:
            radius = max(radius, norm(targets))
            targets = targets * factor / radius
        sources = sources * factor / radius

    else:
        try:
            from torch_scatter import scatter_max
        except ImportError as e:
            raise RuntimeError("Scaling a batch of points by their norms requires torch_scatter") from e

        if norm == "euclidean":
            norm = lambda points, batch: scatter_max(torch.sum(points ** 2, dim=1), batch)[0].sqrt()
        elif norm == "infinity":
            norm = lambda points, batch: scatter_max(points.abs().max(dim=1).values, batch)[0]
        else:
            raise ValueError(f"scale_points_by_norm received unknown norm: {norm}")

        radius = norm(sources, source_batch)
        if targets is not None:
            radius = torch.maximum(radius, norm(targets, target_batch))

        sources = sources * factor / radius[source_batch, None]
        if targets is not None:
            targets = targets * factor / radius[target_batch, None]

    return sources, targets
