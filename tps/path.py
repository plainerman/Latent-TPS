import torch
from abc import ABC, abstractmethod

from model.internal_flow import InternalFlow


class PathProcessor(ABC):
    @abstractmethod
    def _process_path(self, path: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, path: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._process_path(path, *args, **kwargs)


class EveryNthPathProcessor(PathProcessor):
    def __init__(self, path_processor: PathProcessor, n: int):
        assert n >= 1
        self.path_processor = path_processor
        self.n = n
        self.count = -1

    def _process_path(self, path: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.count = (self.count + 1) % self.n
        if self.count == 0:
            return self.path_processor(path)
        else:
            return path


class LatentEquidistantPathProcessor(PathProcessor):
    def __init__(self, flow: InternalFlow, steps=10, atom_filter=None):
        assert steps >= 2, "Steps must be at least 2"

        self.flow = flow.flow
        self.interpolation_points = torch.linspace(0, 1, steps=steps).to(flow.device())
        self.atom_filter = atom_filter

    def _process_path(self, path: torch.Tensor, target_points: int = None,  *args, **kwargs) -> torch.Tensor:
        z = self.flow.inverse(path)
        start = z[:-1]
        end = z[1:]

        # Adapted from here https://discuss.pytorch.org/t/compute-interpolation-of-two-tensors/135352
        # We linearly interpolate at the given self.interpolation_points between each two frames
        z = torch.lerp(start[:, None, ], end[:, None, ], self.interpolation_points[None, :, None])
        z = z.view(-1, z.shape[-1])

        upsampled_path = self.flow.forward(z)
        return downsample_path(upsampled_path, len(path) if target_points is None else target_points, self.atom_filter)


def downsample_path(path: torch.tensor, target_points: int, atom_filter=None):
    assert len(path) > target_points, "Path is already shorter than target points"
    if len(path) == target_points:
        return path

    points = path.shape[0]

    start = path[:-1].view(points - 1, -1, 3)
    end = path[1:].view(points - 1, -1, 3)

    if atom_filter is not None:
        start = start[:, atom_filter]
        end = end[:, atom_filter]

    # calculate the (unaligned) rmsd between each point and the next
    length = ((start - end)**2).sum(-1).mean(-1)**0.5
    # calculate cumulative percentage of path length
    path_percentage = (torch.cumsum(length, dim=0) / length.sum())[:-1]

    # we always keep the first and last point
    points_to_find = target_points - 2

    # define which percentages we actually want to include
    target_distance = 1 / (target_points - 1)
    searching_percentages = torch.arange(1, points_to_find + 1) * target_distance

    idxs = [0]
    start = 0
    for target in searching_percentages:
        # select closest percentage to target (but ignore all previous finds)
        # important that path_percentage is sorted
        new_pos = (path_percentage[start:] - target).abs().argmin()
        idxs.append((new_pos + start + 1).item())
        start = new_pos + 1

    idxs.append(len(path) - 1)

    assert len(idxs) == target_points

    return path[idxs]
