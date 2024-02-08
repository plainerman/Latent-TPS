import model.internal_flow as internal_flow
from datasets.single_mol_dataset import SingleMolDataset
from tps.config import PathSetup
from abc import ABC, abstractmethod
import torch
from tqdm import trange
import normflows as nf

from utils.logging import get_logger

logger = get_logger(__name__)


class PathSampler(ABC):
    def __init__(self, setup: PathSetup):
        self.setup = setup

    @abstractmethod
    def sample(self, num_samples: int, num_steps: int = 40, *args, **kwargs):
        raise NotImplementedError()

    def sample_batched(self, num_samples: int, num_steps: int, batch_size: int, *args, **kwargs):
        samples = []

        assert batch_size >= num_steps, f"Batch size ({batch_size}) must be at least num_steps ({num_steps})"

        # adjust batch size to fit num_steps
        old_batch_size = batch_size
        batch_size = (batch_size // num_steps) * num_steps
        if old_batch_size != batch_size:
            logger.info(
                f"Batch size was reduced from {old_batch_size} to {batch_size} to evenly split num_steps ({num_steps})")

        for i in trange(0, num_samples * num_steps, batch_size):
            next_i = min(i + batch_size, num_samples * num_steps)
            number_of_samples = (next_i - i) // num_steps

            samples.append(self.sample(number_of_samples, num_steps, *args, **kwargs).detach().cpu())

        return torch.cat(samples)


class LinearPathSampler(PathSampler):
    def __init__(self, setup: PathSetup, dataset: 'SingleMolDataset', samples: torch.tensor):
        super().__init__(setup)

        start_mask = setup.start.is_in_state(samples, dataset=dataset)
        end_mask = setup.stop.is_in_state(samples, dataset=dataset)

        assert start_mask.sum() > 0, "No samples in start state"
        assert end_mask.sum() > 0, "No samples in end state"

        self.possible_starts = samples[start_mask]
        self.possible_ends = samples[end_mask]

    def _interpolate(self, a, b, num_steps):
        paths = [torch.lerp(a, b, i / num_steps) for i in range(0, num_steps + 1)]
        return torch.stack(paths, dim=1).to(a.device)

    def _preprocess_samples(self, samples, num_samples: int, num_steps: int):
        return samples

    def _postprocess_paths(self, path, num_samples: int, num_steps: int):
        return path

    def sample(self, num_samples: int, num_steps: int = 40, *args, **kwargs):
        rand_start = self.possible_starts[torch.randint(high=len(self.possible_starts), size=(num_samples,))]
        rand_end = self.possible_ends[torch.randint(high=len(self.possible_ends), size=(num_samples,))]

        z0 = self._preprocess_samples(rand_start, num_samples, num_steps)
        zt = self._preprocess_samples(rand_end, num_samples, num_steps)

        path = self._interpolate(z0, zt, num_steps)
        path = self._postprocess_paths(path, num_samples, num_steps)

        return path

    def __repr__(self):
        return f"LinearPathSampler({self.setup})"


class LinearLatentPathSampler(LinearPathSampler):
    def __init__(self, setup: PathSetup, flow: [internal_flow.InternalFlow | nf.NormalizingFlow], samples: torch.tensor,
                 dataset: 'SingleMolDataset' = None, return_latent: bool = False):
        if isinstance(flow, internal_flow.InternalFlow):
            super().__init__(setup, flow.dataset, samples)
            self.flow = flow.flow
        else:
            super().__init__(setup, dataset if dataset is not None else flow.dataset, samples)
            self.flow = flow
        self.return_latent = return_latent

    def _preprocess_samples(self, samples, num_samples: int, num_steps: int):
        ret = self.flow.inverse(samples)
        if isinstance(ret, tuple):
            return ret[0]
        return ret

    def _postprocess_paths(self, path, num_samples: int, num_steps: int):
        ret = path if self.return_latent else self.flow.forward(path.view(num_samples * (num_steps + 1), -1))
        if isinstance(ret, tuple):
            ret = ret[0]
        return ret.view(num_samples, num_steps + 1, -1)

    def __repr__(self):
        return f"LinearLatentPathSampler({self.setup})"


class NoisyLinearLatentPathSampler(LinearLatentPathSampler):
    def __init__(self, setup: PathSetup, flow: internal_flow.InternalFlow, samples: torch.tensor, noise_scale: float):
        super().__init__(setup, flow, samples)

        logger.warning("I have not thought about whether start and stop should have noise."
                       "Currently they do not, but maybe we should not or filter those out where it is not in the state anymore?")

        self.noise = torch.distributions.Normal(0, noise_scale)

    def _interpolate(self, a, b, num_steps):
        paths = super()._interpolate(a, b, num_steps)
        return paths + self.noise.sample(paths.shape).to(paths.device)
