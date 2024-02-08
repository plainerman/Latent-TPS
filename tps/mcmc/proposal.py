from abc import ABC, abstractmethod

import torch

from model.internal_flow import InternalFlow
from tps.path import LatentEquidistantPathProcessor
from utils.target_distribution import TargetDistribution
from openmm import unit

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
from scipy.special import lambertw

from utils.logging import get_logger

logger = get_logger(__name__)


class ProposalKernel(ABC):
    def propose(self, path: torch.Tensor):
        """
        Propose a new path based on the given path.
        Returns a tuple of the new path and the log(p(new -> old)) - log(p(old -> new)).
        """
        state = self._setup(path)

        while True:
            yield from self._propose(state, path)

    def _setup(self, path):
        """
        This function allows you to return as state that can be used as a cache.
        This function might be called multiple times on the same path if there is an error.
        """
        return None

    def accept(self, path):
        """
        Called once if a path is accepted.
        """
        pass

    @abstractmethod
    def _propose(self, state, path) -> [(torch.Tensor, torch.Tensor)]:
        raise NotImplementedError()

    def __call__(self, path: torch.Tensor):
        return self.propose(path)


class EveryNthProposal(ProposalKernel):
    def __init__(self, proposal_kernel: ProposalKernel, n: int):
        assert n >= 1
        self.proposal_kernel = proposal_kernel
        self.n = n
        self.count = -1

    def _setup(self, path):
        self.count = (self.count + 1) % self.n
        if self.count == 0:
            return self.proposal_kernel._setup(path)
        else:
            return None

    def _propose(self, state, path: torch.Tensor) -> [(torch.Tensor, torch.Tensor)]:
        if self.count == 0:
            return self.proposal_kernel._propose(state, path)
        else:
            return [path, 0]

    def accept(self, path):
        if self.count == 0:
            self.proposal_kernel.accept(path)


class MultiProposalKernel(ProposalKernel):
    """
    Use this kernel if you want to combine multiple kernels into one.
    For example, propose an initial path, which is then smoothed by short md runs.
    """

    def __init__(self, proposal_kernels: [ProposalKernel]):
        assert len(proposal_kernels) > 0
        self.proposal_kernels = proposal_kernels

    def _setup(self, path):
        return [kernel._setup(path) for kernel in self.proposal_kernels]

    def _propose(self, state, path) -> [(torch.Tensor, torch.Tensor)]:
        def propose_recursive(kernels, paths, log_probs):
            if len(kernels) == 0:
                for p, l in zip(paths, log_probs):
                    yield p, l
                return

            kernel = kernels[0]
            current_state = state[self.proposal_kernels.index(kernel)]
            for p in paths:
                new_paths, new_log_probs = zip(*kernel._propose(current_state, p))
                new_log_probs = torch.asarray(new_log_probs) + log_probs

                yield from propose_recursive(kernels[1:], new_paths, new_log_probs)

        return propose_recursive(self.proposal_kernels, [path], torch.zeros(1))

    def accept(self, path):
        for kernel in self.proposal_kernels:
            kernel.accept(path)


class MdRelaxationProposal(ProposalKernel):
    def __init__(self, target: TargetDistribution, num_steps):
        super().__init__()
        self.sim = target.sim
        self.temp = target.temperature
        self.num_steps = num_steps

        # TODO: This might be an issue
        # the string method has originally been implemented for finding the lowest energy path
        # see "String method for calculation of minimum free-energy paths in Cartesian space in freely-tumbling systems"
        # by Davide Branduardi* and José D. Faraldo-Gómez

        logger.warning("This relaxation implementation is not compatible with Metropolis-Hastings.")

    def _propose(self, _state, path: torch.Tensor):
        new_path = []
        original_path = path

        path = path.view(len(path), -1, 3).detach().cpu().numpy()
        for frame in path:
            self.sim.context.setPositions(unit.Quantity(frame, unit.angstrom))
            self.sim.context.setVelocitiesToTemperature(self.temp * unit.kelvin)

            self.sim.step(self.num_steps)

            state = self.sim.context.getState(getPositions=True)
            new_position = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

            new_path.append(new_position.flatten())

        return [(torch.tensor(new_path, device=original_path.device, dtype=original_path.dtype), 0)]


class LatentEquidistantProposal(ProposalKernel):
    def __init__(self, flow: InternalFlow, steps=10, atom_filter=None):
        assert steps >= 2, "Steps must be at least 2"
        super().__init__()

        self.path_processor = LatentEquidistantPathProcessor(flow, steps, atom_filter)

    def _propose(self, state, path) -> [(torch.Tensor, torch.Tensor)]:
        return [(self.path_processor(path), 0)]


class LatentNoiseProposal(ProposalKernel):
    def __init__(self, flow: InternalFlow, noise_scale=0.1, simultaneous_proposals=1):
        super().__init__()
        self.flow = flow.flow

        self.noise_scale = noise_scale
        self.auto_noise = False
        # check if noise_scale is a string and equals auto or is a float
        if isinstance(noise_scale, str):
            if noise_scale != 'auto':
                raise NotImplementedError(f'noise_scale {noise_scale} is not implemented')
            self.auto_noise = True

        self.device = flow.device()
        self.simultaneous_proposals = simultaneous_proposals

    def _setup(self, path):
        # we store the inverse because it is expensive
        z_path, log_det_x = self.flow.inverse_and_log_det(path.to(self.device))

        if torch.isnan(z_path).any() or torch.isnan(log_det_x).any():
            raise RuntimeError("NaN encountered in LatentNoiseProposal._setup. Cannot create state.")

        if self.auto_noise:
            log_det_x_numpy = log_det_x.detach().cpu().numpy()
            log_sigma = lambertw(1 / np.exp(2 * log_det_x_numpy)) / 2 + log_det_x_numpy

            assert (log_sigma.real == log_sigma).all()

            noise = torch.distributions.Normal(0, torch.tensor(np.exp(log_sigma.real), dtype=torch.float32, device=self.device))
            raise NotImplementedError("This is not implemented correctly. The noise is not scaled correctly.")
        else:
            noise = torch.distributions.Normal(0, self.noise_scale)

        return z_path, log_det_x, noise

    def _propose(self, state, path: torch.Tensor):
        z_path, log_det_x, noise = state

        # new_z_paths.shape = (proposal_count, path_length, latent_dim)
        new_z_paths = z_path.repeat(self.simultaneous_proposals, 1, 1)
        new_z_paths += noise.sample(new_z_paths.shape).to(self.device)

        # normal distribution is symmetrical, so p(new -> old) = p(old -> new)
        new_paths, new_log_det_z = self.flow.forward_and_log_det(new_z_paths.view(-1, new_z_paths.shape[-1]))
        new_paths = new_paths.view(self.simultaneous_proposals, -1, new_paths.shape[-1])
        new_log_det_z = new_log_det_z.view(self.simultaneous_proposals, -1)

        # the proposal ratio is J (F(z')) / J (F(z)) = J (F(z')) * J (F^-1(x))
        # log_proposal_ratio = log_det_z_new - (-log_det_x) = log_det_z_new + log_det_x
        # further we assume that all frames are independent
        return [(path, torch.zeros(1, device=path.device) if self.auto_noise else (log_det_z + log_det_x).sum()) for path, log_det_z in zip(new_paths, new_log_det_z)]


class LatentGaussianProcessProposal(ProposalKernel):
    def __init__(self, flow: InternalFlow, window_size: int, conditioned, sampling='fixed', remember_x=False, paths=None,
                 kernel=None, noise_scale=0.1, simultaneous_proposals=1, fixed=False, alpha=1e-2):
        super().__init__()
        self.flow = flow.flow
        self.window_size = window_size
        self.sampling = sampling
        self.remember_x = remember_x
        self.noise_scale = noise_scale
        self.conditioned = conditioned
        if paths is None:
            self.paths = None
        else:
            self.pathXs = np.arange(paths.shape[1]).repeat(len(paths)).reshape(-1, len(paths)).T.flatten().reshape(-1, 1)
            self.paths = paths.detach().cpu().numpy()
            self.next_pathX = np.arange(paths.shape[1]).reshape(-1, 1)
        self._kernel = kernel
        self.device = flow.device()
        self.simultaneous_proposals = simultaneous_proposals
        self.fixed = fixed
        self.gp = None
        self.alpha = alpha

        if self.fixed and paths is None:
            raise ValueError('Cannot use fixed=False and paths=None. Please provide paths.')

    def accept(self, path):
        # we store the inverse because it is expensive
        z_path, log_det_x = self.flow.inverse_and_log_det(path.to(self.device))

        if torch.isnan(z_path).any() or torch.isnan(log_det_x).any():
            raise RuntimeError("NaN encountered in LatentNoiseProposal._setup. Cannot create state.")

        if not self.remember_x or self.paths is None:
            self.next_pathX = np.arange(len(path)).reshape(-1, 1)

        if not self.fixed:
            if self.paths is None:
                self.pathXs = np.arange(len(path)).reshape(-1, 1)
                self.paths = z_path.detach().cpu().numpy()[None,]
            else:
                self.pathXs = np.concatenate([self.pathXs, self.next_pathX])[-self.window_size * len(self.next_pathX):]
                self.paths = np.concatenate([self.paths, z_path.detach().cpu().numpy()[np.newaxis, ...]], axis=0)[
                             -self.window_size:]

        if not self.fixed or self.gp is None:
            logger.info(f'Fitting GP with {self.paths.shape[0]} paths and {self.paths.shape[1]} frames each.')
            gp = GaussianProcessRegressor(kernel=self._kernel, n_restarts_optimizer=2, normalize_y=True,
                                          alpha=self.alpha)
            gp.fit(self.pathXs, self.paths.reshape(-1, z_path.shape[1]))

            if self._kernel is None:
                self._kernel = gp.kernel_
            self._kernel.set_params(**(gp.kernel_.get_params()))

            self.gp = gp

            mean_prediction, std_prediction = gp.predict(self.next_pathX, return_std=True)
            mean_prediction2, std_prediction2 = gp.predict(np.arange(len(path)).reshape(-1, 1), return_std=True)

            print(mean_prediction, std_prediction)
            print(mean_prediction2, std_prediction2)

    def _setup(self, path):
        return self.flow.inverse_and_log_det(path.to(self.device))

    def _propose(self, state, path: torch.Tensor):
        z_path, log_det_x = state

        if self.sampling == 'fixed':
            direction = 0.5 if np.random.random() < 0.5 else -0.5
            self.next_pathX = np.arange(len(path)).reshape(-1, 1) + direction
        elif self.sampling == 'uniform':
            self.next_pathX = np.sort((np.random.random(size=len(path)) * len(path) - 0.5)).reshape(-1, 1)
        elif self.sampling == 'gaussian':
            if abs(self.noise_scale) < 1e-6:
                self.next_pathX = np.arange(len(path)).reshape(-1, 1)
            else:
                self.next_pathX = np.sort(np.random.normal(np.arange(len(path)), scale=self.noise_scale)).reshape(-1, 1)
        else:
            raise NotImplementedError(f'Sampling method {self.sampling} is not implemented.')

        y_mean, y_cov = self.gp.predict(self.next_pathX, return_cov=True)

        if self.conditioned:
            y_mean = z_path.detach().cpu().numpy()

        y_samples = [
            np.random.multivariate_normal(
                y_mean[:, target], y_cov[..., target], self.simultaneous_proposals
            ).T[:, np.newaxis]
            for target in range(y_mean.shape[1])
        ]
        y_samples = np.hstack(y_samples)

        new_z_paths = torch.tensor(np.swapaxes(np.swapaxes(y_samples, 1, 2), 0, 1), dtype=torch.float32, device=self.device)

        new_paths, new_log_det_z = self.flow.forward_and_log_det(new_z_paths.view(-1, new_z_paths.shape[-1]))
        new_paths = new_paths.view(self.simultaneous_proposals, -1, new_paths.shape[-1])
        new_log_det_z = new_log_det_z.view(self.simultaneous_proposals, -1)

        # the proposal ratio is J (F(z')) / J (F(z)) = J (F(z')) * J (F^-1(x))
        # log_proposal_ratio = log_det_z_new - (-log_det_x) = log_det_z_new + log_det_x
        # further we assume that all frames are independent
        return [(path, (log_det_z + log_det_x).sum()) for path, log_det_z in zip(new_paths, new_log_det_z)]

