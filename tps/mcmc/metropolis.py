from math import ceil

import wandb
import traceback
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import torch
from tqdm.contrib.logging import logging_redirect_tqdm

from datasets.single_mol_dataset import SingleMolDataset
from model.internal_flow import InternalFlow
from tps.config import PathSetup
from .density import PathDensityEstimator
from .proposal import ProposalKernel
from utils.logging import get_logger
from ..path import PathProcessor, LatentEquidistantPathProcessor

logger = get_logger(__name__)


class PathProvider(ABC):
    """
    Abstract class for providing paths to the Metropolis algorithm.
    Depending on the implementation you can improve one path, or in each iteration choose from multiple possible paths.
    """

    @abstractmethod
    def get_path(self):
        raise NotImplementedError()

    @abstractmethod
    def accept_path(self, path, log_p_path):
        raise NotImplementedError()


class IterativePathProvider(PathProvider):
    def __init__(self, initial_path, log_p_path):
        super().__init__()
        self.current = initial_path
        self.log_p_current = log_p_path

    def get_path(self):
        return self.current, self.log_p_current

    def accept_path(self, path, log_p_path):
        self.current = path
        self.log_p_current = log_p_path


class RandomPreviousPathProvider(PathProvider):
    def __init__(self, initial_path, log_p_path):
        super().__init__()
        self.current_idx = 0
        self.paths = [(initial_path, log_p_path)]

    def get_path(self):
        return self.paths[self.current_idx]

    def accept_path(self, path, log_p_path):
        self.paths.append((path, log_p_path))
        self.current_idx = np.random.randint(len(self.paths))


class PathValidator(ABC):
    """
    Checks whether a proposed path is valid
    """

    @abstractmethod
    def is_valid(self, path) -> (bool, torch.Tensor):
        raise NotImplementedError()

    def __call__(self, path) -> bool:
        return self.is_valid(path)


class FixedLengthPathValidator(PathValidator):
    def __init__(self, path_setup: PathSetup, dataset: 'SingleMolDataset' = None, period=2 * np.pi):
        self.path_setup = path_setup
        self.dataset = dataset
        self.period = period

    def is_valid(self, path) -> (bool, torch.Tensor):
        start_mask = self.path_setup.start.is_in_state(path, self.dataset, self.period)
        stop_mask = self.path_setup.stop.is_in_state(path, self.dataset, self.period)
        return start_mask[0] and stop_mask[-1], path


class FlexibleLengthPathValidator(PathValidator):
    def __init__(self, path_setup: PathSetup, dataset: 'SingleMolDataset' = None, period=2 * np.pi):
        self.path_setup = path_setup
        self.dataset = dataset
        self.period = period

    def is_valid(self, path) -> (bool, torch.Tensor):
        start_mask = self.path_setup.start.is_in_state(path, self.dataset, self.period)
        stop_mask = self.path_setup.stop.is_in_state(path, self.dataset, self.period)

        starts = torch.where(start_mask)[0]
        stops = torch.where(stop_mask)[0]

        if len(starts) == 0 or len(stops) == 0:
            return False, path

        start = starts[-1]
        stop = stops[0]

        if start < stop:
            return True, path[start:(stop + 1)]
        elif start == stop:
            raise ValueError(
                "Start and stop are the same. Meaning that the states are overlapping. This is not allowed")
        else:
            return True, path[start:(stop + 1)][::-1]


class EquidistantFixedLengthPathValidator(PathValidator):
    """The idea of this validator is that it takes the shortest path between the start and stop state.
    It then adds linearly interpolated points in latent space so that the path is again fixed length
    """

    def __init__(self, path_setup: PathSetup, dataset: 'SingleMolDataset',
                 flow: InternalFlow, steps=10, atom_filter=None, period=2 * np.pi):
        self.flexible_validator = FlexibleLengthPathValidator(path_setup, dataset, period)
        self.steps = steps
        self.processor = LatentEquidistantPathProcessor(flow, steps, atom_filter)

    def is_valid(self, path) -> (bool, torch.Tensor):
        valid, new_path = self.flexible_validator.is_valid(path)
        if not valid:
            return valid, new_path

        if len(new_path) != len(path):
            len_difference = len(path) - len(new_path)

            self.processor.steps = max(self.steps, ceil(len_difference / (len(new_path) - 1)) + 2)

            new_path = self.processor(new_path, target_points=len(path))
        return True, new_path


class MetropolisHastingsStepResult:
    def __init__(self, valid, accepted, path, log_p_path, log=None):
        self.valid = valid
        self.accepted = accepted
        self.path = path
        self.log_p_path = log_p_path
        self.log = log if log is not None else {}


class MetropolisHastingsSampler:
    def __init__(self, path_setup: PathSetup, path_provider: PathProvider, proposal_kernel: ProposalKernel,
                 density_estimator: PathDensityEstimator, path_validator: PathValidator,
                 path_processor: PathProcessor = None, accept_all=False, wandb=False,
                 ignore_proposal_ratio=False):
        """
        :param path_setup: An object defining the start and stop state
        :param path_provider: An object that specifies the next path once one has been accepted.
        Typically, the last accepted path is returned.
        :param proposal_kernel: This kernel takes a path and proposes new possible paths.
        :param density_estimator: This callable takes a path and returns the log probability of the path
        :param path_validator: This callable takes a path and returns whether it is valid.
        Usually, this checks if the path starts and ends in the correct state. If None, all paths are valid.
        :param path_processor: A function that is called on each accepted path and can post-process it.
        :param accept_all: If true, all valid metropolis proposals are accepted.
        :param wandb: If true, log information to wandb
        :param ignore_proposal_ratio: If true, the proposal ratio is ignored and only the path probability is used.
        """
        self.path_setup = path_setup
        self.path_provider = path_provider
        self.proposal_kernel = proposal_kernel
        self.density_estimator = density_estimator
        self.path_validator = path_validator
        self.path_processor = path_processor
        self.accept_all = accept_all
        self.wandb = wandb
        self.ignore_proposal_ratio = ignore_proposal_ratio

    def _step(self, path_generator, log_p_current) -> [None, torch.Tensor]:
        proposal, log_proposal_ratio = next(path_generator)
        if self.ignore_proposal_ratio:
            log_proposal_ratio = torch.zeros(1, device=proposal.device)
            log = {'log_proposal_ratio': torch.nan}
        else:
            log = {'log_proposal_ratio': log_proposal_ratio.item()}

        if self.path_validator is not None:
            valid, proposal = self.path_validator.is_valid(proposal)
            if not valid:
                logger.debug(f"Rejected proposal because it is not a valid path")
                return MetropolisHastingsStepResult(valid=False, accepted=False, path=proposal, log_p_path=None,
                                                    log=log)

        if self.accept_all:
            return MetropolisHastingsStepResult(valid=True, accepted=True, path=proposal, log_p_path=None, log=log)

        log_p_proposal = self.density_estimator(proposal)
        log_acceptance_probability = log_p_proposal - log_p_current + log_proposal_ratio
        log |= {'log_acceptance_probability': log_acceptance_probability.item()}

        if log_acceptance_probability >= np.log(np.random.rand()):
            return MetropolisHastingsStepResult(valid=True, accepted=True, path=proposal, log_p_path=log_p_proposal,
                                                log=log)

        logger.debug(f"Rejected proposal with log acceptance probability {log_acceptance_probability}")
        return MetropolisHastingsStepResult(valid=True, accepted=False, path=proposal, log_p_path=log_p_proposal,
                                            log=log)

    def run_until(self, n_samples, max_errors=10_000):
        paths = []
        overall_steps, invalid_proposals, rejected_proposals = 0, 0, 0

        # Store the log probability of the current path
        path, log_p_current = self.path_provider.get_path()
        if self.path_processor is not None:
            path = self.path_processor(path)
            log_p_current = self.density_estimator(path)

        self.proposal_kernel.accept(path)
        path_generator = self.proposal_kernel(path)

        try:
            with tqdm(total=n_samples) as pbar:
                with logging_redirect_tqdm():
                    while len(paths) < n_samples:
                        pbar.set_description(
                            f'Acceptance probability: {0 if overall_steps == 0 else 100 * len(paths) / overall_steps:.2f}%')
                        try:
                            result = self._step(path_generator, log_p_current)
                            overall_steps += 1
                            if result.accepted:
                                # Parse the path and store log_p_current for the next iteration
                                path, log_p_current = result.path, result.log_p_path
                                if self.path_processor is not None:
                                    path = self.path_processor(path)
                                    log_p_current = self.density_estimator(path)

                                self.proposal_kernel.accept(path)
                                if self.wandb:
                                    log = {
                                        'current_log_p_path': log_p_current,
                                        'rejected_proposals': rejected_proposals,
                                        'invalid_proposals': invalid_proposals,
                                        'current_overall_acceptance_probability': 1 / (
                                                    1 + rejected_proposals + invalid_proposals),
                                        'current_metropolis_acceptance_probability': 1 / (1 + rejected_proposals),
                                        'current_valid_probability': 1 - (
                                                    invalid_proposals / (1 + rejected_proposals + invalid_proposals)),
                                    }
                                    log |= result.log
                                    wandb.log(log)

                                # Store the newly found path
                                paths.append(path)
                                self.path_provider.accept_path(path, log_p_current)

                                # Get the next path and tell the proposal kernel to use it as a starting point
                                path, log_p_current = self.path_provider.get_path()
                                path_generator = self.proposal_kernel(path)

                                # Update logging info
                                pbar.update(1)
                                invalid_proposals, rejected_proposals = 0, 0
                            elif result.valid:
                                rejected_proposals += 1
                            else:
                                invalid_proposals += 1
                        except StopIteration:
                            logger.warning(f"Proposal kernel ran out of proposals")
                            path_generator = self.proposal_kernel.propose(path)
                        except Exception as e:
                            logger.error(f"Exception while sampling:")
                            logger.error(traceback.format_exc())

                            path_generator = self.proposal_kernel.propose(path)

                            max_errors -= 1
                            if max_errors < 0:
                                raise e
        except KeyboardInterrupt:
            logger.info(f"Interrupted after {overall_steps} steps")

        logger.info(f"Accepted {len(paths)} paths in {overall_steps} steps")

        return paths
