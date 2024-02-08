from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from datasets.single_mol_dataset import SingleMolDataset
import openmm as mm
from openmm import app, unit
from scipy.constants import physical_constants
import torch

from tps.path import downsample_path


class PathDensityEstimator(ABC):
    def log_p(self, path):
        return self._log_p(path)

    @abstractmethod
    def _log_p(self, path):
        raise NotImplementedError()

    def __call__(self, path):
        return self.log_p(path)


class LogSumPathDensityEstimator(PathDensityEstimator):
    """
    Applies a function to the path which returns a log probability. We take the sum of this.
    """

    def __init__(self, log_function):
        super().__init__()
        self.log_function = log_function

    def _log_p(self, path, *args, **kwargs):
        return self.log_function(path, *args, **kwargs).sum()


class LangevinDensityEstimator(PathDensityEstimator):
    """
    Use an openmmm engine to estimate the density of a path.
    We take one step of Newtonian dynamics at a time, calculate the probability of a normal distribution to step.
    We repeat this for every step in the path and return the sum of the log probabilities.
    """

    def __init__(self, dataset: SingleMolDataset, independent_steps, step_size=1.0 * unit.femtosecond,
                 friction_coefficient=1.0 / unit.picosecond):
        self.dataset = dataset
        self.independent_steps = independent_steps

        self.friction_coefficient = friction_coefficient
        self.step_size = step_size
        self.temp = self.dataset.args.temp * unit.kelvin

        self.sim = app.Simulation(dataset.target.topology, dataset.target.system,
                                  mm.LangevinIntegrator(self.temp,
                                                        self.friction_coefficient,
                                                        self.step_size),
                                  platform=mm.Platform.getPlatformByName(dataset.args.md_device))

        self.num_particles = self.sim.system.getNumParticles()
        self.masses = unit.Quantity(np.array(
            [self.sim.system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(self.num_particles)]),
            unit.dalton)

        self.v_scale = np.exp(-self.step_size * self.friction_coefficient)
        self.f_scale = (1 - self.v_scale) / self.friction_coefficient

        # variance of the normal distribution with mean at deterministic_new_pos
        unadjusted_noise_variance = unit.BOLTZMANN_CONSTANT_kB * self.temp * (1 - self.v_scale ** 2) / self.masses[:, None]

        noise_scale_SI_units = 1 / physical_constants['unified atomic mass unit'][
            0] * unadjusted_noise_variance.value_in_unit(unit.joule / unit.dalton)

        # temp * standard normal value becomes a velocity
        self.noise_scale = unit.Quantity(np.sqrt(noise_scale_SI_units), unit.meter / unit.second)

        self._noise_dist = torch.distributions.normal.Normal(loc=0, scale=1)

    def log_langevin_step_probability(self, path, start_idx, stop_idx, velocities):
        pos = unit.Quantity(path[start_idx].view(-1, 3).detach().cpu().numpy(), unit.angstrom)
        next_pos = unit.Quantity(path[stop_idx].view(-1, 3).detach().cpu().numpy(), unit.angstrom)

        # TODO: check if path is mutated
        self.sim.context.setPositions(pos)

        # TODO: what to do with velocities? Should the accumulate over the path?
        # self.sim.context.setVelocities(np.zeros_like(path[0].view(-1, 3).detach().cpu().numpy()))
        if velocities is None:
            self.sim.context.setVelocitiesToTemperature(self.temp)
        else:
            self.sim.context.setVelocities(velocities)

        state = self.sim.context.getState(getForces=True, getVelocities=True)
        forces = state.getForces(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)

        deterministic_velocities = self.v_scale * velocities + self.f_scale * forces / self.masses[:, None]

        R = ((next_pos - pos) / self.step_size - deterministic_velocities) / self.noise_scale
        log_prob = self._noise_dist.log_prob(torch.tensor(R)).sum()

        velocities = deterministic_velocities + self.noise_scale * R

        # assert np.allclose(next_pos._value, (pos + self.step_size * velocities).in_units_of(next_pos.unit)._value)

        return log_prob.item(), velocities

    def _log_p(self, path):
        # http://docs.openmm.org/latest/userguide/theory/04_integrators.html
        # https://github.com/openmm/openmm/blob/master/platforms/reference/src/SimTKReference/ReferenceStochasticDynamics.cpp

        log_prob = self.dataset.target.log_prob_A(path[0].view(1, -1, 3)).item()

        # First velocity will be random based on temperature
        velocities = None
        for i in range(len(path) - 1):
            # We can also make the steps parallel
            if self.independent_steps:
                velocities = None

            log_prob_cur, velocities = self.log_langevin_step_probability(path, i, i + 1, velocities)
            log_prob += log_prob_cur

        return log_prob
