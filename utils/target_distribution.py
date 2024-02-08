import multiprocessing

import numpy as np
import openmm as mm
from boltzgen import Boltzmann, BoltzmannParallel
from openmm import app, unit
from openmm.unit import Quantity
from openmmtools import testsystems
import torch
from torch import nn
from utils.featurize import get_graph
from utils.logging import get_logger
logger = get_logger(__name__)

# Gas constant in kJ / mol / K
R = 8.314e-3

def regularize_energy(energy, energy_log_cutoff, energy_max):
    # Cast inputs to same type
    energy_log_cutoff = energy_log_cutoff.type(energy.type())
    energy_max = energy_max.type(energy.type())
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)
    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(
        energy < energy_log_cutoff, energy, torch.log(energy - energy_log_cutoff + 1) + energy_log_cutoff
    )
    energy = torch.where(energy_finite, energy,
                         torch.tensor(np.nan, dtype=energy.dtype, device=energy.device))
    return energy

class EnergyEvaluator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context, temperature):
        device = input.device
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1), dtype=input.dtype)
        forces = torch.zeros_like(input)

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                state = openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = (
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule / unit.mole) / kBT
                )

                # get forces
                f = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule / unit.mole / unit.nanometer
                    )
                    / kBT
                )
                forces[i, :] = torch.from_numpy(-f)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output[:,None], None, None

class HeteroParallelEnergyEvaluator(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """
    @staticmethod
    def var_init(temp, args):
        global temperature
        temperature = temp
    @staticmethod
    def batch_proc(input):
        pos, openmm_context = input
        # Process state
        # openmm context and temperature are passed a global variables
        pos = pos.reshape(-1, 3)


        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            energy = np.nan
            force = np.zeros_like(pos)
        else:
            openmm_context.setPositions(pos)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole) / kBT

            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule / unit.mole / unit.nanometer) / kBT
        return energy, force

    @staticmethod
    def forward(ctx, input, open_mm_contexts, pool):
        device = input[0].device
        input_list = []
        for element in input:
            input_list.append(element.cpu().detach().numpy())
        energies_out, forces_out = zip(*pool.map(HeteroParallelEnergyEvaluator.batch_proc, (input_list, open_mm_contexts)))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.concatenate(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
        energies = energies.type(input.dtype)
        forces = forces.type(input.dtype)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output[:,None], None, None

class ParallelEnergyEvaluator(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """
    @staticmethod
    def var_init(topology, system, temp, args):
        """
        Method to initialize temperature and openmm context for workers
        of multiprocessing pool
        """
        global temperature, openmm_context
        temperature = temp
        sim = app.Simulation(topology, system,
                             mm.LangevinIntegrator(temp * unit.kelvin,
                                                   1.0 / unit.picosecond,
                                                   1.0 * unit.femtosecond),
                             platform=mm.Platform.getPlatformByName(args.md_device))
        openmm_context = sim.context

    @staticmethod
    def batch_proc(input):
        # Process state
        # openmm context and temperature are passed a global variables
        input = input.reshape(-1, 3)
        n_dim = input.shape[0]

        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            energy = np.nan
            force = np.zeros_like(input)
        else:
            openmm_context.setPositions(input)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole) / kBT

            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule / unit.mole / unit.nanometer) / kBT
        return energy, force

    @staticmethod
    def forward(ctx, input, pool):
        device = input.device
        input_np = input.cpu().detach().numpy()
        energies_out, forces_out = zip(*pool.map(
            ParallelEnergyEvaluator.batch_proc, input_np))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.array(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
        energies = energies.type(input.dtype)
        forces = forces.type(input.dtype)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output[:,None], None, None


class TargetDistribution(nn.Module):
    def __init__(self, args, energy_log_cutoff=1.e+8, energy_max=1.e+20, pdb = None):
        """
        Boltzmann distribution
        :param data_path:
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_log_cutoff: Value after which the energy is logarithmically scaled
        :type energy_log_cutoff: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :type n_threads: Integer
        """

        if args.system in testsystems.__dict__:
            self.openmm_test_system = testsystems.__dict__[args.system](constraints=app.HBonds)
            self.topology = self.openmm_test_system.topology
            self.original_positions = self.openmm_test_system.positions
            self.system = self.openmm_test_system.system
        else:
            pdb = pdb if pdb is not None else app.PDBFile(args.system)
            forcefield = app.ForceField(args.forcefield, args.forcefield_water)
            self.system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1 * unit.nanometer, constraints=None)
            self.topology = pdb.topology
            self.original_positions = unit.Quantity(np.array(pdb.positions._value), pdb.positions.unit)

        self.temperature = args.temp
        self.energy_log_cutoff = torch.tensor(energy_log_cutoff)
        self.energy_max = torch.tensor(energy_max)

        self.sim = app.Simulation(self.topology, self.system,
                                  mm.LangevinMiddleIntegrator(self.temperature * unit.kelvin,  # dummy integrator
                                                              1.0 / unit.picosecond,
                                                              1.0 * unit.femtosecond),
                                  platform=mm.Platform.getPlatformByName(args.md_device))

        self.openmm_context = self.sim.context
        if args.parallel_energy:
            logger.info(f"Using parallel energy evaluation with {args.num_energy_processes} processes")
            self.process_pool = multiprocessing.Pool(args.num_energy_processes, ParallelEnergyEvaluator.var_init, (self.topology, self.system, self.temperature, args))
            self.norm_energy = lambda pos: regularize_energy(ParallelEnergyEvaluator.apply(pos, self.process_pool)[:, 0], self.energy_log_cutoff, self.energy_max)
        else:
            self.boltzmann_distribution = Boltzmann(self.openmm_context, self.temperature, energy_cut=self.energy_log_cutoff, energy_max=self.energy_max)
            self.norm_energy = self.boltzmann_distribution.norm_energy

        self.original_energy = self.norm_energy_A(
            torch.from_numpy(self.original_positions.value_in_unit(unit.angstrom)).view(1, -1, 3))

    def log_prob_A(self, x):
        """
        This function computes something relative to the log_prob of atom coordinates in Angstroms.
        So: p = exp(-Energy/kT) / Z but we return log_prob(p) = -Energy/kT
        """
        # Our x space is in Angstroms, but the Boltzmann distribution is in nm.
        # We thus divide by 10 before passing to the Boltzmann distribution.
        return -self.norm_energy(x / 10)

    def norm_energy_A(self, x):
        """
        This function computes the normalized energy of atom coordinates in Angstroms.
        So it returns Energy/kT
        Additionally, I am not sure if it is Energy/kT or Energy/RT
        Bear in mind that this energy is also regularized, so that it is logarithmic above the cutoff.
        """
        return self.norm_energy(x / 10)

    def get_sample(self):
        x = torch.from_numpy(np.asarray(self.original_positions / unit.angstrom))
        return unit.Quantity(x - x.mean(0), unit.angstrom)

    def get_force(self, pos):
        device = pos.device
        n_batch = pos.shape[0] if len(pos.shape) > 2 else 1
        pos = pos.view(n_batch, -1, 3)
        n_dim = pos.shape[1]
        energies = torch.zeros((n_batch, 1), dtype=pos.dtype)
        forces = torch.zeros_like(pos)

        kBT = R * self.temperature
        pos = pos.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = pos[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                self.openmm_context.setPositions(x)
                state = self.openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = (state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole) / kBT)

                # get forces
                f = (state.getForces(asNumpy=True).value_in_unit(unit.kilojoule / unit.mole / unit.angstrom) / kBT)
                forces[i, :] = torch.from_numpy(-f)

        return unit.Quantity(forces.to(device=device), 1/unit.angstrom), energies.to(device=device)


