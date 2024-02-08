import os
from argparse import ArgumentParser
import wandb

import numpy as np
import random
import torch
from openmm import unit
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel

from model.internal_flow import InternalFlow
from tps.evaluation import ground_truth_from_path_setup, evaluate_transition_paths
from tps.evaluation.utils import load_flow_and_args
from tps.mcmc.tfproposal import LatentHamiltonianProposal
from tps.path import LatentEquidistantPathProcessor, EveryNthPathProcessor
from tps.sampling import PathSetup, LinearPathSampler, LinearLatentPathSampler, NoisyLinearLatentPathSampler
from tps.states import ALDP_STATES
from tps.mcmc.metropolis import RandomPreviousPathProvider, IterativePathProvider, FixedLengthPathValidator, \
    FlexibleLengthPathValidator, EquidistantFixedLengthPathValidator
from tps.mcmc.density import LogSumPathDensityEstimator, LangevinDensityEstimator
from tps.mcmc.proposal import LatentNoiseProposal, MdRelaxationProposal, MultiProposalKernel, \
    LatentGaussianProcessProposal
from tps.mcmc.metropolis import MetropolisHastingsSampler
from utils.training import save_yaml_file

parser = ArgumentParser()
# Wandb Arguments
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--project', type=str, default='transitionpath-inf', help='')

# Model Arguments
parser.add_argument('--model_dir', required=True, type=str,
                    help='Path to folder with trained model and hyperparameters')
parser.add_argument('--ckpt', required=True, type=str, help='Path to the checkpoint file to load')

# Path Setup
parser.add_argument('--start_state_idx', type=int, default=0, help='Index of the start state')
parser.add_argument('--end_state_idx', type=int, default=3, help='Index of the end state')
parser.add_argument('--num_paths', type=int, default=1000, help='Number of paths to sample')
parser.add_argument('--num_steps', type=int, default=40, help='Number of steps in each path')

# Sampling Arguments
parser.add_argument('--batch_size', type=int, default=-1,
                    help='Batch size for sampling. If negative, the model batch size will be used')
parser.add_argument('--sampling_method', required=True, type=str,
                    choices=['linear', 'linear_internal', 'linear_latent', 'linear_latent_noise', 'mcmc'],
                    help='Sampling method to use')
parser.add_argument('--run_name', type=str, default=None,
                    help='Name of the run that serves for plots and the output directory.')
parser.add_argument('--noise_scale', type=float, default=0.05, help='Scale of the noise to add to the latent path')

# MCMC parameters
parser.add_argument('--initial_path', type=str, default='linear_latent',
                    choices=['file', 'linear', 'linear_internal', 'linear_latent'],
                    help='How the initial path will be produced')
parser.add_argument('--initial_path_file', type=str, default=None, help='Path to a file containing the initial path')
parser.add_argument('--iteration_style', type=str, default='incremental', choices=['incremental', 'random'])
parser.add_argument('--path_density', type=str, default='energy', choices=['energy', 'langevin'])
parser.add_argument('--accept_all', action='store_true', default=False,
                    help='Accept all proposals regardless of the acceptance probability. This is useful for debugging.')
parser.add_argument('--simultaneous_proposals', type=int, default=100,
                    help='If the proposal kernel supports it, number of simultaneous proposals to use')
parser.add_argument('--base_dist', type=str, choices=['gauss', 'gauss-uni'], default='gauss')
parser.add_argument('--seed', type=int, default=None, help='Seed that will be set for randomness. This does not ensure 100% reproducibility, but drastically reduces the variance.')
parser.add_argument('--md_device', type=str, choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], default='CPU')
parser.add_argument('--torch_device', type=str, choices=['cpu', 'cuda'], default='cpu')

parser.add_argument('--langevin_timestep', type=float, default=1, help='Timestep of the langevin integrator in femtoseconds')
parser.add_argument('--langevin_independent', action='store_true', default=False, help='Evaluate the langevin steps independently')
parser.add_argument('--ignore_proposal_ratio', action='store_true', default=False, help='If this flag is set, we will ignore the proposal ratio. This is useful for debugging.')

parser.add_argument('--parallel_energy', action='store_true', default=False, help='Use parallel energy computation')
parser.add_argument('--num_energy_processes', type=int, default=None,
                    help='Number of processes to use for energy evaluation')

parser.add_argument('--relax', action='store_true', default=False)
parser.add_argument('--equidistant', type=int, default=-1,
                    help='Number of steps between two latent points used to interpolate the path and make equidistant.')
parser.add_argument('--equidistant_every', type=int, default=1, help='How often to make the path equidistant')

parser.add_argument('--gp_conditioned', action='store_true', default=False, help='Condition the gaussian process on the current path.')
parser.add_argument('--gp_fixed', action='store_true', default=False, help='Make the gaussian process fixed by only fitting it to the --gaussian_init paths.')
parser.add_argument('--gp_alpha', type=float, default=1e-2, help='Timestep of the langevin integrator in femtoseconds')
parser.add_argument('--gaussian_init', type=int, default=0,
                    help='How many paths to start the gaussian process with.')
parser.add_argument('--gaussian_window', type=int, default=0,
                    help='How many paths to include to learn a gaussian process.'
                         'If this is set to 0, no gaussian process will be used.')
parser.add_argument('--gaussian_process_sampling', type=str, choices=['fixed', 'uniform', 'gaussian'], default='fixed')
parser.add_argument('--gaussian_process_remember_x', action='store_true', default=False)

parser.add_argument('--path_validator', type=str, default='fixed', choices=['fixed', 'flexible', 'flexible_equidistant'])

parser.add_argument('--latent_hamiltonian', action='store_true', default=False)
parser.add_argument('--latent_hamiltonian_num_leapfrog_steps', type=int, default=2)
parser.add_argument('--latent_hamiltonian_step_size', type=float, default=0.1)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

        np.random.seed(args.seed)
        random.seed(args.seed)

    run_name = args.sampling_method if args.run_name is None else args.run_name
    if args.accept_all:
        args.simultaneous_proposals = 1

    if args.wandb:
        wandb.init(
            entity='coarse-graining-mit',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=run_name,
            config=args
        )

    path_setup = PathSetup(ALDP_STATES[args.start_state_idx], ALDP_STATES[args.end_state_idx])

    flow, model_args = load_flow_and_args(args.model_dir, args.ckpt,
                                          parallel_energy=args.parallel_energy,
                                          num_energy_processes=args.num_energy_processes,
                                          md_device=args.md_device, torch_device=args.torch_device,
                                          num_frames=100  # no need to load all frames :)
    )
    if args.batch_size > 0:
        model_args.__dict__["batch_size"] = args.batch_size

    if args.base_dist == 'gauss':
        flow = flow.as_normal_flow()
    elif args.base_dist == 'gauss-uni':
        flow = flow.as_mixed_flow()

    assert isinstance(flow, InternalFlow), "Only internal flow is supported"
    dataset = flow.dataset

    ground_truth = ground_truth_from_path_setup(path_setup, dataset)

    dataset_samples = dataset.frames.view(-1, 66).to(model_args.torch_device)

    if args.sampling_method == 'linear':
        sampler = LinearPathSampler(path_setup, dataset, dataset_samples)
    elif args.sampling_method == 'linear_internal':
        sampler = LinearLatentPathSampler(path_setup, flow._flows[-1], dataset_samples, dataset=dataset)
    elif args.sampling_method == 'linear_latent':
        sampler = LinearLatentPathSampler(path_setup, flow, dataset_samples)
    elif args.sampling_method == 'linear_latent_noise':
        sampler = NoisyLinearLatentPathSampler(path_setup, flow, dataset_samples, noise_scale=args.noise_scale)
    elif args.sampling_method == 'mcmc':
        if args.path_density == 'energy':
            density_estimator = LogSumPathDensityEstimator(dataset.target.log_prob_A)
        elif args.path_density == 'langevin':
            density_estimator = LangevinDensityEstimator(dataset, step_size=args.langevin_timestep * unit.femtosecond, independent_steps=args.langevin_independent)
        else:
            raise NotImplementedError(f'path density ({args.path_density}) not implemented')

        initial_path, initial_path_provider = None, None
        if args.initial_path_file == 'file':
            import mdtraj as md
            t = md.load(args.initial_path_file)

        elif args.initial_path == 'linear':
            initial_path_provider = LinearPathSampler(path_setup, dataset, dataset_samples)
        elif args.initial_path == 'linear_internal':
            initial_path_provider = LinearLatentPathSampler(path_setup, flow._flows[-1], dataset_samples,
                                                            dataset=dataset)
        elif args.initial_path == 'linear_latent':
            initial_path_provider = LinearLatentPathSampler(path_setup, flow, dataset_samples)
        else:
            raise NotImplementedError(f'initial path ({args.initial_path}) not implemented')

        if initial_path_provider is not None:  # A generator is used instead of a path
            initial_path = initial_path_provider.sample(1, num_steps=args.num_steps - 1).squeeze()

        assert initial_path is not None, "Initial path has not been set"

        path_provider = IterativePathProvider if args.iteration_style == 'incremental' else RandomPreviousPathProvider
        path_provider = path_provider(initial_path, density_estimator(initial_path))

        if args.gaussian_window > 0:
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e3)) \
                     * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e3)) \
                     + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
            proposal_kernels = [LatentGaussianProcessProposal(flow, args.gaussian_window,
                                                              conditioned=args.gp_conditioned,
                                                              sampling=args.gaussian_process_sampling,
                                                              remember_x=args.gaussian_process_remember_x,
                                                              paths=LinearLatentPathSampler(path_setup, flow, dataset_samples,return_latent=True).
                                                              sample(args.gaussian_init) if args.gaussian_init > 0 else None,
                                                              kernel=kernel,
                                                              noise_scale=args.noise_scale,
                                                              simultaneous_proposals=args.simultaneous_proposals,
                                                              fixed=args.gp_fixed,
                                                              alpha=args.gp_alpha)]
        elif args.latent_hamiltonian:
            proposal_kernels = [LatentHamiltonianProposal(flow,
                                                          step_size=args.latent_hamiltonian_step_size,
                                                          num_leapfrog_steps=args.latent_hamiltonian_num_leapfrog_steps,
                                                          simultaneous_proposals=args.simultaneous_proposals,
                                                          seed=args.seed)]
        else:
            proposal_kernels = [LatentNoiseProposal(flow, noise_scale=args.noise_scale,
                                                    simultaneous_proposals=args.simultaneous_proposals)]

        if args.relax:
            proposal_kernels.append(MdRelaxationProposal(dataset.target, num_steps=1))

        proposal_kernel = MultiProposalKernel(proposal_kernels)

        filterHs = [a.element.symbol != 'H' for a in dataset.mdtraj_topology.atoms]

        if args.equidistant > 0:
            path_processor = LatentEquidistantPathProcessor(flow, args.equidistant, atom_filter=filterHs)
            if args.equidistant_every > 1:
                path_processor = EveryNthPathProcessor(path_processor, args.equidistant_every)
        else:
            path_processor = None

        if args.path_validator == 'fixed':
            path_validator = FixedLengthPathValidator(path_setup, dataset)
        elif args.path_validator == 'flexible':
            path_validator = FlexibleLengthPathValidator(path_setup, dataset)
        elif args.path_validator == 'flexible_equidistant':
            path_validator = EquidistantFixedLengthPathValidator(path_setup, dataset, flow, 10, atom_filter=filterHs)
        else:
            raise NotImplementedError(f'path validator ({args.path_validator}) not implemented')

        mcmc = MetropolisHastingsSampler(path_setup, path_provider, proposal_kernel, density_estimator,
                                         path_validator, path_processor=path_processor, accept_all=args.accept_all,
                                         wandb=args.wandb, ignore_proposal_ratio=args.ignore_proposal_ratio)
    else:
        raise NotImplementedError(f'sampling method ({args.sampling_method}) not implemented')

    with torch.no_grad():
        if args.sampling_method != 'mcmc':
            paths = sampler.sample_batched(args.num_paths, args.num_steps - 1, model_args.batch_size)
        else:
            paths = mcmc.run_until(args.num_paths)

    results_dir = evaluate_transition_paths(run_name, dataset, ground_truth, paths, args=args)
    save_yaml_file(os.path.join(results_dir, 'eval_args.yaml'), args.__dict__)

