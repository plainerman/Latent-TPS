import os.path
from pathlib import Path

import mdtraj
import pdbfixer
from Bio.PDB import PDBParser, PDBIO
from openmm import app, unit
import openmm as mm
from openmmtools import testsystems
import numpy as np
import torch
import copy

from rdkit import Chem
from rdkit.Chem import MolToPDBFile, AddHs
from torch_geometric.data import Dataset
from tqdm import tqdm, trange

from tps.config import PathSetup
from utils.featurize import get_graph
from utils.general import generate_conformer, remove_all_hs
from utils.target_distribution import TargetDistribution
from utils.training import rmsdalign
from utils.logging import get_logger
from multiprocessing import Pool

logger = get_logger(__name__)


def get_quadrant_indices(quadtrants_to_remove, phi, psi):
    indices = []
    if '1' not in quadtrants_to_remove:
        indices.append(np.where((phi > 0) & (psi > 0))[0])
    if '2' not in quadtrants_to_remove:
        indices.append(np.where((phi < 0) & (psi > 0))[0])
    if '3' not in quadtrants_to_remove:
        indices.append(np.where((phi < 0) & (psi < 0))[0])
    if '4' not in quadtrants_to_remove:
        indices.append(np.where((phi > 0) & (psi < 0))[0])
    return np.concatenate(indices)


def single_direction_paths(args):
    start_state_mask, end_state_mask, forbidden_state_mask, shortest_paths_only = args

    paths = []
    source_states = np.where(start_state_mask)[0]
    target_states = np.where(end_state_mask)[0]
    forbidden_states = np.where(forbidden_state_mask)[0]

    with tqdm(total=len(source_states)) as pbar:
        i = 0
        while i < len(source_states):
            a = source_states[i]
            remaining_targets = target_states[target_states > a]
            if len(remaining_targets) > 0:
                end_state = remaining_targets[0]
                if shortest_paths_only:
                    # only take the shortest paths between the two states (so no A, A, A, B) but only (A, B)
                    remaining_sources = source_states[(source_states >= a) & (source_states < end_state)]
                    start_state = remaining_sources[-1]
                    next_i = np.where(source_states == start_state)[0][0] + 1
                    pbar.update(next_i - i)
                    i = next_i
                else:
                    start_state = a
                    i += 1
                    pbar.update(1)

                # check if any other state is on the way
                if not ((forbidden_states >= start_state) & (forbidden_states <= end_state)).any():
                    paths.append((start_state, end_state + 1))
            else:
                logger.debug(f'No more targets found. Stopping search early.')
                break

    return paths


class SingleMolDataset(Dataset):
    def __init__(self, args, target_distribution: TargetDistribution = None, transform=None):
        super(SingleMolDataset, self).__init__(transform)
        name = self.get_name(args)

        data_path = os.path.join(args.data_path, name + '.npy')
        if args.system in testsystems.__dict__:
            self.target = TargetDistribution(args)
        elif ".pdb" in args.system:
            self.target = TargetDistribution(args, pdb=app.PDBFile(args.system))
        else:
            pdb_path = os.path.join(args.data_path, name + '.pdb')
            if not os.path.exists(pdb_path):
                mol = Chem.MolFromSmiles(args.system)
                ENJOYABLE_MOLECULE = AddHs(
                    mol)  # because RDKit is a little fun, we need to add all hydrogens here and then remove them
                # again. Otherwise RDKit sometimes generatetes conformers that
                generate_conformer(ENJOYABLE_MOLECULE)
                mol = remove_all_hs(ENJOYABLE_MOLECULE)
                MolToPDBFile(mol, pdb_path)
                fixer = pdbfixer.PDBFixer(pdb_path)
                fixer.addMissingHydrogens(7.0)
                app.PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_path, 'w'))
            self.target = TargetDistribution(args, pdb=app.PDBFile(pdb_path))

        self.molecule = get_graph(args, self.target.topology)
        self.args = args
        self.original_positions_torch_angstrom = torch.from_numpy(
            self.target.original_positions.value_in_unit(unit.angstrom)).float().to(args.torch_device)
        self.mdtraj_topology = mdtraj.Topology.from_openmm(self.target.topology)

        if not os.path.exists(data_path):
            os.makedirs(args.data_path, exist_ok=True)
            logger.info(f'Generating MD data for {args.num_frames} steps at {args.temp}K at {data_path}')
            frames = []
            sim = app.Simulation(self.target.topology, self.target.system,
                                 mm.LangevinMiddleIntegrator(args.temp * unit.kelvin,
                                                             1.0 / unit.picosecond,
                                                             1.0 * unit.femtosecond),
                                 platform=mm.Platform.getPlatformByName(args.md_device))
            sim.context.setPositions(self.target.original_positions)
            sim.minimizeEnergy()
            sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)
            sim.step(args.warmup_steps)
            for _ in trange(args.num_frames):
                sim.step(args.data_save_frequency)
                frames.append(
                    sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom))
            frames = np.array(frames)
            logger.info(f'Saving generated dataset at {data_path}')
            np.save(data_path, frames)
        self.time_delta_idx = args.sampling_time_delta // args.data_save_frequency
        logger.info(f'Loading data from {data_path}')
        frames = np.float32(np.load(data_path))
        if args.max_number_of_frames < len(frames):
            frames = frames[np.random.choice(np.arange(len(frames)), size=args.max_number_of_frames, replace=False)]
        logger.info(f'Loaded {len(frames)} frames')

        phi, psi = self.phis_psis(frames)
        if args.quadrants_to_remove is not None:
            indices = get_quadrant_indices(args.quadrants_to_remove, phi, psi)
            frames = frames[indices]
            psi = psi[indices]
            phi = phi[indices]
        try:
            from utils.plotting import save_ramachandran_plot
            save_ramachandran_plot(phi, psi, 0, args, name='train_data')
        except Exception:
            pass

        self.frames = torch.from_numpy(frames)
        self.frames = self.frames.double() if args.double_precision else self.frames.float()
        reference_frame = self.frames[0]
        reference_frame = reference_frame - reference_frame.mean(dim=0)
        self.reference_frame = reference_frame.detach().cpu().numpy()
        self.heavy_atom_mask = self.molecule.node_attr0[:, 0] != 0

    def get_name(self, args):
        return\
            (f'{Path(args.system).stem}_md_data') + \
            (f'_temp{args.temp}') + \
            (f'_saveFreq{args.data_save_frequency}') + \
            (f'_numFrame{args.num_frames}') + \
            (f'_{args.forcefield}_{args.forcefield_water}' if args.system not in testsystems.__dict__ else '') \
            .replace("/", "_")

    def phis_psis(self, position):
        if torch.is_tensor(position):
            position = position.detach().cpu().numpy()
        traj = mdtraj.Trajectory(position.reshape(-1, self.mdtraj_topology.n_atoms, 3), self.mdtraj_topology)
        phi = mdtraj.compute_phi(traj)[1].squeeze()
        psi = mdtraj.compute_psi(traj)[1].squeeze()
        return phi, psi

    def transition_paths(self, path_setup: 'PathSetup', all_states, shortest_paths_only: bool = False):
        start_state_mask = path_setup.start.is_in_state(self.frames, dataset=self)
        end_state_mask = path_setup.stop.is_in_state(self.frames, dataset=self)

        forbidden_states = set(all_states)
        forbidden_states -= {path_setup.start}
        forbidden_states -= {path_setup.stop}
        forbidden_state_mask = torch.zeros_like(start_state_mask)

        for state in forbidden_states:
            forbidden_state_mask |= state.is_in_state(self.frames, dataset=self)

        with Pool(2) as p:
            result = p.map(single_direction_paths,
                           [(start_state_mask, end_state_mask, forbidden_state_mask, shortest_paths_only),
                            (end_state_mask, start_state_mask, forbidden_state_mask, shortest_paths_only)])

            # last state is exclusive
            return set(result[0] + result[1])

    def transition_paths_phis_psis(self, path_setup: 'PathSetup', all_states,
                                   shortest_paths_only: bool = False):
        paths = self.transition_paths(path_setup, all_states, shortest_paths_only)
        logger.info(f'Found {len(paths)} transition paths')

        phis, psis = self.phis_psis(self.frames)
        points = np.stack([np.array(phis), np.array(psis)]).T

        paths_phi_psi = []
        for start, end in paths:
            for i in range(start, end):
                paths_phi_psi.append(points[i])

        return np.array(paths_phi_psi)

    def len(self):
        return len(self.frames)

    def get(self, idx):
        data = copy.deepcopy(self.molecule)

        pos = torch.randn(self.frames[0].shape).to(self.args.torch_device) * self.args.prior_std
        data.pos = pos - pos.mean(dim=0)
        if self.args.gaussian_target:
            target_pos = torch.randn(data.pos.shape).to(self.args.torch_device)
            data.target_pos = target_pos - target_pos.mean(dim=0)
        else:
            data_frame = self.frames[np.random.randint(0, len(self.frames))].detach().cpu().numpy()
            aligned_pos = rmsdalign(data_frame, self.reference_frame)
            data.target_pos = torch.from_numpy(aligned_pos).to(self.args.torch_device)

        data.target_pos = data.target_pos.double() if self.args.double_precision else data.target_pos.float()

        data.vel = torch.randn(data.pos.shape).to(self.args.torch_device) * self.args.prior_std
        return data


def save_pdb_transition(source_pdb, out, frames):
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()

    parser = PDBParser()
    structure = parser.get_structure("input_structure", source_pdb)
    model = structure[0]

    writer = PDBIO()

    # Open the output PDB file
    with open(out, "w") as output_file:
        # Loop through each frame
        for i, positions in enumerate(frames):
            # Update the coordinates of each atom in the structure
            for atom, position in zip(model.get_atoms(), positions):
                atom.set_coord(position)

            output_file.write("MODEL\n")

            # Write the current frame to the output PDB file
            writer.set_structure(model)
            writer.save(output_file, write_end=False)

            output_file.write("ENDMDL\n")
