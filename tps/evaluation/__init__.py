import os
import pathlib
import pickle

import wandb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import subprocess
import torch

from datasets.single_mol_dataset import SingleMolDataset, save_pdb_transition
from utils.plotting import kl_divergence
from .utils import kl_divergence2d, hist_abs_dist
from boltzgen import CoordinateTransform
from model.internal_flow import ind_circ_dih, z_matrix, cart_indices, default_std

from utils.logging import get_logger
from ..config import PathSetup
from ..path import downsample_path
from ..plot import PeriodicPathHistogram
from ..states import ALDP_STATES

logger = get_logger(__name__)

bins_1d = 250
bins_2d = 100


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def ground_truth_from_path_setup(path_setup: PathSetup, dataset: SingleMolDataset):
    dataset_name = dataset.get_name(dataset.args)
    data_path = os.path.join(dataset.args.data_path, dataset_name + str(path_setup) + '.pickle')
    if os.path.exists(data_path):
        logger.info(f"Loading ground truth from {data_path} for {path_setup}")
        with open(data_path, 'rb') as handle:
            ground_truth = pickle.load(handle)
    else:
        ground_truth = dataset.transition_paths(path_setup, ALDP_STATES, shortest_paths_only=False)
        logger.info(f"Saving ground truth to {data_path}")
        with open(data_path, 'wb') as handle:
            pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return [dataset.frames[start:stop] for start, stop in ground_truth]


def evaluate_transition_paths(run_name, dataset: SingleMolDataset, ground_truth: list[np.array],
                              other: list[np.array], results_dir='./results', args={}):
    out_dir = os.path.join(results_dir, run_name)
    if os.path.exists(out_dir):
        logger.warning(f"Results directory {out_dir} already exists, overwriting")
    else:
        os.makedirs(out_dir)

    pathlib.Path(os.path.join(out_dir, 'commit.sha')).write_text(get_git_revision_hash())

    logger.info(f"Saving results to {out_dir}")
    with open(os.path.join(out_dir, 'paths.pickle'), 'wb') as handle:
        pickle.dump(other, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dim = dataset.reference_frame.shape[0] * 3
    coordinate_transform = CoordinateTransform(torch.from_numpy(dataset.reference_frame).view(-1, dim), dim,
                                               z_matrix, cart_indices, mode='internal',
                                               ind_circ_dih=ind_circ_dih, shift_dih=False,
                                               default_std=default_std)

    aligned = coordinate_transform.forward(coordinate_transform.inverse(other[-1].view(-1, 66).detach().cpu())[0])[0]
    save_pdb_transition('./data/aldp.pdb', os.path.join(out_dir, f'aldp_{run_name}_path_-1.pdb'), aligned.view(-1, 22, 3))
    if len(aligned) > 40:
        save_pdb_transition('./data/aldp.pdb', os.path.join(out_dir, f'aldp_{run_name}_path_-1_equidistant.pdb'), downsample_path(aligned, 40).view(-1, 22, 3))

    ground_truth = [np.array(dataset.phis_psis(path)).T for path in ground_truth]
    other = [np.array(dataset.phis_psis(path)).T for path in other]

    _save_path_histogram(run_name, ground_truth, other)
    plt.savefig(os.path.join(out_dir, 'path_histogram.png'))
    plt.clf()

    _save_path_histogram_non_smoothed(run_name, ground_truth, other)
    plt.savefig(os.path.join(out_dir, 'path_histogram_non_smoothed.png'))
    plt.clf()

    ground_truth = np.concatenate(ground_truth)
    other = np.concatenate(other)

    _save_marginal_densities(run_name, ground_truth, other)
    plt.savefig(os.path.join(out_dir, 'marginal_densities.png'))
    plt.clf()

    # TODO: probably use path histograms or more samples for KL divergence with interpolated paths
    kl_info = _calculate_kl(ground_truth, other, out_dir)

    if args.wandb:
        wandb.log({"path_histogram": wandb.Image(os.path.join(out_dir, 'path_histogram.png'))})
        wandb.log({"path_histogram_non_smoothed": wandb.Image(os.path.join(out_dir, 'path_histogram_non_smoothed.png'))})
        wandb.log(kl_info)

    return out_dir


def _save_marginal_densities(run_name, ground_truth, other):
    plt.gcf().set_figwidth(15)

    plt.suptitle("Marginal Densities")

    plt.subplot(1, 2, 1)
    plt.hist(ground_truth[:, 0], bins=bins_1d, label="MD (ground truth)", density=True)
    plt.hist(other[:, 0], bins=bins_1d, label=f"Simulation ({run_name})", density=True)
    plt.xlabel(r"$\phi$")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(ground_truth[:, 1], bins=bins_1d, label="MD (ground truth)", density=True)
    plt.hist(other[:, 1], bins=bins_1d, label=f"Simulation ({run_name})", density=True)
    plt.xlabel(r"$\psi$")
    plt.legend()


def _save_path_histogram(run_name, ground_truth, other):
    #plt.suptitle("State transitions")

    # plt.gcf().set_figwidth(15)
    # plt.subplot(1, 2, 1)
    # plt.title("MD (ground truth)")
    # path_hist = PeriodicPathHistogram()
    # path_hist.add_paths(ground_truth)
    # path_hist.plot(norm=colors.LogNorm())

    # plt.subplot(1, 2, 2)
    plt.title(f"Simulation")
    path_hist = PeriodicPathHistogram()
    path_hist.add_paths(other)
    path_hist.plot(norm=colors.LogNorm())

def _save_path_histogram_non_smoothed(run_name, ground_truth, other):
    #plt.suptitle("State transitions")

    #plt.gcf().set_figwidth(15)
    #plt.subplot(1, 2, 1)
    #plt.title("MD (ground truth)")
    #plt.hist2d(np.concatenate(ground_truth)[:, 0], np.concatenate(ground_truth)[:, 1], norm=colors.LogNorm(), bins=150)
    #plt.xlim(-np.pi, np.pi)
    #plt.ylim(-np.pi, np.pi)
    #plt.xlabel(r"$\phi$")
    #plt.ylabel(r"$\psi$")

    #plt.subplot(1, 2, 2)
    plt.title(f"Simulation (non-smoothed)")
    plt.hist2d(np.concatenate(other)[:, 0], np.concatenate(other)[:, 1], norm=colors.LogNorm(), bins=150)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\psi$")

def _calculate_kl(ground_truth, other, out_dir):
    kl_info = {}
    kl_info |= {'kl_divergence_phi': kl_divergence(ground_truth[:, 0], other[:, 0], bins_1d)}
    kl_info |= {'kl_divergence_psi': kl_divergence(ground_truth[:, 1], other[:, 1], bins_1d)}
    logger.info(f"KL divergence (phi): {kl_info['kl_divergence_phi']}")
    logger.info(f"KL divergence (psi): {kl_info['kl_divergence_psi']}")

    kl_info |= {'kl_divergence2d_nn': kl_divergence2d(ground_truth, other)}
    kl_info |= {'kl_divergence2d_hist': kl_divergence2d(ground_truth, other, bins_2d)}
    kl_info |= {'phi_psi_hist_abs_dist': hist_abs_dist(ground_truth, other, bins_2d)}
    logger.info(f"KL divergence (2d, nn): {kl_info['kl_divergence2d_nn']}")
    logger.info(f"KL divergence (2d, hist): {kl_info['kl_divergence2d_hist']}")
    logger.info(f"2d histogram absolute distance: {kl_info['phi_psi_hist_abs_dist']}")

    df = pd.DataFrame(kl_info, index=[0])
    df.to_csv(os.path.join(out_dir, 'kl_divergence.csv'), index=False)

    return kl_info
