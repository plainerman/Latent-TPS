import os
import numpy as np
import torch
from scipy.spatial import KDTree
from datasets.single_mol_dataset import SingleMolDataset
from utils.model import construct_model
from utils.parsing import load_train_args

def hist_abs_dist(samples_p, samples_q, bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]]):
    Hp, xedges, yedges = np.histogram2d(*samples_p.T, bins=bins, range=range, density=True)
    Hq, _, _ = np.histogram2d(*samples_q.T, bins=(xedges, yedges), density=True)

    return np.sum(np.abs(Hp - Hq))

# Taken from https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
def kl_divergence2d(samples_p, samples_q, bins=None, range=[[-np.pi, np.pi], [-np.pi, np.pi]]):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    samples_p : 2D array (n, d)
      Samples from distribution P, which typically represents the true distribution.
    samples_q : 2D array (m, d)
      Samples from distribution Q, which typically represents the approximate distribution.
    bins: int
      Number of bins for the density estimation with the histogram. If this parameter
      is specified, histogram approximation of the density will be used instead of
      nearest neighbor.
    range: [[xmin, xmax], [ymin, ymax]]
      The range for the histograms if bins are spefcified.
    Returns
    -------
    out : float
      The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    For nearest neighbor based KL divergence estimation:
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of continuous distributions
    IEEE International Symposium on Information Theory, 2008.
    """

    # Check the dimensions are consistent
    samples_p = np.atleast_2d(samples_p)
    samples_q = np.atleast_2d(samples_q)

    n, d = samples_p.shape
    m, dy = samples_q.shape

    assert (d == dy)

    if bins is not None:
        assert d == 2, "Histogram KL-Divergence estimation only works in 2D"

        Hp, xedges, yedges = np.histogram2d(*samples_p.T, bins=bins, range=range, density=True)
        Hq, _, _ = np.histogram2d(*samples_q.T, bins=(xedges, yedges), density=True)
        xdist, ydist = xedges[1] - xedges[0], yedges[1] - yedges[0]

        kl = Hp * (np.log(Hp) - np.log(Hq)) * xdist * ydist
        return np.sum(kl[np.isfinite(kl)])

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(samples_p)
    ytree = KDTree(samples_q)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(samples_p, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(samples_p, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    kl = np.log(s) - np.log(r)
    n_eff = np.isfinite(kl).sum()
    return kl[np.isfinite(kl)].sum() * d / n_eff + np.log(m / (n_eff - 1.))


def load_flow_and_args(model_dir, ckpt, **override_args):
    args = load_train_args(os.path.join(model_dir, 'args.yaml'))
    args.__dict__["batch_size"] = 4096
    args.__dict__["wandb"] = False

    for arg_key, arg_val in override_args.items():
        args.__dict__[arg_key] = arg_val

    dataset = SingleMolDataset(args)

    flow = construct_model(args, dataset)
    state_dict = torch.load(os.path.join(model_dir, ckpt), map_location=torch.device(args.__dict__["torch_device"]))
    flow.load_state_dict(state_dict['model'], strict=True)
    flow.to(args.torch_device)
    flow.eval()

    return flow, args
