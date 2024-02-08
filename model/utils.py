import numpy as np
import torch.nn as nn
import math, torch
import torch.nn.functional as F


def pca(X0, keepdims=None, device='cpu'):
    if keepdims is None:
        keepdims = X0.shape[1]
    # pca
    X0mean = X0.mean(axis=0)
    X0meanfree = X0 - X0mean
    C = np.dot(X0meanfree.T, X0meanfree) / (X0meanfree.shape[0] - 1.0)
    eigval, eigvec = np.linalg.eigh(C)
    # sort in descending order and keep only the wanted eigenpairs
    I = np.argsort(eigval)[::-1]
    I = I[:keepdims]
    eigval = eigval[I]
    std = np.sqrt(eigval)
    eigvec = eigvec[:, I]
    # whiten and unwhiten matrices
    X0mean = torch.from_numpy(X0mean).to(device)
    Twhiten = torch.from_numpy(eigvec.dot(np.diag(1.0 / std))).to(device)
    Tblacken = torch.from_numpy(np.diag(std).dot(eigvec.T)).to(device)
    return X0mean, Twhiten, Tblacken, std

def log_mean_exp(x):
    max_val = np.max(x)
    return max_val + np.log(np.mean(np.exp(x - max_val)))
class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(Encoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(x, embedding_dim, max_positions=embedding_scale))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).float()

class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        Parameters
        ----------
        r_max : float
            Cutoff radius
        num_basis : int
            Number of Bessel Basis functions
        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))
