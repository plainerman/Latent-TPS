import os

from numpy.linalg import LinAlgError
import torch, yaml
import numpy as np
from .logging import get_logger
from scipy.spatial.transform import Rotation as R

logger = get_logger(__name__)


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_scheduler(args, optimizer):
    if args.warmup_dur == 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1., total_iters=args.warmup_dur)
    else:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1. / args.warmup_dur, end_factor=1., total_iters=args.warmup_dur)

    if args.lr_schedule == 'constant':
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)
    elif args.lr_schedule == 'cosine':
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.constant_dur)

    decay = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=args.lr_end, total_iters=args.decay_dur)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, lr_schedule, decay],
                                                 milestones=[args.warmup_dur, args.warmup_dur + args.constant_dur])

def get_flow_optimizer(args, flow):
    optimizer = torch.optim.Adam([{'params': flow.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},])
    return optimizer

def compute_ess(log_ws):
    log_ws = np.array(log_ws)
    log_ws = log_ws - np.max(log_ws)
    ws = np.exp(log_ws)
    return np.sum(ws) ** 2 / np.sum(ws ** 2) / len(ws)


def find_rigid_alignment(A, B):
    """
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def rmsdalign(X, Y, align=True): # align X to Y
    X = X-X.mean(0); YY = Y-Y.mean(0)
    if align:
        RR = R.align_vectors(X, YY)[0]
        X = X @ RR.as_matrix()
    return X + Y.mean(0)

def rmsd(X, Y, align=True):
    try:
        X = rmsdalign(X, Y, align=align)
        return ((Y-X)**2).sum(-1).mean()**0.5
    except LinAlgError as e:
        print('LinAlgError', e)
        return np.inf

class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]
