import copy


import normflows as nf
import numpy as np
import torch

import torch.nn as nn
from boltzgen import CoordinateTransform

import datasets.single_mol_dataset as single_mol
from model.dist import UniformGaussianFlow

from utils.training import rmsdalign, rmsd

z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
cart_indices = [8, 6, 14]

ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]

default_std = {'bond': 0.005, 'angle': 0.15, 'dih': 0.2}


class InternalFlow(nn.Module):
    def __init__(self, args, dataset: single_mol.SingleMolDataset):
        super().__init__()

        self.args = args
        self.dataset = dataset

        dim = dataset.reference_frame.shape[0] * 3
        coordinate_transform = CoordinateTransform(torch.from_numpy(dataset.reference_frame).view(-1, dim), dim,
                                                        z_matrix, cart_indices, mode='internal',
                                                        ind_circ_dih=ind_circ_dih, shift_dih=False,
                                                        default_std=default_std)
        ncarts = coordinate_transform.transform.len_cart_inds
        permute_inv = coordinate_transform.transform.permute_inv.cpu().numpy()
        dih_ind_ = coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
        std_dih = coordinate_transform.transform.ic_transform.std_dih.cpu().numpy()

        ndim = 60
        ind = np.arange(ndim, dtype=int)
        ind = np.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=int), ind[3 * ncarts - 6:]], dtype=int)
        ind = ind[permute_inv]
        dih_ind = ind[dih_ind_]

        ind_circ = dih_ind[ind_circ_dih]
        bound_circ = np.pi / std_dih[ind_circ_dih]
        tail_bound = 5. * np.ones(ndim)
        tail_bound[ind_circ] = bound_circ
        base_scale = np.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2

        base_scale = torch.from_numpy(base_scale)
        base_scale = base_scale.double() if args.double_precision else base_scale.float()
        ind_circ = torch.from_numpy(ind_circ)
        bound_circ = torch.from_numpy(bound_circ)
        tail_bound = torch.from_numpy(tail_bound)
        tail_bound = tail_bound.double() if args.double_precision else tail_bound.float()

        flows = []
        if args.base_dist == 'gauss':
            self.internal_prior = nf.distributions.base.DiagGaussian(ndim, trainable=False)
            flows.append(UniformGaussianFlow(ndim, ind_circ, scale=base_scale))
        elif args.base_dist == 'gauss-uni':
            self.internal_prior = nf.distributions.base.UniformGaussian(ndim, ind_circ, scale=base_scale)

        if args.flow_architecture == 'rnvp':
            # RNVP is not circular, so we ensure that uniformly distributed variables are in the correct range
            flows.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))

        for i in range(args.update_layers):
            if args.flow_architecture == 'rnvp':
                # Coupling layer
                hl = args.rnvp_hidden_layers * [args.hidden_dim]
                scale_map = args.rnvp_scale_map
                scale = scale_map is not None
                if scale_map == 'tanh':
                    output_fn = 'tanh'
                    scale_map = 'exp'
                else:
                    output_fn = None
                param_map = nf.nets.MLP([(ndim + 1) // 2] + hl + [(ndim // 2) * (2 if scale else 1)], init_zeros=False,
                                        output_fn=output_fn)
                flows.append(nf.flows.AffineCouplingBlock(param_map, scale=scale, scale_map=scale_map))
            elif args.flow_architecture == 'circular-coup-nsf':
                if i % 2 == 0:
                    mask = nf.utils.masks.create_random_binary_mask(ndim, seed=i)
                    assert not (mask == 1).all(), "I am not sure about that assert, but I think it can break with it"
                else:
                    mask = 1 - mask
                flows.append(nf.flows.CircularCoupledRationalQuadraticSpline(ndim, 1, args.hidden_dim, ind_circ, tail_bound=tail_bound,
                                                                             num_bins=8, init_identity=True,
                                                                             dropout_probability=0.0, mask=mask))
            elif args.flow_architecture == 'circular-ar-nsf':
                flows.append(nf.flows.CircularAutoregressiveRationalQuadraticSpline(ndim,
                                                                                     1, args.hidden_dim, ind_circ,
                                                                                     tail_bound=tail_bound, num_bins=8,
                                                                                     permute_mask=True,
                                                                                     init_identity=True,
                                                                                     dropout_probability=0.0))

            else:
                raise NotImplementedError(f'{args.flow_architecture} is not implemented')

            if i % 2 == 1 and i != args.update_layers - 1:
                # circ_shift == 'random':
                gen = torch.Generator().manual_seed(i)
                shift_scale = torch.rand([], generator=gen) + 0.5
                flows.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ, shift=shift_scale * bound_circ))

        flows.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))

        flows.append(coordinate_transform)
        self._flows = flows
        self.flow = nf.NormalizingFlow(q0=self.internal_prior, flows=flows, p=dataset.target.boltzmann_distribution)

    def forward(self, data, prior_x, logs=None):
        target_x, log_det = self.flow.forward_and_log_det(prior_x)
        return target_x, log_det

    def reverse(self, data, target_x, logs=None):
        target_x = target_x.view(self.args.batch_size, -1)
        x, log_det = self.flow.inverse_and_log_det(target_x)
        return x, log_det

    def device(self):
        return next(self.flow.parameters()).device

    def check_invertible(self, data, prior_x):
        self.eval()
        test_copy = copy.deepcopy(prior_x)

        # We transform it twice because the PeriodicWrap changes the values slightly (i.e., modulo ranges)
        # But when using it again, it should result in the same values
        new_x, _ = self.forward(data, prior_x)
        back_x1, _ = self.reverse(data, new_x)
        new_x, _ = self.forward(data, back_x1)
        back_x, _ = self.reverse(data, new_x)
        assert torch.allclose(prior_x, test_copy)
        assert torch.allclose(back_x1, back_x, rtol=1e-04, atol=1e-02)
        for x, y in zip(new_x.view(prior_x.shape[0], -1, 3), new_x.view(prior_x.shape[0], -1, 3)):
            assert torch.allclose(x.detach().cpu().float(), rmsdalign(y.detach().cpu(), x.detach().cpu(), align=True).float(), rtol=1e-04, atol=1e-04)

    def as_mixed_flow(self):
        """
        Returns this flow as one with a UniformGaussian base distribution.
        Note that some data is shared, so this is not a deepcopy.
        """
        if self.args.base_dist == 'gauss-uni':
            return self

        assert isinstance(self._flows[0], UniformGaussianFlow)

        args = copy.deepcopy(self.args)
        args.base_dist = 'gauss-uni'

        flow = InternalFlow(args, self.dataset)

        flow._flows = self._flows[1:]
        flow.flow = nf.NormalizingFlow(q0=flow.internal_prior, flows=flow._flows, p=flow.dataset.target.boltzmann_distribution)

        return flow

    def as_normal_flow(self):
        """
        Returns this flow as one with a Gaussian base distribution.
        Note that some data is shared, so this is not a deepcopy.
        """
        if self.args.base_dist == 'gauss':
            return self

        assert not isinstance(self._flows[0], UniformGaussianFlow)

        args = copy.deepcopy(self.args)
        args.base_dist = 'gauss'

        flow = InternalFlow(args, self.dataset)
        layers = [
            UniformGaussianFlow(self.internal_prior.ndim,
                                self.internal_prior.ind,
                                self.internal_prior.scale)
        ]
        layers.extend(self._flows)
        flow._flows = layers
        flow.flow = nf.NormalizingFlow(q0=flow.internal_prior, flows=flow._flows, p=flow.dataset.target.boltzmann_distribution)

        return flow
