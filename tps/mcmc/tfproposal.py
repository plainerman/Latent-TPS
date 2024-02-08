from model.internal_flow import InternalFlow
from tps.mcmc.proposal import ProposalKernel
import tensorflow_probability as tfp
import torch
import tensorflow as tf


class LatentHamiltonianProposal(ProposalKernel):
    """Frame independent Hamiltonian Dyanmics proposal kernel.
        As implemented by NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport
        https://arxiv.org/pdf/1903.03704.pdf
        Also, see: https://arxiv.org/pdf/2006.16653.pdf, Appendix B.8
    """
    def __init__(self, flow: InternalFlow, step_size=0.1, num_leapfrog_steps=2, simultaneous_proposals=1, seed=None):
        super().__init__()
        self.flow = flow.flow

        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.simultaneous_proposals = simultaneous_proposals
        self.seed = seed

        self.device = flow.device()

    def _setup(self, path):
        # we store the inverse because it is expensive
        z_path, log_det_x = self.flow.inverse_and_log_det(path.to(self.device))

        if torch.isnan(z_path).any() or torch.isnan(log_det_x).any():
            raise RuntimeError("NaN encountered in LatentNoiseProposal._setup. Cannot create state.")

        kernel = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=tfp.distributions.MultivariateNormalDiag(loc=[0] * z_path.shape[-1],
                                                                        scale_diag=[1] * z_path.shape[-1]).log_prob,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            state_gradients_are_stopped=False,
            store_parameters_in_results=False,
            experimental_shard_axis_names=None,
            name=None
        )

        return tf.convert_to_tensor(z_path.cpu().detach()), log_det_x, kernel

    def _propose(self, state, path: torch.Tensor):
        z_path, log_det_x, kernel = state

        # Repeat so that we have [z_path, z_path, ..., z_path] with shape (simultaneous_proposals * path_len, ndim)
        repeated_z_path = tf.tile(z_path, (self.simultaneous_proposals, 1))
        initial_previous_result = kernel.bootstrap_results(repeated_z_path)
        z_proposed, result = kernel.one_step(repeated_z_path, initial_previous_result, seed=self.seed)

        z_proposed = torch.from_numpy(z_proposed.numpy()).to(self.device)
        new_paths, new_log_det_z = self.flow.forward_and_log_det(z_proposed)
        # Simply log(p(momentum_after) / p(momentum_before)), where p is a standard normal distribution
        # It is independent between dimensions and frames
        acceptance_ratio = torch.from_numpy(result.log_acceptance_correction.numpy()).to(self.device)

        new_paths = new_paths.view(self.simultaneous_proposals, -1, new_paths.shape[-1])
        new_log_det_z = new_log_det_z.view(self.simultaneous_proposals, -1)
        acceptance_ratio = acceptance_ratio.view(self.simultaneous_proposals, -1)

        return [(path, (ratio + log_det_z + log_det_x).sum()) for path, log_det_z, ratio in
                zip(new_paths, new_log_det_z, acceptance_ratio)]
