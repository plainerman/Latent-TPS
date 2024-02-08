import numpy as np
from utils.logging import get_logger
import torch, time
from torch_scatter import scatter
import tqdm

logger = get_logger(__name__)

def eval_flow(args, data, dataset, flow, i, logs):
    phis = np.zeros(args.val_samples)
    psis = np.zeros(args.val_samples)
    with torch.no_grad():
        for j in tqdm.trange(0, args.val_samples, args.batch_size):
            count = args.batch_size
            if j + args.batch_size >= args.val_samples:
                count = args.val_samples - j

            # we give it 10 tries to generate samples
            for k in range(10):
                try:
                    if args.flow_type == 'internal_coords':
                        prior_x = flow.internal_prior.sample(count)
                    else:
                        prior_x = data.pos

                    zs, _ = flow.forward(data, prior_x)
                    a, b = dataset.phis_psis(zs)
                    phis[j:j + count] = a
                    psis[j:j + count] = b

                    break
                except Exception as e:
                    logger.warning(f'An error was encountered in evaluation of iteration {i}')
                    logger.warning(str(e))

                    phis[j:j + count] = np.nan
                    psis[j:j + count] = np.nan

            if args.flow_type != 'internal_coords':
                logger.error('eval does not work for properly non-internal coords flows. Only drawing one batch')
                break

    # Filter out the values where the forward pass failed
    phis = phis[~np.isnan(phis)]
    psis = psis[~np.isnan(psis)]

    assert phis.shape == psis.shape

    if len(phis) != args.val_samples:
        logger.warning(f'Only {len(phis)} samples were generated instead of {args.val_samples}')

    from utils.plotting import save_ramachandran_plot, save_histogram

    save_ramachandran_plot(phis, psis, i, args, name='prop')
    save_histogram(logs, i, args, key='sample_rmsd')

    return phis, psis


def run_flow(args, flow, data, prior_x, target_x, dataset, logs, ignore_torsional_jac=False):
    start = time.time()
    bg_prior_x, bg_logjac_reverse = flow.reverse(data, target_x, logs=logs)
    bg_target_x, bg_logjac_forward = flow.forward(data, prior_x, logs=logs)
    logs['flow_forward_backward_time'].append(time.time() - start)
    start = time.time()
    system_dim = dataset.target.topology.getNumAtoms() * 3
    prior_log_partition = -torch.log(2*torch.tensor(torch.pi,device=args.torch_device)) * system_dim / 2 - system_dim * np.log(args.prior_std)
    target_log_partition = -torch.log(2*torch.tensor(torch.pi,device=args.torch_device)) * system_dim / 2

    with torch.no_grad():
        target_logPE = -scatter(torch.square(target_x).sum(-1), data.batch) / 2 + target_log_partition  if args.gaussian_target else dataset.target.log_prob_A(target_x.view(args.batch_size, -1))
        prior_logPE = -scatter(torch.square(prior_x).sum(-1), data.batch) / 2 / args.prior_std ** 0.5 + prior_log_partition if not args.flow_type == 'internal_coords' else flow.flow.q0.log_prob(prior_x)
    logs['energy_time'].append(time.time() - start)
    bg_target_logPE = -scatter(torch.square(bg_target_x).sum(-1), data.batch) / 2 + target_log_partition if args.gaussian_target else dataset.target.log_prob_A(bg_target_x)
    bg_prior_logPE = -scatter(torch.square(bg_prior_x).sum(-1), data.batch) / 2 / args.prior_std ** 0.5 + prior_log_partition if not args.flow_type == 'internal_coords' else flow.flow.q0.log_prob(bg_prior_x)

    logs[f'bg_logjac_reverse'].extend(bg_logjac_reverse.detach().cpu().numpy().tolist())
    logs[f'bg_logjac_forward'].extend(bg_logjac_forward.detach().cpu().numpy().tolist())
    logs[f'bg_target_logPE'].extend(bg_target_logPE.detach().cpu().numpy().tolist())
    logs[f'bg_prior_logPE'].extend(bg_prior_logPE.detach().cpu().numpy().tolist())
    logs[f'target_logPE'].extend(target_logPE.cpu().numpy().tolist())
    logs[f'prior_logPE'].extend(prior_logPE.cpu().numpy().tolist())

    # forward KL = - target energy + prior energy - log_jac_reverse
    forward_kl = target_logPE - bg_prior_logPE - bg_logjac_reverse # minus for the bg_prior_logPE because it is the negative energy and here we want the energy

    # Check if our custom implementation of the KL divergence is correct
    # We do not check target_logPE because it does not have a gradient and thus does not contribute to the loss
    # TODO: I think this assert only makes sense if we have flow.flow (so for an internal coordinate flow, maybe the others named it differently)
    assert torch.allclose((-bg_prior_logPE - bg_logjac_reverse).mean().float(), flow.flow.forward_kld(target_x.view(args.batch_size, -1)).float(), rtol=1e-04, atol=1e-04)

    # reverse KL = - prior energy + target energy - log_jac_forward
    reverse_kl = prior_logPE - bg_target_logPE - bg_logjac_forward # - bg_logjac_forward + prior_logPE/logKE/logtKE is q(x) here

    # we cannot assert that our reverse_kl is equal to the one from use flow.flow.reverse_kld for assertion because they sample new values

    log_w = bg_target_logPE - (bg_logjac_forward + prior_logPE) # log_w = log_p(x) - log_qq(x) = log_p(x) - (log_U(prior_x) + log_jac)
    logs['log_w'].extend(log_w.detach().cpu().numpy().tolist())

    return reverse_kl, forward_kl, bg_target_x

def iteration(args, flow,data, prior_x, target_x, dataset, logs, iter):

    start = time.time()
    reverse_kl, forward_kl, bg_target_x = run_flow(args, flow, data, prior_x, target_x, dataset, logs)
    logs['run_flow_time'].append(time.time() - start)

    with torch.no_grad():
        fkl = forward_kl.cpu().numpy().tolist()
        rkl = reverse_kl.cpu().numpy().tolist()
        logger.debug(f'forward_kl: ' + str(fkl))
        logger.debug(f'reverse_kl: ' + str(rkl))
        logs[f'forward_kl'].extend(fkl)
        logs[f'reverse_kl'].extend(rkl)

    loss = torch.zeros(1, device=args.torch_device, requires_grad=True)
    if args.kl_loss_weight > 0:
        loss = loss + args.kl_loss_weight * forward_kl
    if args.rkl_loss_weight > 0 and iter > args.rkl_start_iter:
        loss = loss + args.rkl_loss_weight * reverse_kl
    return loss, bg_target_x


