import os
import time
import traceback
import wandb

import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
import mdtraj
import tqdm
import copy

from model.utils import log_mean_exp
from utils.flow_iters import iteration, eval_flow

from utils.model import construct_model
from utils.parsing import parse_train_args

from utils.training import get_flow_optimizer, save_yaml_file, get_scheduler
from collections import defaultdict

from datasets.single_mol_dataset import SingleMolDataset, save_pdb_transition
from utils.logging import get_logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = get_logger(__name__)


def main(args):
    model_dir = os.environ['MODEL_DIR']
    yaml_file_name = os.path.join(model_dir, 'args.yaml')
    save_yaml_file(yaml_file_name, args.__dict__)
    logger.info(f"Saving training args to {yaml_file_name}")

    dataset = SingleMolDataset(args)
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.train_iters * args.batch_size)
    loader = DataLoader(dataset=dataset, sampler=sampler, num_workers=args.num_workers, batch_size=args.batch_size)
    flow = construct_model(args, dataset)

    if args.ckpt:
        state_dict = torch.load(os.path.join(model_dir, args.ckpt), map_location=torch.device('cpu'))
        flow.load_state_dict(state_dict['model'], strict=True)
        logger.info(f'Loading checkpoint {args.ckpt}')

    flow.to(args.torch_device)

    numel = sum([p.numel() for p in flow.parameters()])
    logger.info(f'Model with {numel} parameters')

    if args.wandb:
        wandb.init(
            entity='coarse-graining-mit',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})
        if args.debug:
            wandb.watch(flow)

    with torch.no_grad():
        data = next(iter(loader))
        data.to(args.torch_device)
        curr_x = data.pos
        if args.flow_type == 'internal_coords': curr_x = flow.internal_prior.sample(args.batch_size)
        if not args.ckpt: flow.check_invertible(data, curr_x)

    optimizer = get_flow_optimizer(args, flow)
    scheduler = get_scheduler(args, optimizer)
    train_flow(args, dataset, flow, loader, optimizer, scheduler)


def train_flow(args, dataset, flow, loader, optimizer, scheduler):
    logs = defaultdict(list)
    data_iterator = iter(loader)

    consecutive_errors = 0

    iterations = tqdm.trange(args.train_iters) if not args.wandb else range(args.train_iters)
    last_successful_state = None
    for i in iterations:
        try:
            flow.train()
            start = time.time()
            data = next(data_iterator)
            logs['data_load_time'].append(time.time() - start)
            data.to(args.torch_device)

            if args.flow_type == 'internal_coords':
                prior_x = flow.internal_prior.sample(args.batch_size)
            else:
                prior_x = data.pos

            target_x = data.target_pos
            logs['num_atoms'].append(data.num_nodes / data.num_graphs)
            logs['num_torsions'].append(data.edge_mask.sum().item() / data.num_graphs)
            optimizer.zero_grad()
            loss, bg_target_x = iteration(args, flow, data, prior_x, target_x, dataset, logs, i)

            if torch.isinf(loss).any():
                logger.warning('Loss contained inf')
                logger.info(loss)
            if torch.isnan(loss).any():
                logger.warning('Loss contained nan')
                logger.info(loss)

            start = time.time()
            loss.nanmean().backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            logs['backprop_time'].append(time.time() - start)
            if i % args.val_freq == 0:
                flow.eval()
                eval_flow(args, data, dataset, flow, i, logs)
                flow.train()

            if i % args.ckpt_freq == 0:
                state = {'model': flow.state_dict()}
                path = f'{args.log_dir}/{args.run_name}/model_{i}.ckpt'
                torch.save(state, path)
                logger.info(f'Saved checkpoint {path}')

            if i % args.print_freq == 0:
                log = {key: np.nanmean(logs[key]) for key in logs}
                log |= {'iter': i}
                log |= {'sample_rmsd.std': np.std(logs['sample_rmsd'])}
                log |= {'mean_weights': log_mean_exp(logs['log_w'])}
                logger.info(str(log))
                logger.info(f"reverse_kl {log['reverse_kl']}")
                logger.info(f"forward_kl {log['forward_kl']}")
                logger.info(
                    f'Run name: {args.run_name}')  # please keep this. I have many tmux windows and do not know what is running where sometimes :3
                if args.wandb: wandb.log(log)
                logs = defaultdict(list)

            last_successful_state = copy.deepcopy(flow.state_dict())
            consecutive_errors = 0
        except Exception as e:
            logger.warning(f'An error was encountered in iteration {i}')
            logger.warning(str(e))
            logger.info(traceback.format_exc())

            i -= 1  # try again

            consecutive_errors += 1
            if consecutive_errors == 10 and last_successful_state is not None:
                logger.warning('Too many consecutive errors. Restoring last successful state.')
                flow.load_state_dict(last_successful_state, strict=True)
            if consecutive_errors > 100:
                logger.error('Too many consecutive errors. Exiting.')
                raise e


if __name__ == '__main__':
    args = parse_train_args()
    main(args)
