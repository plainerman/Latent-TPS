import sys, subprocess, os
from argparse import ArgumentParser, FileType
import yaml


def load_train_args(path):
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)

    args = parse_train_args([])
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value

    if 'base_dist' not in config_dict:
        args.base_dist = 'gauss-uni'

    return args


def parse_train_args(args=sys.argv[1:]):

    parser = ArgumentParser()
    # Target system
    parser.add_argument('--system', type=str, default='AlanineDipeptideImplicit', help='Name of openmmtools.testsystem OR path to PDB file OR 4AA')
    parser.add_argument('--forcefield', type=str, default='amber14-all.xml',help='Forcefield')
    parser.add_argument('--forcefield_water', type=str, default='implicit/gbn.xml', help='Forcefield water')
    parser.add_argument('--temp', type=float, default=300, help='Boltzmann distribution temperature (K)')
    parser.add_argument('--torch_device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--md_device', type=str, choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], default='CPU')

    ## Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_schedule', type=str, choices=['constant', 'cosine'], default='constant', help='The type of learning rate schedule used after warmup')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--val_freq', type=float, default=5, help='Frequency of validation iters')
    parser.add_argument('--val_samples', type=int, default=100_000, help='The number of samples to draw when evaluating the model')
    parser.add_argument('--warmup_dur', type=float, default=0)
    parser.add_argument('--constant_dur', type=float, default=5e9)
    parser.add_argument('--decay_dur', type=float, default=0)
    parser.add_argument('--lr_end', type=float, default=1)
    parser.add_argument('--train_iters', type=int, default=1000000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument("--energy_log_cutoff", type=float, default=1.e+8, help='')
    parser.add_argument("--energy_max", type=float, default=1.e+20, help='')
    parser.add_argument('--kl_loss_weight', type=float, default=0)
    parser.add_argument('--rkl_loss_weight', type=float, default=0)
    parser.add_argument('--rkl_start_iter', type=int, default=0, help='Iteration at which to start training with RKL')
    parser.add_argument('--gaussian_target', action='store_true', default=False)
    parser.add_argument('--double_precision', action='store_true', default=False)

    ##  dataset
    parser.add_argument('--data_path', type=str, default='data/flow_datasets', help='Path to a flow dataset')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--parallel_energy', action='store_true', default=False)
    parser.add_argument('--num_energy_processes', type=int, default=None, help='Number of processes to use for energy evaluation')
    parser.add_argument('--num_frames', type=int, default=10, help='')
    parser.add_argument('--warmup_steps', type=int, default=3, help='')
    parser.add_argument('--data_save_frequency', type=int, default=10, help='In femtoseconds')
    parser.add_argument('--sampling_time_delta', type=int, default=30, help='Time in femtoseconds')
    parser.add_argument('--max_number_of_frames', type=int, default=1000000, help='Bat')
    parser.add_argument('--quadrants_to_remove', type=list, default=None, help='List of dihedral angle quadrants to remove from the dataset. For example, 12 will remove all points in the first and second phi psi space quadrant')
    parser.add_argument('--node_embeddings', action='store_true', default=False, help='learn node embeddings')
    parser.add_argument('--edge_embeddings', action='store_true', default=False, help='learn edge embeddings')
    parser.add_argument('--edges', type=str, choices=['dense', 'radius'], default='radius')

    ## Logging
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--project', type=str, default='transitionpath', help='')
    parser.add_argument('--run_name', type=str, default='default', help='Name that will be used to save the model in log_dir and that will be used for WandB')
    parser.add_argument('--saved_dir', type=str, default=None, help='Where the saved weights are loaded from when running inference')
    parser.add_argument('--log_dir', type=str, default='workdir', help='Folder in which to save model and logs')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=1000)
    parser.add_argument('--plot_freq', type=int, default=2)
    parser.add_argument('--ckpt_freq', type=int, default=10000)
    parser.add_argument('--save_traj_freq', type=int, default=1, help='Never saves trajectory if this is 0. Frequency at which to save a genertive trajectory.')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num_hist_bins', type=int, default=50)

    ## All flows
    parser.add_argument('--flow_type', type=str, choices=['atom_coupling', 'coordinate_coupling', 'phase_space', 'internal_coords'], default='radius')
    parser.add_argument('--update_layers', type=int, default=3)
    parser.add_argument('--no_node_features', action='store_true', default=False)
    parser.add_argument('--pca_whiten', action='store_true', default=False)
    parser.add_argument('--remove_dof', type=int, default=6)
    parser.add_argument('--prior_std', type=float, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--scale_activation', type=str, default=None, help='Activation function for the scale. If this is none then the normal activation is used')

    ## Coordinate coupling Flow
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='Number of hidden features per node of order 0')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Number of hidden features per node of order 0')
    parser.add_argument('--embed_dim', type=int, default=20, help='Number of hidden features per node of order 0')
    parser.add_argument('--equivariant_coupling', action='store_true', default=False)
    parser.add_argument('--partitions', type=str, nargs='+', default=['0', '01','12','02'])
    parser.add_argument('--conv_layers', type=int, default=2)
    parser.add_argument('--scale', action='store_true', default=False)
    parser.add_argument('--scale_type', type=str, choices=['exp', 'sigmoid', 'sigmoid_inv'],default='exp')

    ## Augmented Flow
    parser.add_argument('--torsional_update_layers', type=int, default=0)
    parser.add_argument('--velocity_conv_layers', type=int, default=2)
    parser.add_argument('--embed_conv_layers', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--initial_vel_coefficient', type=float, default=0)
    parser.add_argument('--ignore_torsional_jac', action='store_true', default=False)

    ## e3nn
    parser.add_argument('--order', type=int, choices=[1, 2], default=1)
    parser.add_argument('--sh_lmax', type=int, default=1)
    parser.add_argument('--ns', type=int, default=32)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--radius_emb_type', type=str, choices=['sinusoidal', 'gaussian', 'bessel'], default='gaussian')
    parser.add_argument('--radius_emb_dim', type=int, default=50)
    parser.add_argument('--radius_emb_max', type=float, default=10, help='angstroms')
    parser.add_argument('--radius_max', type=float, default=5, help='angstroms')
    parser.add_argument('--use_fixed_bessel', action='store_true', default=False, help='Train the :math:`n \pi` part or not.')
    parser.add_argument('--ntps', type=int, default=16)
    parser.add_argument('--ntpv', type=int, default=4)
    parser.add_argument('--fc_dim', type=int, default=128)
    parser.add_argument('--lin_self', action='store_true', default=False)
    parser.add_argument('--parity', action='store_true', default=True)
    parser.add_argument('--affine_transform', action='store_true', default=False)
    parser.add_argument('--differentiate_convolution', action='store_true', default=False)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--scalar_transformation', action='store_true', default=False)
    parser.add_argument('--tp_final_sigmoid', action='store_true', default=False)


    # Internal Coordinate Flow
    parser.add_argument('--flow_architecture', type=str, choices=['rnvp', 'circular-coup-nsf', 'circular-ar-nsf'], default='circular-coup-nsf')
    parser.add_argument('--rnvp_hidden_layers', type=int, default=3)
    parser.add_argument('--rnvp_scale_map', type=str, default=None)
    parser.add_argument('--base_dist', type=str, choices=['gauss', 'gauss-uni'], default='gauss')


    args = parser.parse_args(args)
    # args.train_tmin = args.tmin # backwards compatibility
    
    model_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.environ['MODEL_DIR'] = model_dir
    if args.debug:
        os.environ['LOGGER_LEVEL'] = 'debug'
    
    return args