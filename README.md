# Transition Path Sampling with Boltzmann Generator-based MCMC Moves
[![arXiv](https://img.shields.io/badge/arXiv-2312.05340-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.05340)
[![python](https://img.shields.io/badge/language-python%20-%2300599C.svg?style=flat-square)](https://github.com/plainerman/Latent-TPS)
[![License](https://img.shields.io/github/license/plainerman/Latent-TPS?style=flat-square)](LICENSE)

## Setting up the conda environment
We provide a conda environment file for CUDA and CPU. You can create it by using one of the files (with or without the cpu flag).

```bash
conda env create -f environment[-cpu].yml
```

## Training the Boltzmann Generator
To train the Boltzmann Generator, you can use the `train.py` script. It has a number of options, but for ALDP you can train the model like this:

```bash
python -m train --system AlanineDipeptideImplicit --data_save_frequency 120 --num_frames 1000000 --print_freq 250 --ckpt_freq 250 --val_freq 250 --flow_type internal_coords --batch_size 1024 --lr 5.e-4 --weight_decay 1.e-5 --lr_schedule cosine --warmup_dur 1000 --grad_clip 1000 --kl_loss_weight 1 --rkl_loss_weight 0 --hidden_dim 256 --update_layers 12 --run_name ALDP_RKL0_KL1_h256_u12_warmup_lrcosine_rerun
```

If you are working with cuda, you can add the flags

```bash
--torch_device cuda --md_device CUDA
```

## Run Latent TPS
You can find all the different options in `inference.py`. 
You can change the states to find paths between by changing the `--start_state_idx` and `--end_state_idx` flags.

Here is an example using the gaussian kernel, which adds random gaussian noise to the frames in latent space.
```bash
python -m inference --run_name mcmc_prob_langevin_40_noise0.05_seed0 --sampling_method mcmc --model_dir ./workdir/best --ckpt model_4250.ckpt --path_density langevin --noise_scale 0.05 --num_steps 40 --langevin_timestep 40 --num_paths 100 --seed 0
```

# Acknowledgements
We thank the authors of [Flow Annealed Importance Sampling Bootstrap](https://github.com/lollcat/fab-torch) and [normflows](https://github.com/VincentStimper/normalizing-flows) which our flow training uses.