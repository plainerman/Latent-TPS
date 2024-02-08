import matplotlib, os, time, wandb
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl

import tps.states


def x_y_radius_to_dict(x, y, radius):
    """Converts a circle specified by x, y, radius to a dict containing x0, x1, y0, y1"""
    return {
        'x0': x - radius,
        'x1': x + radius,
        'y0': y - radius,
        'y1': y + radius
    }


def ramachandran_paper(phi, psi, states=None, bins=100, alpha=0.6, hist=True, path=None):
    if hist:
        import matplotlib.colors as colors
        plt.hist2d(phi, psi, bins=bins, norm=colors.LogNorm(), rasterized=True)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.gca().set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.gca().set_xticklabels([r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'])

    plt.gca().set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    plt.gca().set_yticklabels([r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'])

    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')
    plt.gca().set_aspect('equal', adjustable='box')

    if path is not None:
        dist = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
        mask = np.hstack([dist > np.pi, [False]])
        masked_path_x, masked_path_y = np.ma.MaskedArray(path[:, 0], mask), np.ma.MaskedArray(path[:, 1], mask)
        plt.plot(masked_path_x, masked_path_y, color="red")

    for state in (states if states is not None else []):
        c = plt.Circle(state.center, radius=state.radius(), edgecolor='gray', alpha=alpha, facecolor='white', ls='--',
                       lw=0.7)
        plt.gca().add_patch(c)
        plt.gca().annotate(state.name, xy=state.center, fontsize=6, ha="center", va="center")


def ramachandran_plot(phis, psis, state_info: 'list[tps.states.State]' = [], phis_t=None, psis_t=None, title=None, interactive=True, bins=100):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # We implement the hist2d ourselves, so that we can use a log color scale
    H, xedges, yedges = np.histogram2d(phis, psis, bins=bins)
    # Histogram does not follow Cartesian convention
    H = H.T

    lognorm = mpl.colors.LogNorm()
    H = lognorm(H)
    H, mask = H.data, H.mask
    H[mask] = None

    fig.add_heatmap(
        x=xedges,
        y=yedges,
        z=H,
        colorscale='Viridis',
        colorbar=dict(title='log(counts)'),
    )

    if len(state_info) > 0:
        states = []
        for i, state in enumerate(state_info):
            s = dict(type="circle",
                     xref="x", yref="y",
                     fillcolor='magenta',
                     line=dict(color="blue"),
                     label=dict(text=f"{state.name}"),
                     opacity=.8,
                     **x_y_radius_to_dict(*state.center, state.radius()))

            fig.add_shape(**s)

            states.append(s)

        if interactive:
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(label='States',
                                 method="relayout",
                                 args=["shapes", states],
                                 args2=["shapes", []]),
                        ]
                    )
                ]
            )

    width = 520
    if phis_t is not None and psis_t is not None:
        fig.add_trace(go.Scatter(x=phis_t, y=psis_t, mode='lines+markers', name='Trajectory', line=dict(color="red"),))
        width = 570

    fig.update_layout(
        title=title,
        width=width,
        height=500,
        xaxis_title=r"$\phi$",
        yaxis_title=r"$\psi$",
        xaxis_range=[-np.pi, np.pi],
        yaxis_range=[-np.pi, np.pi],
        showlegend=True,
        legend={"yanchor": "bottom"}
    )

    return fig


def save_ramachandran_plot(phis, psis, iteration, args, name=''):
    # Ramachandran plot
    phis = np.concatenate([np.array(phis), np.array([0, 0])])
    psis = np.concatenate([np.array(psis), np.array([0, 0])])
    iteration = str(iteration).zfill(8)

    title = f'Iteration={iteration}'
    if 'START_TIME' in os.environ: title = f'{title} wall={int(time.time() - float(os.environ["START_TIME"]))}sec'

    fig = ramachandran_plot(phis, psis, interactive=False, title=title)
    fig.write_image(os.path.join(os.environ['MODEL_DIR'], f'ramachandran_{iteration}{name}.png'))

    if args.wandb:
        metric_name = f'ramachandran_{name}'
        wandb.log({metric_name: wandb.Image(
            os.path.join(os.environ['MODEL_DIR'], f'ramachandran_{iteration}{name}.png'), caption=f"{metric_name}")})


def kl_divergence(samples_p, samples_q, nbins=50):  # it is assumed the support of ref is wider but we want KL(other || ref)
    """
    Computes the KL divergence between two distributions, given samples from each.
    P is usually the reference distribution, and Q is the distribution we want to compare.
    """

    reference_count, bins = np.histogram(np.array(samples_q), bins=nbins, density=True)
    other_count, _ = np.histogram(np.array(samples_p), bins=bins, density=True)
    kl = other_count * (np.log(other_count) - np.log(reference_count)) * (bins[1]-bins[0])
    return np.sum(kl[np.isfinite(kl)])


def save_histogram(logs, iteration, args, key, as_free_energy=False):
    iteration = str(iteration).zfill(8)
    plt.figure(figsize=(5, 5))

    title = f'Iteration={iteration}'
    if 'START_TIME' in os.environ: title = f'{title} wall={int(time.time() - float(os.environ["START_TIME"]))}sec'

    need_legend = type(logs) != defaultdict
    if not need_legend: logs = {'mcmc': logs}
    for log_key in logs:
        log = logs[log_key]
        count, bins = np.histogram(np.array(log[key]), bins=args.num_hist_bins, density=True)
        bins = (bins[:-1] + bins[1:]) / 2
        if as_free_energy: plt.plot(bins, -np.log(count), label=log_key)
        else: plt.bar(bins, count, width=bins[1]-bins[0])
        
    if need_legend: plt.legend()
    plt.title(title)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    plt.xlabel(key, fontsize=12)
    plt.ylabel('free energy (kT)' if as_free_energy else 'frequency', fontsize=12)
    plt.savefig(os.path.join(os.environ['MODEL_DIR'], f'{key}_hist_{iteration}.png'), dpi=150)
    plt.close()
    if args.wandb:
        metric_name = f'{key}_hist'
        wandb.log({metric_name: wandb.Image(os.path.join(os.environ['MODEL_DIR'], f'{key}_hist_{iteration}.png'),caption=f"{metric_name}")})
