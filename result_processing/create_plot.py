import numpy as np
import json
#  import matplotlib as m
#  m.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
#  import pandas as pd
#  from pathlib import Path
#  sem: standard error of the mean  = std/sqrt(n) where std is the SAMPLE std
#  (std^2 = sum(x_i-mean)^2/(n-1))
#  from scipy.stats import sem


def create_plot(parent_folder, dir_name, plot_name='plot'):
    #  input_dir = Path(__file__).parent / dir_name
    input_dir = parent_folder / dir_name

    with open(input_dir / 'info.json') as f:
        info = json.load(f)

    n_arrays = info['n_completed_tasks']
    print(f"{dir_name} contains {n_arrays} completed tasks.")

    params = info['parameters']
    q_learning_subclass = params.get('subclass', None)
    if params['system_class'] == 'LongRangeIsing':
        initial_reward = info['initial_reward']
    n_episodes = params['n_episodes']
    reward_array = np.empty((n_arrays, n_episodes), dtype=np.float32)
    reward_array = np.load(input_dir / 'rewards.npy')

    #  max_final_reward = np.max(reward_array[:, -1])
    max_reward = np.max(reward_array)

    n_sites = params['n_sites']
    if q_learning_subclass != 'WithReplayMemory':
        n_bins = 50
        n_slices = 11
        slice_episodes = np.linspace(0, n_episodes-1, n_slices).astype(int)
        r_slices = [reward_array[:, ep] for ep in slice_episodes]
        probs, bins = zip(*[
            np.histogram(r, bins=n_bins, density=True, range=[0.0, 1.0]) for r
            in r_slices
        ])
        bins = bins[0]
        thetas = np.array([-np.log(p)/n_sites for p in probs])
        thetas[thetas > 0.0] = 0.0

    n_skip = max(n_episodes // 500, 1)
    reward_array = reward_array[:, ::n_skip]

    x = range(n_episodes)[::n_skip]
    #  df = pd.DataFrame(reward_array, columns=x).melt(
    #      var_name='episode', value_name='reward')
    reward_mean = reward_array.mean(axis=0)
    #  reward_std = reward_array.std(axis=0)
    #  reward_sem = sem(reward_array, axis=0)

    #  y1 = reward_mean + 1 * reward_std
    #  y2 = reward_mean - 1 * reward_std

    eps_max = params['epsilon_max']
    eps_min = params['epsilon_min']
    eps_decay = params['epsilon_decay']
    eps = [max(eps_min, eps_max * eps_decay**i) for i in range(n_episodes)]

#  sns.lineplot(x='episode', y='reward', data=df, ci=None,
#               err_kws={'y1': y1, 'y2': y2})

    #  with sns.axes_style("darkgrid"):
    sns.set_style('darkgrid')
    #  fig, (ax, axtext) = plt.subplots(1, 2, figsize=(8*3/2, 7),
    #                                   gridspec_kw={'width_ratios': [2, 1]})
    if q_learning_subclass == 'WithReplayMemory':
        plt.figure(figsize=(8.5*7/2, 9))
        ax = plt.subplot2grid((2, 5), (0, 0), colspan=2, rowspan=2)
        ax_hist = plt.subplot2grid((2, 5), (0, 2), colspan=2, rowspan=2)
        #  ax0 = plt.subplot2grid((2, 7), (0, 4), colspan=2, rowspan=1)
        #  ax1 = plt.subplot2grid((2, 7), (1, 4), colspan=2, rowspan=1)
        axtext = plt.subplot2grid((2, 5), (0, 4), colspan=1, rowspan=2)
    else:
        plt.figure(figsize=(8.5*7/2, 9))
        ax = plt.subplot2grid((2, 7), (0, 0), colspan=2, rowspan=2)
        axpost = plt.subplot2grid((2, 7), (0, 2), colspan=2, rowspan=2)
        ax0 = plt.subplot2grid((2, 7), (0, 4), colspan=2, rowspan=1)
        ax1 = plt.subplot2grid((2, 7), (1, 4), colspan=2, rowspan=1)
        #  ax2 = plt.subplot2grid((2, 5), (1, 2), colspan=1, rowspan=1)
        #  ax3 = plt.subplot2grid((2, 5), (1, 3), colspan=1, rowspan=1)
        axtext = plt.subplot2grid((2, 7), (0, 6), colspan=1, rowspan=2)
    for i in range(len(reward_array)):
        ax.scatter(x[::4], reward_array[i, ::4], c='k',
                   alpha=0.1, marker='.', s=3)
    ax.plot(eps, label='epsilon', c='orange')
    ax.plot(x, reward_mean)
    #  for i in range(0, len(reward_array), len(reward_array)//3):
    #      ax.plot(x, reward_array[i])
    #  ax.fill_between(x, y1, y2, alpha=0.3,
    #                  label=r'$\pm$ sample $\sigma$'
    #                  + f' (#samples = {n_arrays})')
    ax.axhline(max_reward,
               label=f'maximal reward = {max_reward:.2f}', c='r')
    if params['system_class'] == 'LongRangeIsing':
        ax.axhline(initial_reward,
                   label='initial reward (Trotter)',
                   c='r', linestyle='--')
    ax.set_ylim([0, 1.01])
    ax.set_xlim([0, n_episodes])
    ax.set_xlabel(r'$t$ (episode)')
    ax.set_ylabel('reward')
    ax.legend()

    n_submitted_tasks = info.get('n_submitted_tasks', info.get('n_arrays'))
    textstr = '\n'.join((f"{dir_name}, "
                         f"n_tasks={n_arrays}/{n_submitted_tasks}",
                        pprint.pformat(params)))
    axtext.text(0.05, 0.95, textstr, transform=axtext.transAxes,
                fontsize=11,
                verticalalignment='top', horizontalalignment='left')
    axtext.axis('off')
    #  ax.set_title(
    #      f"{dir_name} \n"
    #      f"n_sites={params['n_sites']}, n_steps={params['n_steps']},\n"
    #      f"n_dir={params['n_directions']}, "
    #      f"n_1={params['n_oqbgate_parameters']}, "
    #      f"n_all={params['n_allqubit_actions']} \n"
    #      f"{params['system_class']}"
    #  )

    if q_learning_subclass == 'WithReplayMemory':
        history_array = np.load(input_dir / 'NN_histories.npy')
        hist_shape = history_array.shape
        n_hist = hist_shape[2]
        # shape is (3, n_arrays, len)
        # we want to average over n_arrays
        hist_average = history_array.mean(axis=1)

        n_skip_hist = max(n_hist // 100, 1)
        history_array_reduced = history_array[:, :, ::n_skip_hist]
        x_reduced = range(n_hist)[::n_skip_hist]

        colors = sns.color_palette()
        for i in range(hist_shape[1]):
            ax_hist.scatter(x_reduced, history_array_reduced[0, i],
                            c=colors[0], alpha=0.1, marker='.', s=3)
            ax_hist.scatter(x_reduced, history_array_reduced[1, i],
                            c=colors[1], alpha=0.1, marker='.', s=3)
            ax_hist.scatter(x_reduced, history_array_reduced[2, i],
                            c=colors[2], alpha=0.1, marker='.', s=3)
        ax_hist.plot(hist_average[0], label='loss function (logcosh)',
                     c=colors[0])
        ax_hist.plot(hist_average[1], label='mean squared error',
                     c=colors[1])
        ax_hist.plot(hist_average[2], label='mean asbolute error',
                     c=colors[2])
        ax_hist.legend()

    else:
        xs = 0.5*(bins[1:] + bins[:-1])
        #  colors = sns.color_palette("hls", n_slices)
        colors = sns.color_palette("coolwarm", n_slices)
        for i in range(n_slices):
            ax0.plot(xs, probs[i], color=colors[i],
                     label=r'$t = {}$'.format(slice_episodes[i]))
            ax1.plot(xs, thetas[i], color=colors[i])
            #  ax.axvline(slice_episodes[i], c='gray', linestyle='--')
        ax1.set_xlabel(r'$r$ (reward)')
        ax1.set_ylabel(r'$\theta(t, r)$')
        ax0.set_ylabel(r'$P(t, r)$')
        ax0.legend(fontsize=7)

        r1_best, r2_best = np.load(input_dir /
                                   'post_episode_rewards__best.npy')
        r1_final, r2_final = np.load(input_dir /
                                     'post_episode_rewards__final.npy')

        params = info['parameters']
        if params['system_class'] == 'LongRangeIsing':
            #  initial_reward = info['initial_reward']
            r1_trotter, r2_trotter = \
                np.load(input_dir / 'post_episode_rewards__trotter.npy')

        #  n_episodes = params['n_episodes']

        sns.set_style('darkgrid')
        c = sns.color_palette("Set1", 8)

        axpost.plot(r1_best, label='best protocol (absolute)', c=c[0])
        axpost.plot(r2_best, label='best protocol (relative)', c=c[0],
                    linestyle='--')
        #  plt.plot(r1_final, label='absolute fidelity (best final protocol))')
        #  plt.plot(r2_final, label='relative fidelity (best final protocol))')
        if params['system_class'] == 'LongRangeIsing':
            axpost.plot(r1_trotter, label='Trotter (absolute)', c=c[1])
            axpost.plot(r2_trotter, label='Trotter (relative)',
                        c=c[1], linestyle='--')

        axpost.legend(fontsize=7)
        axpost.set_xlabel(r'$n$ (applying the same protocol $n$ times)')
        axpost.set_ylabel('Fidelity')
        axpost.set_ylim([0, 1.01])

    plt.savefig(input_dir / f'{plot_name}.pdf', format='pdf')
    plt.close()
    return True
