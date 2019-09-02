import numpy as np
import json
#  from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import matplotlib.colors as mc
import colorsys
import re
import pandas as pd
from matplotlib.lines import Line2D


#  plt.rcParams['text.usetex'] = True

#  info_database_path = \
#      Path('home/bolensadrien/Documents/RL/results/info_database.json')

with open('info_database.json') as f:
    info_dict = json.load(f)


def plot_rewards_vs_episodes(path_to_data,
                             n_x=None,
                             scatter=True,
                             n_x_scatter=120,
                             n_individual_runs=2,
                             n_slices=0,
                             yrange=[0, 1.01],
                             ax=None):
    """
    Plot of the rewards during training as a function of the training episodes.
    For each episode, the mean over all runs is plotted. In addition, a scatter
    plot of individual runs can also be plotted. The exploration schedule and a
    horizontal line showing the maximum reward are also plotted.  (As well as
    one for the initial reward for some system, e.g. the Ising model)

    Args
    ----
    path_to_data (pathlib.Path)
    n_x (int): the number of episodes to consider for the plot of the mean
    scatter (bool): whether or not to add the scatter plot
    n_x_scatter (int): the number of episodes to consider for the scatter plot
        (a too high number can result in a heavy file)
    n_individual_runs (int)
    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)
    info = info_dict[path_to_data.name]

    params = info['parameters']
    n_episodes = params['n_episodes']
    n_runs = info['n_completed_tasks']

    if params['system_class'] == 'LongRangeIsing':
        initial_reward = info['initial_reward']

    reward_array = np.load(path_to_data / 'rewards.npy')[:, :n_episodes]
    if reward_array.shape[0] != n_runs:
        raise ValueError('`n_completed_tasks` in `info.json` does not match '
                         'the shape of `rewards.npy`.')

    max_reward = np.max(reward_array)
    reward_mean = reward_array.mean(axis=0)
    eps_max = params['epsilon_max']
    eps_min = params['epsilon_min']
    eps_decay = params['epsilon_decay']
    eps = [max(eps_min, eps_max * eps_decay**i) for i in range(n_episodes)]

    x = range(n_episodes)
    if n_x is None:
        n_skip = 1
    else:
        n_skip = max(n_episodes // n_x, 1)
    n_skip_scatter = max(n_episodes // n_x_scatter, 1)

    #  Use colors of the ggplot style, even if global style is different
    with plt.style.context(('ggplot')):
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_eps = cs[4]
    c_mean = cs[1]
    c_vline = cs[0]
    cs = [c for i, c in enumerate(cs) if i not in [0, 1, 4]]
    #  cs = sns.color_palette()[4:]

    #  slices used in the distribution plot:
    if n_slices > 0:
        slice_episodes = np.linspace(0, n_episodes-1, n_slices).astype(int)
        for ep in slice_episodes:
            ax.axvline(ep, c='gray', linestyle='--', alpha=0.7)

    #  exploration schedule
    ax.plot(eps, label=r'$\varepsilon$', c=c_eps)

    #  scatter of all rewards
    if scatter:
        for i in range(len(reward_array)):
            ax.scatter(x[::n_skip_scatter], reward_array[i, ::n_skip_scatter],
                       c='k', marker='.', s=1.5,
                       alpha=min(30/n_runs, 0.1))

    #  maximal reward
    ax.axhline(max_reward, label=rf'max $r={max_reward:.2f}$', c=c_vline)
    #  initial reward (Trotter)
    if params['system_class'] == 'LongRangeIsing':
        ax.axhline(initial_reward, label=rf'Trotter $r={initial_reward:.2f}$',
                   c=c_vline, linestyle='--')

    # individual runs
    for i in range(n_individual_runs):
        ax.plot(x[::n_skip], reward_array[i, ::n_skip], alpha=1.0,
                c=cs[i % len(cs)])

    #  mean of rewards
    ax.plot(x[::n_skip], reward_mean[::n_skip], c=c_mean,
            label=r'mean $r$')

    ax.set_ylim(yrange)
    ax.set_xlim([0, n_episodes])
    ax.set_xlabel(r'$t$ (episode)')
    ax.set_ylabel('reward')
    ax.set_title('Evolution of the episodic reward during training')
    ax.annotate(path_to_data.name, (0.0, 0.95), xycoords='axes fraction')


def plot_energy_vs_episodes(path_to_data,
                            n_x=None,
                            scatter=True,
                            n_x_scatter=120,
                            n_individual_runs=2,
                            n_slices=0,
                            yrange=None,
                            gs_energy=-7.817385707357522,
                            ax=None):
    """
    Plot of the rewards during training as a function of the training episodes.
    For each episode, the mean over all runs is plotted. In addition, a scatter
    plot of individual runs can also be plotted. The exploration schedule and a
    horizontal line showing the maximum reward are also plotted.  (As well as
    one for the initial reward for some system, e.g. the Ising model)

    Args
    ----
    path_to_data (pathlib.Path)
    n_x (int): the number of episodes to consider for the plot of the mean
    scatter (bool): whether or not to add the scatter plot
    n_x_scatter (int): the number of episodes to consider for the scatter plot
        (a too high number can result in a heavy file)
    n_individual_runs (int)
    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)
    info = info_dict[path_to_data.name]

    params = info['parameters']
    n_episodes = params['n_episodes']
    n_runs = info['n_completed_tasks']

    #  if params['system_class'] == 'LongRangeIsing':
    #      initial_reward = info['initial_reward']

    reward_array = np.load(path_to_data / 'rewards.npy')[:, :n_episodes]
    if reward_array.shape[0] != n_runs:
        raise ValueError('`n_completed_tasks` in `info.json` does not match '
                         'the shape of `rewards.npy`.')

    min_energy = -np.max(reward_array)
    xx, yy = np.argwhere(reward_array == -min_energy)[0]
    print(xx, yy)
    print(np.max(reward_array[xx]))
    print(np.max(reward_array))

    energy_mean = -reward_array.mean(axis=0)
    eps_max = params['epsilon_max']
    eps_min = params['epsilon_min']
    eps_decay = params['epsilon_decay']
    eps = [max(eps_min, eps_max * eps_decay**i) for i in range(n_episodes)]

    x = range(n_episodes)
    if n_x is None:
        n_skip = 1
    else:
        n_skip = max(n_episodes // n_x, 1)
    n_skip_scatter = max(n_episodes // n_x_scatter, 1)

    #  Use colors of the ggplot style, even if global style is different
    with plt.style.context(('ggplot')):
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_eps = cs[4]
    c_mean = cs[1]
    c_vline = cs[0]
    cs = [c for i, c in enumerate(cs) if i not in [0, 1, 4]]
    #  cs = sns.color_palette()[4:]

    #  slices used in the distribution plot:
    if n_slices > 0:
        slice_episodes = np.linspace(0, n_episodes-1, n_slices).astype(int)
        for ep in slice_episodes:
            ax.axvline(ep, c='gray', linestyle='--', alpha=0.7)

    #  exploration schedule
    ax.plot(eps, label=r'$\varepsilon$', c=c_eps)

    #  scatter of all rewards
    if scatter:
        for i in range(len(reward_array)):
            ax.scatter(x[::n_skip_scatter], -reward_array[i, ::n_skip_scatter],
                       c='k', marker='.', s=1.5,
                       alpha=min(30/n_runs, 0.1))

    #  maximal reward
    ax.axhline(min_energy, label=rf'max $r={min_energy:.2f}$', c=c_vline)
    #  initial reward (Trotter)
    ax.axhline(gs_energy, label=rf'GS energy $r={gs_energy:.2f}$',
               c=c_vline, linestyle='--')

    # individual runs
    for i in list(range(n_individual_runs)) + [xx]:
        print(i)
        ax.plot(x[::n_skip], -reward_array[i, ::n_skip], alpha=1.0,
                c=cs[i*2 % len(cs)])

    #  mean of rewards
    ax.plot(x[::n_skip], energy_mean[::n_skip], c=c_mean,
            label=r'mean $r$')

    ax.set_ylim(yrange)
    ax.set_xlim([0, n_episodes])
    ax.set_xlabel(r'$t$ (episode)')
    ax.set_ylabel('expected value of energy')
    ax.set_title('Evolution of the episodic final energy during training')
    ax.annotate(path_to_data.name, (0.0, 0.95), xycoords='axes fraction')


def plot_comparison_of_rewards(paths_to_data,
                               class_def=None,
                               param_filter=None,
                               with_legend=True,
                               reward_range=[0, 1],
                               ax=None):
    """
    Barplot of the key values for `different` datasets, in order to compare
    them. Those values are:
        1. the mean final reward
        2. the best reward
        3. the initial reward (if it exists)

    Args
    ----
    paths_to_data (list of pathlib.Path)

    class_def (list): List of keys of the `parameters` dictionary in info.json
        (or tuples of keys for nested dictionaries) . The list defines how to
        classify the datasets. Different classes are plotted with different
        colors and are grouped together.  E.g. class_def = ['n_sites',
        'n_steps', ('ham_params', 'alpha')]

    param_filter (dictionary): Only dataset with `parameters[key] =
        param_filter[key]` are considered, for all the keys in param_filter.
        (`parameters` is a dictionary found in info.json of the dataset)
        E.g. param_filter = {'subclass': 'WithReplayMemory'}

    with_legend (bool)

    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    if class_def is None:
        class_def = []
    if param_filter is None:
        param_filter = {}

    data_indices = []
    final_rewards = []
    max_rewards = []
    initial_rewards = []
    classes = []

    def get_param(param_dict, key, na_value=np.nan):
        if type(key) == tuple:
            if len(key) == 1:
                return param_dict.get(key[0], na_value)
            else:
                if key[0] in param_dict:
                    return get_param(param_dict[key[0]], key[1:])
                else:
                    return na_value
        else:
            return param_dict.get(key, na_value)

    for path in paths_to_data:
        if not path.is_dir():
            continue
        #  with open(path / 'info.json') as f:
        #      info = json.load(f)
        info = info_dict[path.name]
        params = info['parameters']
        for p, value in param_filter.items():
            if type(value) is list:
                if get_param(params, p) not in value:
                    break
            else:
                if get_param(params, p) != value:
                    break
        else:
            data_indices.append(re.search(r'\d+', path.name).group())
            n_episodes = params['n_episodes']
            reward_array = np.load(path / 'rewards.npy')[:, :n_episodes]
            final_rewards.append(reward_array[:, -1].mean())
            max_rewards.append(np.max(reward_array))
            initial_rewards.append(info['initial_reward'])
            classes.append(tuple(get_param(params, p) for p in class_def))

    df = pd.DataFrame({'index': data_indices,
                       'final_reward': final_rewards,
                       'max_reward': max_rewards,
                       'initial_reward': initial_rewards,
                       'class': classes}).sort_values(by=['class', 'index'])
    #  , how=lambda x: (x is None, x))

    classes = sorted(list(set(classes)))

    with plt.style.context(('ggplot')):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2

    print(classes)
    colors = sns.color_palette(n_colors=len(classes))

    amounts = [0.45, 1.0]
    colors_all = [[lighten_color(c, a) for c in colors] for a in amounts]
    c_dicts = [dict(zip(classes, colors)) for colors in colors_all]

    x = range(len(df))
    #  features to plot from bottom to top
    features = ['initial_reward', 'max_reward']
    tops = [df[feat].values for feat in features]
    bottoms = [np.zeros(len(df))] + tops[:-1]

    for bottom, top, c_dict in zip(bottoms, tops, c_dicts):
        ax.bar(x=x, height=top-bottom, bottom=bottom,
               color=list(map(c_dict.get, df['class'])))

    ax.bar(x=x, height=np.zeros(len(df)), bottom=df['final_reward'],
           edgecolor='k')

    ax.set_xticks(x)
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(df['index'], rotation=90)
    ax.set_ylim(reward_range)
    ax.set_title('Comparison of rewards/fidelities accross different runs',
                 y=1.08)

    #  # label above the bars:
    #  for x_value, y_value, label in zip(x, df['max_reward'], df['index']):
    #      # Number of points between bar and label
    #      space = 5
    #      # Vertical alignment (for positive values)
    #      va = 'bottom'
    #      # Create annotation
    #      ax.annotate(
    #          label,
    #          (x_value, y_value),
    #          xytext=(0, space),
    #          textcoords="offset points",
    #          ha='center',
    #          va=va)

    if not with_legend:
        return None

    custom_lines = [Line2D([0], [0], color=c_dicts[1][cl], lw=4)
                    for cl in classes]

    short_name = {'n_steps': 'steps',
                  'n_sites': 'sites',
                  'time_segment': 't',
                  'alpha': r'$\alpha$'}

    class_params = [short_name.get(p[-1], p[-1]) if type(p) == tuple
                    else short_name.get(p, p) for p in class_def]

    label_format = {'float': '{0}: {1:.2f}'}
    labels = [str(', '.join(
        [label_format.get(type(value), '{0}: {1}').format(param, value)
         for param, value in zip(class_params, cl)]
    )) for cl in classes]

    #  labels = [l.replace('_', '\_') for l in labels]
    legend_colors = ax.legend(custom_lines, labels, loc='lower left')

    labels_reward = ['max reward', 'init. reward', 'final reward (mean)']
    label_colors = reversed(['k'] +
                            [lighten_color(colors[0], a) for a in amounts])
    lws = [10, 10, 1]
    custom_lines = [Line2D([0], [0], color=c, lw=lw)
                    for c, lw in zip(label_colors, lws)]

    ax.legend(custom_lines, labels_reward, loc='lower right')
    ax.add_artist(legend_colors)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_rewards_vs_one_parameter(paths_to_data,
                                  parameter,
                                  class_def=None,
                                  param_filter=None,
                                  with_legend=True,
                                  ax=None):
    """
    Scatter plot of the best reward as a function of a single parameter for
    the datasets in `paths_to_data`.

    Args
    ----
    paths_to_data (list of pathlib.Path)

    parameter (str or list of str for nested parameters): abscissa of the plot

    class_def (list): List of keys of the `parameters` dictionary in info.json
        (or tuples of keys for nested dictionaries) . The list defines how to
        classify the datasets. Different classes are plotted with different
        colors and are grouped together.  E.g. class_def = ['n_sites',
        'n_steps', ('ham_params', 'alpha')]

    param_filter (dictionary): Only dataset with `parameters[key] =
        param_filter[key]` are considered, for all the keys in param_filter.
        (`parameters` is a dictionary found in info.json of the dataset)
        E.g. param_filter = {'subclass': 'WithReplayMemory'}

    with_legend (bool)

    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    if class_def is None:
        class_def = []
    if param_filter is None:
        param_filter = {}

    indices = []
    #  final_rewards = []
    max_rewards = []
    initial_rewards = []
    x_values = []
    classes = []

    def get_param(param_dict, key):
        if type(key) == tuple:
            if len(key) == 1:
                return param_dict.get(key[0])
            else:
                return get_param(param_dict.get(key[0]), key[1:])
        else:
            return param_dict.get(key)

    for path in paths_to_data:
        if not path.is_dir():
            continue
        #  with open(path / 'info.json') as f:
        #      info = json.load(f)
        info = info_dict[path.name]
        params = info['parameters']
        for p, value in param_filter.items():
            if get_param(params, p) != value:
                break
        else:
            p = params[parameter]
            if p is None:
                continue
            x_values.append(p)
            indices.append(re.search(r'\d+', path.name).group())
            n_episodes = params['n_episodes']
            reward_array = np.load(path / 'rewards.npy')[:, :n_episodes]
            #  final_rewards.append(reward_array[:, -1].mean())
            max_rewards.append(np.max(reward_array))
            initial_rewards.append(info['initial_reward'])
            classes.append(tuple(get_param(params, p) for p in class_def))
    classes, indices, max_rewards, initial_rewards, x_values = (
        zip(*sorted(zip(classes, indices, max_rewards,
                        initial_rewards, x_values)))
    )

    unique_classes = sorted(list(set(classes)))

    with plt.style.context(('ggplot')):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2

    colors = sns.color_palette(n_colors=len(unique_classes))
    c_dict = dict(zip(unique_classes, colors))
    cs = [c_dict[cl] for cl in classes]

    ax.scatter(x_values, max_rewards, s=10**2, c=cs, marker='o')
    ax.scatter(x_values, initial_rewards, s=10**2, c=cs, marker='s')

    ax.set_ylim([0, 1])
    ax.set_title(f'Rewards as a function of {parameter}',
                 y=1.08)

    ax.set_xlabel(parameter)
    ax.set_ylabel('reward')
    if not with_legend:
        return None

    custom_handles = [Line2D([0], [0], color=c_dict[cl], marker='o',
                      linestyle='', markersize=10) for cl in unique_classes]

    short_name = {'n_steps': 'steps',
                  'n_sites': 'sites',
                  'time_segment': 't',
                  'alpha': r'$\alpha$'}

    class_params = [short_name.get(p[-1], p[-1]) if type(p) == tuple
                    else short_name.get(p, p) for p in class_def]

    label_format = {float: '{0}: {1:.2f}'}
    labels = [str(', '.join(
        [label_format.get(type(value), '{0}: {1}').format(param, value)
         for param, value in zip(class_params, cl)]
    )) for cl in unique_classes]

    legend_colors = ax.legend(custom_handles, labels, loc='lower left')

    labels_reward = ['init. reward', 'max reward']
    handles = [
        Line2D([0], [0], color='k', marker='s', linestyle='', markersize=10),
        Line2D([0], [0], color='k', marker='o', linestyle='', markersize=10)
    ]

    ax.legend(handles, labels_reward, loc='lower right')
    ax.add_artist(legend_colors)


def plot_rewards_vs_extra_episodes(path_to_data,
                                   n_x=None,
                                   scatter=False,
                                   n_x_scatter=120,
                                   n_individual_runs=2,
                                   ax=None):

    """
    Plot of the rewards as a function of the `extra_episodes`, i.e. obtained
    after the end the `official` runs to study the individual effects of:

       1. update of Q (the neural networks) (i.e. without any exploration)
       2. exploration (with a fixed Q-function/neural network)
       3. Neither -> usually, it should be deterministic, i.e. no change in
           the rewards (could check, e.g., the action selection algorithm)

    The mean over all runs is plotted. In addition, inidividual runs can and a
    scatter plot of all individual runs can also be plotted.

    Args
    ----
    path_to_data (pathlib.Path or str)
    n_x (int): the number of episodes to consider for the plot of the mean
    scatter (bool): whether or not to add the scatter plot
    n_x_scatter (int): the number of episodes to consider for the scatter plot
        (a too high number can result in a heavy file)
    n_individual_runs (int)
    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)
    info = info_dict[path_to_data.name]

    params = info['parameters']
    n_episodes = params['n_episodes']
    n_extra_episodes = params.get('n_extra_episodes', None)
    n_runs = info['n_completed_tasks']

    reward_array = np.load(path_to_data / 'rewards.npy')[:, n_episodes:]
    if reward_array.shape[0] != n_runs:
        raise ValueError('`n_completed_tasks` in `info.json` does not match '
                         'the shape of `rewards.npy`.')
    if n_extra_episodes is None:
        n_extra_episodes = reward_array.shape[1]
    if reward_array.shape[1] != n_extra_episodes:
        raise ValueError('`n_extra_episodes` in `info.json` does not '
                         ' match the shape of `rewards.npy`.')

    reward_mean = reward_array.mean(axis=0)

    x = list(range(n_extra_episodes))
    if n_x is None:
        n_skip = 1
    else:
        n_skip = max(n_episodes // n_x, 1)
    n_skip_scatter = max(n_extra_episodes // n_x_scatter, 1)

    #  Use colors of the ggplot style, even if global style is different
    with plt.style.context(('ggplot')):
        cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_mean = cs[0]
    cs = [c for i, c in enumerate(cs) if i not in [0]]

    #  scatter of all rewards
    if scatter:
        for i in range(len(reward_array)):
            ax.scatter(x[::n_skip_scatter], reward_array[i, ::n_skip_scatter],
                       c='k', marker='.', s=1.7,
                       alpha=min(50/n_runs, 0.1))

    # individual runs
    for i in range(n_individual_runs):
        ax.plot(x[::n_skip], reward_array[i, ::n_skip], alpha=1.0,
                c=cs[i % len(cs)], label=f'run #{i}')

    #  mean of rewards
    ax.plot(x[::n_skip], reward_mean[::n_skip], c=c_mean,
            label=r'mean $r$')

    ax.set_ylim([0, 1.01])
    ax.set_xlim([0, n_extra_episodes])
    ax.set_xlabel(
        r'$t$ (extra episodes: no exploration / no update / neither)'
    )
    ax.set_ylabel('reward')
    ax.set_title('Rewards obtained by the agent after being trained')
    ax.legend(loc='lower right')
    ax.annotate(path_to_data.name, (0.0, 0.05), xycoords='axes fraction')


def plot_reward_distribution(path_to_data,
                             n_bins=50,
                             n_slices=11,
                             with_theta=True,
                             ax1=None,
                             ax2=None):
    """
    Plot of the reward distribution accross the different runs at selected
    training episodes.

    Args
    ----
    path_to_data (pathlib.Path or str)
    n_bins (int): number of bins considered for the histogram
    n_slices: number of episodes to be considered (equally distributed between
            the first and last episodes, which are both included)
    with_theta (bool): whether or not to plot an additional plot of θ
            e^(-n_sites * θ(r)) = p(r), the reward density/probability
    ax1 (matplotlib.axes.Axes)
    ax2 (matplotlib.axes.Axes): only considered when with_theta is True
    """

    if with_theta:
        if (ax1 is None) != (ax2 is None):
            raise ValueError('Only one plt.axes was specified. Either zero or'
                             ' two axes must be specified when'
                             ' `with_theta=True`.')
        if (ax1 is None and ax2 is None):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    elif ax1 is None:
        ax1 = plt.gca()

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)
    info = info_dict[path_to_data.name]

    params = info['parameters']
    n_episodes = params['n_episodes']
    n_sites = params['n_sites']

    reward_array = np.load(path_to_data / 'rewards.npy')[:, :n_episodes]

    slice_episodes = np.linspace(0, n_episodes-1, n_slices).astype(int)
    r_slices = [reward_array[:, ep] for ep in slice_episodes]
    probs, bins = zip(*[
        np.histogram(r, bins=n_bins, density=True, range=[0.0, 1.0])
        for r in r_slices
    ])
    bins = bins[0]
    if with_theta:
        thetas = np.array([-np.log(np.maximum(p, 1e-4))/n_sites
                           for p in probs])
        thetas[thetas > 0.0] = 0.0
    x = 0.5*(bins[1:] + bins[:-1])
    #  colors = sns.color_palette("hls", n_slices)
    colors = sns.color_palette("coolwarm", n_slices)
    for i in range(n_slices):
        ax1.plot(x, probs[i], color=colors[i],
                 label=r'$t = {}$'.format(slice_episodes[i]))
        if with_theta:
            ax2.plot(x, thetas[i], color=colors[i])
        #  ax.axvline(slice_episodes[i], c='gray', linestyle='--')
    if with_theta:
        ax2.set_ylabel(r'$\theta(t, r)$')
        ax2.set_xlabel('reward')
    else:
        ax1.set_xlabel('reward')
    ax1.set_ylabel(r'$P(t, r)$')
    ax1.set_title('Distribution of the rewards at given episodes during '
                  'training')
    ax1.legend()
    ax1.annotate(path_to_data.name, (0.0, 0.95), xycoords='axes fraction')


def plot_network_metrics(path_to_data,
                         scatter=True,
                         n_x_scatter=50,
                         ax=None):
    """
    Plot of the history of the metrics of the neural network during the
    training of the agent. Note that the target of the neural network is
    constantly evolving (following the q-learning algorithm), so that the
    neural network error should converge on small time scales (a single fit),
    but can increase again at the start of the following fit, until the
    q-learning itself converges.

    Args
    ----
    path_to_data (pathlib.Path or str)
    scatter (bool): whether or not to add the scatter plot
    n_x_scatter (int): the number of episodes to consider for the scatter plot
    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)
    info = info_dict[path_to_data.name]

    params = info['parameters']
    n_episodes = params['n_episodes']
    n_runs = info['n_completed_tasks']

    #  hist_shape is (n_metrics, n_arrays, n_episodes (cutting out extras))
    history_array = (
        np.load(path_to_data / 'NN_histories.npy')[:, :, :n_episodes]
    )
    default_metrics = ['loss function (logcosh)',
                       'mean squared error',
                       'mean asbolute error']
    metrics = info.get('NN_metrics', default_metrics)

    hist_shape = history_array.shape

    #  we want to average over n_arrays
    hist_average = history_array.mean(axis=1)

    if scatter:
        n_skip_scatter = max(n_episodes // n_x_scatter, 1)
        history_array_reduced = history_array[:, :, ::n_skip_scatter]
        x_scatter = range(n_episodes)[::n_skip_scatter]

    colors = sns.color_palette('Set1', len(metrics))
    if scatter:
        for i in range(hist_shape[1]):
            for c, hist in zip(colors, history_array_reduced[:, i]):
                ax.scatter(x_scatter, hist, c=c, marker='.', s=2,
                           alpha=min(50/n_runs, 0.1))

    for hist, metric, c in zip(hist_average, metrics, colors):
        ax.plot(hist, label=metric, c=c)
    ax.set_xlabel(r'$t$ (episode)')
    ax.set_title('History of the metrics of the neural network')
    with plt.rc_context({'text.usetex': False}):
        ax.legend()
    ax.annotate(path_to_data.name, (0.0, 0.95), xycoords='axes fraction')


def plot_post_episode_rewards(path_to_data, ax=None):
    """
    Plot of the rewards obtained after the episode for the best run, as a
    function of the number of extra episodes. The evolution after the end of
    the episode is simply obtained by reapplying the sequence of gates of the
    best run.

    The rewards are the fidelities with repesct to the state obtained for the
    exact dynamic starting from:
        1. The initial state at time 0 ("absolute reward")
        2. The state at the start of the extra episode ("relative reward")

    For some systems, the rewards using the initial (e.g. Trotter) sequence
    of gates is also shown.

    Args
    ----
    path_to_data (pathlib.Path or str)
    ax (matplotlib.axes.Axes)
    """

    if ax is None:
        ax = plt.gca()

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)
    info = info_dict[path_to_data.name]

    params = info['parameters']

    r1_best, r2_best = np.load(path_to_data / 'post_episode_rewards__best.npy')
    if params['system_class'] == 'LongRangeIsing':
        r1_trotter, r2_trotter = (np.load(path_to_data /
                                          'post_episode_rewards__trotter.npy'))

    c = sns.color_palette("Set1", 8)
    ax.plot(r1_best, label='best run (absolute)', c=c[0])
    ax.plot(r2_best, label='best run (relative)', c=c[0], linestyle='--')
    if params['system_class'] == 'LongRangeIsing':
        ax.plot(r1_trotter, label='Trotter (absolute)', c=c[1])
        ax.plot(r2_trotter, label='Trotter (relative)', c=c[1], linestyle='--')
    ax.legend(loc='lower right')
    ax.set_xlabel(r'$n$ (applying the same sequence of gates $n$ times)')
    ax.set_ylabel('Fidelity')
    ax.set_ylim([0, 1.01])
    ax.set_title('Fidelity obtained by repeating the best sequence found '
                 'several times')
    ax.annotate(path_to_data.name, (0.0, 0.95), xycoords='axes fraction')


def plot_q_arrays(path_to_data, n=100, ax=None):
    """
    Plot of the q-values chosen (by the neural network) when calculating
    the best action (i.e. a = argmax_a Q).
    This plot only works for a q-learning run in which q-values where
    systematically evaluated on a grid of action-values. The q-values are then
    compared with the min and max q-values in the grid.

    personal note: I only used this plot once to check if the argmax worked
    properly (it did). It should be modified for new data.

    Args
    ----
    path_to_data (pathlib.Path or str)
    n (int): number of data point to consider. The data comes from all the
        argmax calculations in given episodes, for regularly spaced episodes,
        for all the runs (just appended one after the other)
        The length of episodic segments is irregular because of exploration:
        the argmax calculation isn't done when exploring
    ax (matplotlib.axes.Axes)
    """
    if ax is None:
        ax = plt.gca()

    q_chosen, q_max, q_min = np.load(path_to_data / 'q_arrays_comparison.npy')
    ax.plot(q_max[:n], 'g.', label='discrete max')
    ax.fill_between(range(n), q_min[:n], q_max[:n], alpha=0.4,
                    facecolor='green', label='discrete min to max')
    ax.plot(q_chosen[:n], '.-', label='chosen')
    ax.legend()


def plot_info(path_to_data, categories=None, ax=None, print_instead=False):
    """
    Simply write the info about the dataset on a white background.

    Args
    ----
    path_to_data (pathlib.Path)
    categories (list of str): list of categories to show.
        E.g. categories = ['main', 'neural_network']
    ax (matplotlib.axes.Axes)
    """

    if ax is None and not print_instead:
        ax = plt.gca()

    if categories is None:
        categories = ['main', 'q_learning', 'neural_network']
    elif categories == 'all':
        categories = ['main', 'ham_params', 'q_learning', 'neural_network',
                      'max_q_optimizer', 'other']

    #  with open(path_to_data / 'info.json') as f:
    #      info = json.load(f)

    dir_name = path_to_data.name
    info = info_dict[dir_name]

    n_arrays = info['n_completed_tasks']
    params = info['parameters']

    param_categories = {'main': ['n_sites', 'n_steps', 'env_type',
                                 'time_segment', 'system_class'],
                        'ham_params': ['ham_params'],
                        'q_learning': ['n_episodes', 'n_replays',
                                       'replay_spacing', 'exploration'],
                        'neural_network': ['network_type', 'subclass',
                                           'architecture',
                                           'model_update_spacing',
                                           'capacity', 'sampling_size'],
                        'max_q_optimizer': ['max_q_optimizer']}

    all_keys = params.keys()
    keys_in_categories = [key for cat_params in param_categories.values()
                          for key in cat_params]

    param_categories['other'] = [key for key in all_keys
                                 if key not in keys_in_categories]

    n_submitted_tasks = info.get('n_submitted_tasks', info.get('n_arrays'))
    text_list = [f"{dir_name}, n_tasks={n_arrays}/{n_submitted_tasks}"]
    for cat in categories:
        param_dict = {key: params[key] for key in param_categories[cat]
                      if key in params}
        text_list += [cat + ': \n' + pprint.pformat(param_dict)]

    textstr = '\n\n'.join(text_list)

    if not print_instead:
        with plt.rc_context({'text.usetex': False}):
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='left')
        ax.axis('off')
    else:
        print(textstr)
