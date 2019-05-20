import matplotlib as m
m.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

from pathlib import Path

plt.rcParams['text.usetex'] = True

#  parent_folder = Path('.')
parent_folder = Path('/data3/bolensadrien/output')
dir_name = '44_q_learning'

input_dir = parent_folder / dir_name

with open(input_dir / 'info.json') as f:
    info = json.load(f)

n_arrays = info['n_completed_tasks']
print(f"{dir_name} contains {n_arrays} completed tasks.")

params = info['parameters']
if params['system_class'] == 'LongRangeIsing':
    initial_reward = info['initial_reward']
n_episodes = params['n_episodes']

reward_array = np.empty((n_arrays, n_episodes), dtype=np.float32)
reward_array = np.load(input_dir / 'rewards.npy')[:]
max_reward = np.max(reward_array)

n_skip_scatter = n_episodes // 5000
reward_scatter = reward_array[:, ::n_skip_scatter]
x_scatter = range(n_episodes)[::n_skip_scatter]

n_skip = 1
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

sns.set_style('whitegrid')
#  c = sns.color_palette("Set1", 8)

plt.rcParams.update({'font.size': 23})
f, ax = plt.subplots(figsize=(10, 8))

ax.plot(eps, c='green', alpha=0.8, linewidth=1.5,
        label=r'$\epsilon$ (exploration schedule)')
for i in range(len(reward_scatter)):
    ax.scatter(x_scatter, reward_scatter[i],
               c='k', alpha=0.009, marker='.', s=0.5)
ax.plot(x, reward_mean, label='mean fidelity')
#  ax.fill_between(x, y1, y2, alpha=0.3,
#                  label=r'$\pm$ sample $\sigma$'
#                  + f' (#samples = {n_arrays})')
ax.axhline(max_reward, label=fr'max fidelity = ${max_reward:.2f}$', c='r')
if params['system_class'] == 'LongRangeIsing':
    ax.axhline(initial_reward,
               label=rf'Trotter fidelity = ${initial_reward:.2f}$',
               c='r', linestyle='--')
#  ax.set_ylim([0, 1.01])
ax.set_ylim([0, 1.0])
ax.set_xlim([0, 1e5])
ax.set_xlabel('training time (episode)')
ax.set_ylabel('fidelity')
handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 3, 0]
handles, labels = zip(*[(handles[i], labels[i]) for i in order])
#  ax.legend(handles, labels, loc=4, bbox_to_anchor=(1.0, 0.25))
ax.legend(handles, labels, loc=1, bbox_to_anchor=(1.01, 1.01))
#  , fontsize=20)
#  ax.legend(fontsize=16, loc=4)

ticks = [r'$0$'] + [fr'${i} \cdot 10^4$' for i in [2, 4, 6, 8]] + [r'$10^5$']
ax.set_xticklabels(ticks)
#  [0, 2e4, 4e4, 6e4, 8e4, 1e5], ticks)


#  ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0),
#  useMathText=True)
plt.tight_layout()
plt.show()
