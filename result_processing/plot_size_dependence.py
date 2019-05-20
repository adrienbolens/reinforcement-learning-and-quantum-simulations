import numpy as np
import matplotlib as m
m.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True

parent_folder = Path('.')
parent_folder = Path('/data3/bolensadrien/output')
group_1 = list(range(17, 24+1)) + [35]
group_100 = list(range(26, 33+1)) + [36]
dir_paths_1 = [parent_folder / f'{i}_q_learning' for i in group_1]
dir_paths_100 = [parent_folder / f'{i}_q_learning' for i in group_100]
x_1 = list(range(3, 12))
x_100 = list(range(3, 12))

rewards_best_1 = []
rewards_trotter_1 = []
for i in range(len(dir_paths_1)):
    rewards = np.load(dir_paths_1[i] / 'post_episode_rewards__best.npy')
    rewards_best_1.append(rewards[0][0])
    rewards = np.load(dir_paths_1[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_1.append(rewards[0][0])

rewards_best_100 = []
rewards_trotter_100 = []
for i in range(len(dir_paths_100)):
    rewards = np.load(dir_paths_100[i] / 'post_episode_rewards__best.npy')
    rewards_best_100.append(rewards[0][0])
    rewards = np.load(dir_paths_100[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_100.append(rewards[0][0])

sns.set_style('darkgrid')
#  f, ax = plt.subplots(2, figsize=(8, 9))

#  ax[0].plot(x_1, rewards_best_1, label='best protocol (t=1)')
#  ax[0].plot(x_1, rewards_trotter_1, label='Trotter protocol (t=1)')
#  ax[0].legend()
#  ax[1].plot(x_100, rewards_best_100, label='best protocol (t=100)')
#  ax[1].plot(x_100, rewards_trotter_100, label='Trotter protocol (t=100)')
#  ax[1].legend()

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 18})
f, ax = plt.subplots(1, figsize=(5, 4.5))

ax.plot(x_100, rewards_best_100, label='best protocol', marker='x')
ax.plot(x_100, rewards_trotter_100, label='Trotter',
        marker='x', linestyle='--')
ax.set_xlabel('number of sites')
ax.set_ylabel('fidelity')
ax.set_ylim([0, 1.0])
ax.set_xlim([3, 11])
ticks = list(range(3, 12))
ax.set_xticks(ticks)
#  plt.box(on=None)
ax.legend(loc=1, bbox_to_anchor=(1.03, 1.03))
plt.tight_layout()
plt.show()
