import numpy as np
import matplotlib as m
m.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True

parent_folder = Path('.')
parent_folder = Path('/data3/bolensadrien/output')
#  group_1 = list(range(17, 24+1)) + [35]
#  group_100 = list(range(26, 33+1)) + [36]
group_1 = list(range(61, 70))
group_3 = list(range(89, 100))
#with 10 gates:
group_3b = list(range(100, 111))
group_100 = list(range(70, 79))
group_10 = list(range(79, 89))

dir_paths_1 = [parent_folder / f'{i}_q_learning' for i in group_1]
dir_paths_3 = [parent_folder / f'{i}_q_learning' for i in group_3]
dir_paths_3b = [parent_folder / f'{i}_q_learning' for i in group_3b]
dir_paths_10 = [parent_folder / f'{i}_q_learning' for i in group_10]
dir_paths_100 = [parent_folder / f'{i}_q_learning' for i in group_100]
x_1 = list(range(3, 12))
x_3 = list(range(3, 14))
x_3b = list(range(3, 14))
x_10 = list(range(3, 13))
x_100 = list(range(3, 12))

rewards_best_1 = []
rewards_trotter_1 = []
for i in range(len(dir_paths_1)):
    rewards = np.load(dir_paths_1[i] / 'post_episode_rewards__best.npy')
    rewards_best_1.append(rewards[0][0])
    rewards = np.load(dir_paths_1[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_1.append(rewards[0][0])

rewards_best_3 = []
rewards_trotter_3 = []
for i in range(len(dir_paths_3)):
    rewards = np.load(dir_paths_3[i] / 'post_episode_rewards__best.npy')
    rewards_best_3.append(rewards[0][0])
    rewards = np.load(dir_paths_3[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_3.append(rewards[0][0])

rewards_best_3b = []
rewards_trotter_3b = []
for i in range(len(dir_paths_3b)):
    rewards = np.load(dir_paths_3b[i] / 'post_episode_rewards__best.npy')
    rewards_best_3b.append(rewards[0][0])
    rewards = np.load(dir_paths_3b[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_3b.append(rewards[0][0])

rewards_best_10 = []
rewards_trotter_10 = []
for i in range(len(dir_paths_10)):
    rewards = np.load(dir_paths_10[i] / 'post_episode_rewards__best.npy')
    rewards_best_10.append(rewards[0][0])
    rewards = np.load(dir_paths_10[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_10.append(rewards[0][0])

rewards_best_100 = []
rewards_trotter_100 = []
for i in range(len(dir_paths_100)):
    rewards = np.load(dir_paths_100[i] / 'post_episode_rewards__best.npy')
    rewards_best_100.append(rewards[0][0])
    rewards = np.load(dir_paths_100[i] / 'post_episode_rewards__trotter.npy')
    rewards_trotter_100.append(rewards[0][0])

sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 18})
f, ax = plt.subplots(3, 2, figsize=(14, 9))
ax = ax.flat

ax[0].plot(x_1, rewards_best_1, label='best protocol (t=1)', marker='x')
ax[0].plot(x_1, rewards_trotter_1, label='Trotter protocol (t=1)', marker='x')
ax[0].legend()
ax[0].set_xlabel('number of sites')
ax[0].set_ylabel('fidelity')
ax[0].set_ylim([0, 1.0])
ax[0].set_xlim([x_1[0], x_1[-1]])
ticks = list(x_1)
ax[0].set_xticks(ticks)

ax[1].plot(x_3, rewards_best_3, label='best protocol (t=3)', marker='x')
ax[1].plot(x_3, rewards_trotter_3, label='Trotter protocol (t=3)', marker='x')
ax[1].legend()
ax[1].set_xlabel('number of sites')
ax[1].set_ylabel('fidelity')
ax[1].set_ylim([0, 1.0])
ax[1].set_xlim([x_3[0], x_3[-1]])
ticks = list(x_3)
ax[1].set_xticks(ticks)

#  ax[2].plot(x_3b, rewards_best_3b, label='best protocol (t=3, 10 gates)',
#             marker='x')
#  ax[2].plot(x_3b, rewards_trotter_3b, label='Trotter protocol (t=3b)', marker='x')
#  ax[2].legend()
#  ax[2].set_xlabel('number of sites')
#  ax[2].set_ylabel('fidelity')
#  ax[2].set_ylim([0, 1.0])
#  ax[2].set_xlim([x_3b[0], x_3b[-1]])
#  ticks = list(x_3b)
#  ax[2].set_xticks(ticks)

ax[3].plot(x_10, rewards_best_10, label='best protocol (t=10)', marker='x')
ax[3].plot(x_10, rewards_trotter_10, label='Trotter protocol (t=10)',
           marker='x')
ax[3].legend()
ax[3].set_xlabel('number of sites')
ax[3].set_ylabel('fidelity')
ax[3].set_ylim([0, 1.0])
ax[3].set_xlim([x_10[0], x_10[-1]])
ticks = list(x_10)
ax[3].set_xticks(ticks)

ax[4].plot(x_100, rewards_best_100, label='best protocol (t=100)', marker='x')
ax[4].plot(x_100, rewards_trotter_100, label='Trotter protocol (t=100)',
           marker='x')
ax[4].legend()
ax[4].set_xlabel('number of sites')
ax[4].set_ylabel('fidelity')
ax[4].set_ylim([0, 1.0])
ax[4].set_xlim([x_100[0], x_100[-1]])
ticks = list(x_100)
ax[4].set_xticks(ticks)

#  sns.set_style('whitegrid')
#  plt.rcParams.update({'font.size': 18})
#  f, ax = plt.subplots(1, figsize=(5, 4.5))

#  ax.plot(x_100, rewards_best_100, label='best protocol', marker='x')
#  ax.plot(x_100, rewards_trotter_100, label='Trotter',
#          marker='x', linestyle='--')
#  ax.set_xlabel('number of sites')
#  ax.set_ylabel('fidelity')
#  ax.set_ylim([0, 1.0])
#  ax.set_xlim([3, 11])
#  ticks = list(range(3, 12))
#  ax.set_xticks(ticks)
#  #  plt.box(on=None)
#  ax.legend(loc=1, bbox_to_anchor=(1.03, 1.03))
plt.tight_layout()
plt.show()
