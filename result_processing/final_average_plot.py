import numpy as np
import sys
import matplotlib as m
m.use('TkAgg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import re

database_path = '/home/bolensadrien/Documents/RL'
sys.path.insert(0, database_path)
from database import read_database

plt.rcParams['text.usetex'] = True

output = Path('/data3/bolensadrien/output')


database = read_database()
dir_names = [ent['name'] for ent in database if
             ent['status'] == 'processed' and
             ent['algorithm'] == 'DQN_ReplayMemory']

dir_paths = [output / d for d in dir_names]
dir_index = [int(re.search(r'\d+', d).group()) for d in dir_names]

final_reward = np.zeros(len(dir_paths))
n_sites_steps = []
for i, path in enumerate(dir_paths):
    final_reward[i] = np.load(path / 'rewards.npy')[:, -1].mean()
    with open(path / 'info.json') as f:
        info = json.load(f)
    params = info['parameters']
    n_sites_steps.append((params['n_sites'], params['n_steps']))

classes = list(set(n_sites_steps))

df = pd.DataFrame({'index': dir_index,
                   'final_reward': final_reward,
                   'class': n_sites_steps})

## -- ##
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c_dict = dict(zip(classes, colors))
xs = range(len(df))

fig, ax = plt.subplots(figsize=(15, 7))
ax.bar(x=xs, height=df['final_reward'],
       color=list(map(c_dict.get, df['class'])))

custom_lines = [Line2D([0], [0], color=c_dict[cl], lw=4) for cl in classes]
labels = [f'{n_sites} sites, {n_steps} steps' for n_sites, n_steps in classes]
ax.legend(custom_lines, labels)

ax.set_xticks(xs)
ax.set_xticklabels(df['index'], rotation=90)

for i, rect in enumerate(ax.patches):
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    va = 'bottom'

    # Use Y value as label and format number with one decimal place
    label = df['index'][i]

    # Create annotation
    ax.annotate(
        label,                       # Use `label` as label
        (x_value, y_value),          # Place label at end of the bar
        xytext=(0, space),           # Vertically shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        ha='center',                 # Horizontally center label
        va=va)
    # Vertically align label differently for # positive and negative values.

plt.savefig('final_average.pdf')
plt.show()

## -- ##
