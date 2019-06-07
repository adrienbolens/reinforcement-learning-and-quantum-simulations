## -- ##
import json
import numpy as np
import matplotlib as m
m.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True

parent_folder = Path('/data3/bolensadrien/output')
result_dir = '114_deep_q_learning'
result_path = parent_folder / result_dir

with open(result_path / 'info.json') as f:
    info_dic = json.load(f)

array_dirs = [
    d.name for d in result_path.iterdir() if
    d.is_dir() and
    d.name[:6] == 'array-' and
    d.name[6:].isdigit()
]

len_list = []
for a_dir in array_dirs:
    a_path = result_path / a_dir
    len_list.append(np.load(a_path / 'list_q_max_chosen.npy').shape[0])

len_cum = np.cumsum(len_list)
len_cum = np.insert(len_cum, 0, 0)
intervals = list(zip(len_cum[:-1], len_cum[1:]))

n_arrays = info_dic['n_completed_tasks']
n_steps = info_dic['parameters']['n_steps']

## -- ##

q_chosen, q_max, q_min = np.load(result_path / 'q_arrays_comparison.npy')

for (i, j) in intervals[0:30]:
    #  for k, qs in enumerate(np.reshape(q_max[i:j], (-1, n_steps))):
    #      xs = range(k*n_steps, (k+1)*n_steps)
    #      plt.plot(xs, qs, label='discrete max')
    fig = plt.figure(figsize=(15, 10))
    plt.plot(q_max[i:j], 'g.', label='discrete max')
    #  plt.plot(q_min[i:j], '.', label='discrete min')
    plt.fill_between(range(j-i), q_min[i:j], q_max[i:j], alpha=0.4,
                     facecolor='green', label='discrete min to max')
    plt.plot(q_chosen[i:j], '.-', label='chosen')
    plt.legend()
    plt.show()
    plt.close(fig)

## -- ##
