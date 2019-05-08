import matplotlib as m
m.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import json
from math import sqrt
from main import q_learning_parallel

with open('./.parallel_average/database.json') as f:
    results = json.load(f)


kwords = ['n_episodes', 'n_directions', 'n_onequbit_actions',
          'n_allqubit_actions']

lists = {}
for kw in kwords:
    lists[kw] = [results[i]['kwargs'][kw] for i in range(len(results))]

lists['N_runs'] = [results[i]['N_runs'] for i in range(len(results))]
lists['output'] = [results[i]['output'] for i in range(len(results))]
lists['job_name'] = [results[i]['job_name'] for i in range(len(results))]

for i in range(len(results)):
    q_learning_parallel(**results[i]['kwargs'])

means = []
variances = []
for output in lists['output']:
    with open(output) as f:
        data = json.load(f)
    means.append(data['result']['data'])
    variances.append(data['estimated_variance']['data'])

stds = [[sqrt(abs(v)) for v in variances[i]] for i in range(len(variances))]

#  x = [1, 2, 3, 4, 5]
n_istates = 5
x = np.arange(1, n_istates+1)

id_to_plot = [12, 13, 14, 15]
index_to_plot = [lists['job_name'].index(f'{id}_q_learning_parallel') for
                 id in id_to_plot]
for i in index_to_plot:
    plt.figure(i)
    plt.errorbar(x=x, y=means[i], yerr=stds[i])
    plt.ylim([-0.05, 1.05])
    plt.title(f"{lists['job_name'][i]} \n"
              f"n_episodes: {lists['n_episodes'][i]}, n_1: "
              f"{lists['n_onequbit_actions'][i]}, n_all: "
              f"{lists['n_allqubit_actions'][i]}, n_dir: "
              f"{lists['n_directions'][i]}")

plt.show()
