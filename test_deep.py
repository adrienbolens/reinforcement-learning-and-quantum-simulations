from importlib import reload
import numpy as np
import deep_q_learning as deep
from pathlib import Path
import json
## -- ##

deep = reload(deep)
output = Path('/data3/bolensadrien/output')
result_dir = '113_deep_q_learning'
result_path = output / result_dir


with open(result_path / 'info.json') as f:
    info_dic = json.load(f)

parameters = info_dic['parameters']
#  parameters['n_steps'] = 1
q_learning = deep.DeepQLearning(seed=1, **parameters)

weights = np.load(result_path / 'final_weights.npy')
q_learning.model.set_weights(weights)
q_learning.run_episode(verbose=True, mode='greedy', update=False)

print(q_learning.list_q_chosen_actions)
print(q_learning.list_q_discretized_actions)
## -- ##
