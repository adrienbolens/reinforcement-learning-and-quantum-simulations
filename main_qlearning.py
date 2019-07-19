import numpy as np
import q_learning as ql
import sys
import json
from pathlib import Path

import time
start_time = time.time()

if Path('info.json').is_file():
    with open('info.json') as f:
        info = json.load(f)
    parameters = info['parameters']
else:
    from parameters import parameters, parameters_vanilla
    parameters.update(parameters_vanilla)

is_rerun = parameters['is_rerun']
if is_rerun:
    print("This is a rerun using the q_matrix in:")
    rerun_path = Path(parameters['rerun_path'])
    print(rerun_path)
    q_matrix_file = rerun_path / 'q_matrix.npy'
    q_matrix = np.load(q_matrix_file)
    with open(rerun_path / 'info.json') as f:
        info = json.load(f)
    parameters = info['parameters']

print(f"env.n_steps = {parameters['n_steps']}.")

if len(sys.argv) < 2:
    array_index = 1
    create_output_files = False
else:
    array_index = int(sys.argv[1])
    create_output_files = True

# The seed here is for the exploration randomness
# The initial state of the system use a different seed (if random)
print(f"Array n. {array_index}")
seed_qlearning = array_index
print(f'The seed used for the q_learning algorithm = {seed_qlearning}.')
if is_rerun:
    q_learning = ql.Rerun(
        q_matrix,
        seed=seed_qlearning,
        **parameters
    )
else:
    q_learning = ql.QLearning(
        #  environment=env,
        #  seed=array_index,
        seed=seed_qlearning,
        **parameters
    )
initial_action_sequence = q_learning.env.initial_action_sequence()
initial_reward = q_learning.env.reward(initial_action_sequence)

rewards = q_learning.run()

if create_output_files:
    q_learning.save_best_encountered_actions('best_gate_sequence.txt')
    q_learning.save_q_matrix('q_matrix.npy')

    n_rewards = 100
    q_learning.save_post_episode_rewards(
        'post_episode_rewards__best.npy',
        n_rewards,
        q_learning.best_encountered_actions
    )

    #  q_learning.save_post_episode_rewards(
    #      'post_episode_rewards__final.npy',
    #      n_rewards,
    #      q_learning.env.action_sequence
    #  )

    if parameters['system_class'] == 'LongRangeIsing':
        q_learning.save_post_episode_rewards(
            'post_episode_rewards__trotter.npy',
            n_rewards,
            initial_action_sequence
        )

    end_time = time.time()

    with open('rewards.npy', 'wb') as f:
        np.save(f, rewards)
    info_dic = {
        'initial_reward': initial_reward,
        'final_reward': rewards[-1],
        'best_reward': q_learning.best_encountered_reward,
        'total_time': end_time - start_time
        }
    if is_rerun:
        info_dic['rerun_path'] = str(rerun_path)
    with open('results_info.json', 'w') as f:
        json.dump(info_dic, f, indent=2)
    print("results_info.json written.")
