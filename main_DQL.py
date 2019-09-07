import numpy as np
import sys
import deep_q_learning as dql
import json
from pathlib import Path

import time
start_time = time.time()

if Path('info.json').is_file():
    with open('info.json') as f:
        info = json.load(f)
    parameters = info['parameters']
else:
    from parameters import parameters, parameters_deep
    parameters.update(parameters_deep)

if len(sys.argv) < 2:
    array_index = 1
    create_output_files = False
    print("Output files won't be created.")
else:
    array_index = int(sys.argv[1])
    create_output_files = True
    print("Output files will be created.")

# The seed here is for the exploration randomness
# The initial state of the system use a different seed (if random)
print(f"Array n. {array_index}")
seed_qlearning = array_index
print(f'The seed used for the q_learning algorithm = {seed_qlearning}.')
if parameters['subclass'] == 'WithReplayMemory':
    q_learning = dql.DQLWithReplayMemory(
        seed=seed_qlearning,
        **parameters
    )
else:
    raise NotImplementedError(f"subclass {parameters['subclass']} in "
                              'parameters.py not recognized.')

initial_action_sequence = q_learning.env.initial_action_sequence()
initial_reward = q_learning.env.reward(action_sequence=initial_action_sequence)

ground_state_energy = q_learning.env.system.ground_state_energy()

rewards = q_learning.run()

if create_output_files:
    q_learning.save_best_encountered_actions('best_gate_sequence.txt')
    q_learning.save_weights('final_weights.npy')
    q_learning.save_lists_q_max('list_q_max.npy')

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
    if parameters['subclass'] == 'WithReplayMemory':
        q_learning.save_history('NN_history.csv')

    end_time = time.time()

    try:
        with open('rewards.npy', 'wb') as f:
            np.save(f, rewards)
    except Exception as e:
        print('`rewards.npy` could not be saved.')
        print('--->', e)

    info_dic = {
        #  'parameters': parameters,
        'initial_reward': initial_reward,
        'ground_state_energy': ground_state_energy,
        #  'final_reward': rewards[-1],
        'best_reward': q_learning.best_encountered_rewards,
        'total_time': end_time - start_time
        }
    #  print("Compare 'best_encountered_reward' = "
    #        f"{q_learning.best_encountered_reward:.5f} with 'max(rewards)
    #        = "
    #        f"{np.max(rewards):.5f}.")
    try:
        with open('results_info.json', 'w') as f:
            json.dump(info_dic, f, indent=2)
        print("results_info.json written.")
    except Exception as e:
        print('`results_info.json` could not be saved.')
        print('--->', e)
