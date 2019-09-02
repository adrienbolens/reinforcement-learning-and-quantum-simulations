import math
import __main__

pi = math.pi

parameters = {
    'n_sites':  3,
    'n_steps': 6,
    'time_segment': 1.0,
    'bc': 'open',
    #  'ferro' or 'random'
    'initial_state': 'random_product_state',
    #  'initial_state': 'antiferro',
    #  'initial_state': 'ferro',
    #  'initial_state': None,
    #  'transmat_in_memory': False
    #  seed_env = 999
    #  'ham_params': {'J': 1.0},
    #  'system_class': 'SpSm',
    'system_class': 'LongRangeIsing',
    'n_directions': 2,
    'ham_params': {
        'J': 1.0,
        #  In order to obtain a good Trotter decomposition
        #  g: x, h: z
        'g': 2.0,
        'h': 2.0,
        'alpha': 3.0
    },
    'seed_initial_state': 42,
    #  'seed_initial_state': 0,

    # q_learning parameters:
    'n_episodes': int(1e5),
    #  'n_episodes': 10,
    'learning_rate': 0.618,
    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    #  'epsilon_decay': 0.005**(1/0.9e5)
    #  'n_replays': 200,
    #  'n_replays': 1,
    'n_replays': 10,
    'replay_spacing': 50,
    #  'replay_spacing': 100,
    #  'lam': 0.6
    #  was called `lam` before (λ)
    #  'trace_decay_rate': 0.8
}
#  epsilon_decay is such that epsilon_min is reached after pp*100% of the
#  episodes
pp = 0.9
#  pp = 2.0/3.0
parameters['epsilon_decay'] = (
    parameters['epsilon_min']/parameters['epsilon_max']
)**(1/(pp*parameters['n_episodes']))

parameters_vanilla = {
    #  always choose even to allow for identity gate
    #  Not anymore (from 0 to 2pi instead of -pi to pi)
    #  'n_onequbit_actions': 4,
    #  For LongRangeIsing: choose odd to allow for identity gate
    # actually, choose s.t. (n - 1) % 4 ==0
    'n_oqbgate_parameters': 9,

    #  always choose odd to allow for identity gate
    # n_all - 1 should be multiple of 4 for the Trotter decomp of LRI
    'n_allqubit_actions': 9,
    #  One action is the sequence "one all-to-all gate + n_sites onequbit gate"
    #  n_actions = n_onequbit_actions**n_sites * n_allqubit_actions
    'is_rerun': False
    #  'is_rerun': True
}

if parameters_vanilla['is_rerun']:
    parameters_vanilla['rerun_path'] = \
        'data3/bolensadrien/output/132_q_learning'


parameters_deep = {
    # --- for deep_q_learning:
    #  'model_update_spacing': 100,
    'model_update_spacing': 20,
    'max_q_optimizer': {
        'algorithm': 'NAG',
        'momentum': 0.9,
        'learning_rate': 0.6,
        'n_initial_actions': 15,
        # was mistakenly set to 3 up to run 229:
        'n_iterations': 10,
        'convergence_threshold': 0.005,
        'clip_action': False
        #  'clip_action': True
    },

    #  'network_type': 'LSTM',
    'network_type': 'Dense',

    'env_type': 'EnergyMinimizer',
    #  'architecture': {'LSTM': [5, 1], 'activation': None},
    'architecture': [(150, 'tanh'),
                     (40, 'relu'),
                     (20, 'relu'),
                     (1, None)],

    #  'env_type': 'DynamicalEvolution',
    #  'architecture': {'LSTM': [5, 1], 'activation': 'sigmoid'},
    #  'architecture': [(150, 'tanh'),
    #                   (40, 'relu'),
    #                   (20, 'relu'),
    #                   (1, 'sigmoid')],

    'capacity': 100,
    'sampling_size': 100,
    #  'sampling_size': 1,
    'subclass': 'WithReplayMemory',
    #  'NN_optimizer': 'adam',
    'NN_optimizer': 'SGD',
    'n_epochs': 1,
    'exploration': 'gaussian',
    #  'exploration': 'uniform'
    #  'range_one': math.pi
    'range_all': 0.5,
    'range_one': 1.0,
    #  'n_extra_episodes': 3000,
    'n_extra_episodes': 0,
    'verify_argmax_q': False
    #  'verify_argmax_q': True
}
#  parameters_deep['range_all'] = min(2 * parameters['ham_params']['J'] *
#                                     parameters['time_segment'] /
#                                     parameters['n_steps'], 1.0)

#  parameters_deep['range_one'] = min(2 * parameters['ham_params']['h'] *
#                                     parameters['time_segment'] /
#                                     parameters['n_steps'], pi)

print('parameters.py was imported with __main__ = ', __main__.__file__)
#  if __main__.__file__ == 'deep_q_learning.py':
#      parameters.update(parameters_deep)
#  elif __main__.__file__ == 'q_learning.py':
#      parameters.update(parameters_vanilla)
