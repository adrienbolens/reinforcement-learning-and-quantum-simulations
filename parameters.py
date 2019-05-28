import math
import __main__

parameters = {
    'n_sites': 3,
    'n_steps': 6,
    'time_segment': 1.0,
    'bc': 'open',
    #  'ferro' or 'random'
    'initial_state': 'random_product_state',
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

    # q_learning parameters:
    'n_episodes': int(1e4),
    #  'n_episodes': 10,
    'learning_rate': 0.618,
    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    #  'epsilon_decay': 0.005**(1/0.9e5)
    'n_replays': 100,
    #  'n_replays': 200,
    'replay_spacing': 200,
    #  'replay_spacing': 100,
    #  'lam': 0.6
    'lam': 0.8
}
#  epsilon_decay is such that epsilon_min is reached after pp*100% of the
#  episodes
pp = 0.9
#  pp = 2.0/3.0
parameters['epsilon_decay'] = (
    parameters['epsilon_min']/parameters['epsilon_max']
)**(1/(pp*parameters['n_episodes']))

parameters_deep = {
    # --- for deep_q_learning:
    'model_update_spacing': 100,
    'optimization_method': 'NAG',
    'GD_eta': 0.6,
    'GD_gamma': 0.9,
    'range_one': math.pi,
    'n_initial_actions': 13
}
parameters_deep['range_all'] = 10 * parameters['ham_params']['J'] \
    * parameters['time_segment'] / parameters['n_steps']

print('parameters.py was imported with __main__ = ', __main__.__file__)
if __main__.__file__ == 'deep_q_learning.py':
    parameters.update(parameters_deep)
