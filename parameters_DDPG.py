import math
import __main__

pi = math.pi

parameters = {
    # =======================================================================
    # physical system
    # =======================================================================
    'n_sites':  3,
    'n_steps': 4,
    'time_segment': 1.0,
    'bc': 'open',
    #  'ferro' or 'random'
    'initial_state': 'random_product_state',
    #  'initial_state': 'antiferro',
    #  'initial_state': 'ferro',
    #  'system_class': 'SpSm',
    'system_class': 'LongRangeIsing',
    'n_directions': 2,
    'ham_params': {
        'J': 1.0,
        #  g: x, h: z
        'g': 2.0,
        'h': 2.0,
        'alpha': 3.0
    },
    'seed_initial_state': 42,

    # =======================================================================
    # environment and reinforcement learning
    # =======================================================================
    'env_type': 'DynamicalEvolution',
    'n_episodes': int(5e4),
    #  'n_episodes': 10,
    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    # epsilon_decay is set automatically (see below)

    'n_replays': 10,
    'replay_spacing': 50,
    'exploration': 'gaussian',
    #  'exploration': 'uniform'
    'n_extra_episodes': 3000,

    'range_all': 0.5,
    'range_one': 1.0,


    # =======================================================================
    # neural networks
    # =======================================================================
    #  'network_type': 'LSTM',
    'network_type': 'Dense',

    'capacity': 50,
    'sampling_size': 50,
    #  'sampling_size': 1,
    'optimizer': 'adam',
}

#  epsilon_decay is such that epsilon_min is reached after pp*100% of the
#  episodes
pp = 0.9
#  pp = 2.0/3.0
parameters['epsilon_decay'] = (
    parameters['epsilon_min']/parameters['epsilon_max']
)**(1/(pp*parameters['n_episodes']))


#  parameters_deep['range_all'] = min(2 * parameters['ham_params']['J'] *
#                                     parameters['time_segment'] /
#                                     parameters['n_steps'], 1.0)

#  parameters_deep['range_one'] = min(2 * parameters['ham_params']['h'] *
#                                     parameters['time_segment'] /
#                                     parameters['n_steps'], pi)

print('parameters_DDPG.py was imported with __main__ = ', __main__.__file__)
