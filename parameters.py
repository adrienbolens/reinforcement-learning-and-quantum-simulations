parameters = {
    'n_sites': 3,
    'n_steps': 6,
    'time_segment': 100.0,
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
        'h': 2.0
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
    'n_episodes': int(1e5),
    #  'n_episodes': 10,
    'learning_rate': 0.618,
    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    #  'epsilon_decay': 0.005**(1/0.9e5)
    'n_replays': 100,
    'replay_spacing': 200,
    #  'lam': 0.6
    'lam': 0.8,
    'model_update_spacing': 100
}
#  epsilon_decay is such that epsilon_min is reached after 90% of the episodes
parameters['epsilon_decay'] = (
    parameters['epsilon_min']/parameters['epsilon_max']
)**(1/(0.9*parameters['n_episodes']))

#  parameters['ham_params'] = {
#      'J': 1.0,
#      In order to obtain a good Trotter decomposition
#      'g': 2*np.pi*parameters['n_steps']/parameters['n_oqbgate_parameters'],
#      'h': 2*np.pi*parameters['n_steps']/parameters['n_oqbgate_parameters']
#  }
