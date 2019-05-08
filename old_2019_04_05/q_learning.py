import random
import numpy as np
import sys
import systems as sy
import environments as envs
#  from math import sqrt

parameters = {
    'n_sites': 3,
    'n_steps': 3,
    'time_segment': 1.0,
    'bc': 'open',
    #  'ferro' or 'random'
    'initial_state': 'random_product_state',
    #  'initial_state': None,
    #  'transmat_in_memory': False
    #  seed_env = 999
    'ham_params': {'J': 1.0, 'g': 0.0, 'h': 0.5},
    #  'ham_params': {'J': 1.0},
    #  'J': 1.0,
    #  'g': 0.0,
    #  'h': 0.5,
    #  'system': 'SpSm',
    'system': 'LongRangeIsing',
    'n_directions': 1,
    #  always choose even to allow for identity gate
    'n_onequbit_actions': 4,
    #  always choose odd to allow for identity gate
    'n_allqubit_actions': 3,
    #  One action is the sequence "one all-to-all gate + n_sites onequbit gate"
    #  n_actions = n_onequbit_actions**n_sites * n_allqubit_actions

    # q_learning parameters:
    #  'n_episodes': int(1e5),
    'n_episodes': 100000,
    'learning_rate': 0.618,
    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    'epsilon_decay': 0.99995
}

if(parameters['system'] == 'SpSm'):
    system = sy.SpSm(
        parameters['n_sites'],
        parameters['ham_params'],
        bc=parameters['bc']
    )
elif(parameters['system'] == 'LongRangeIsing'):
    system = sy.LongRangeIsing(
        n_sites=parameters['n_sites'],
        ham_params=parameters['ham_params'],
        bc=parameters['bc']
    )
else:
    ValueError('System specified not implemented')

#  params = ['bc', 'initial_state', 'J', 'n_sites', 'time_segment',
#            'n_onequbit_actions', 'n_directions', 'n_allqubit_actions',
#            'n_steps', 'n_episodes']
#  , 'transmat_in_memory', 'seed_env']
#  for p in params:
#      print(f'{p}: {eval(p)}')

env = envs.CurrentGateStateEnv(system,
                               parameters['n_steps'],
                               parameters['time_segment'],
                               parameters['initial_state'],
                               parameters['n_onequbit_actions'],
                               parameters['n_directions'],
                               parameters['n_allqubit_actions'],
                               transmat_in_memory=False)


def q_learning(n_episodes, learning_rate, epsilon_max, epsilon_min,
               epsilon_decay, seed=None, ret_q_matrix=False, **other_params):
    # very basic Q-learning algorithm:
    print("\nCreating the Q-function of size {} (state) x {} (action) = {}."
          .format(env.observation_space.n, env.action_space.n,
                  env.observation_space.n * env.action_space.n))
    q_matrix = np.zeros((env.observation_space.n, env.action_space.n),
                        dtype=np.float32)
    print(f'The Q-function takes {sys.getsizeof(q_matrix)/1e6:.1f} MB of '
          'memory.\n')
    print(f'Learning rate: {learning_rate}.')
    epsilon = epsilon_max
    print(f'Epsilon decay: {epsilon_decay}.')

    random.seed(seed)
    for episode in range(n_episodes):
        verbose = False
        if episode % (n_episodes//10) == 0:
            verbose = True
            print(f'Episode {episode}: ')
        run_episode(q_matrix, episode, learning_rate, verbose, epsilon)
        if epsilon >= epsilon_min:
            epsilon *= epsilon_decay

    print(f"Final epsilon: {epsilon:.2f}.\n")
    final_fidelity = run_episode(q_matrix, epsilon, update_q_matrix=False)
    env.render()
    print(f'\nFidelity of run with final Q: {final_fidelity:.4f}')
    if ret_q_matrix:
        return (final_fidelity, q_matrix)
    else:
        return final_fidelity


def run_episode(q_matrix, epsilon, learning_rate=None, verbose=False,
                update_q_matrix=True):
    done = False
    total_reward, reward = 0, 0
    state = env.reset()
    while not done:
        # With the probability of (1 - epsilon) take the best action in our
        # Q-table
        if random.uniform(0, 1) > epsilon or not update_q_matrix:
            action = np.argmax(q_matrix[state, :])
            # Else take a random action
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # assuming q_matrix[next_state, i] = 0 for all i.
        # Otherwise: += learning_rate * (reward - q_matrix[state, action]) if
        # done == True
        if update_q_matrix:
            q_matrix[state, action] += learning_rate *\
                (reward +
                 np.max(q_matrix[next_state]) - q_matrix[state, action])
        total_reward += reward
        state = next_state
    if verbose:
        print(f'----------Total Reward: {total_reward:.2f}, current epsilon: '
              f'{epsilon:.2f}')
    return total_reward


if __name__ == '__main__':
    import json
    # env use np.random whereas q_learning use random: the two seeds are
    # unrelated
    npseed, seed = 42, 100
    env.set_initial_state(seed=npseed, initial_state='random_product_state')
    final_fidelity, q_matrix = q_learning(seed=seed, ret_q_matrix=True,
                                          **parameters)

    with open('output.json', 'w') as f:
        json.dump({'parameters': parameters,
                   'final_fidelity': final_fidelity},
                  f, indent=2)
    with open('q_matrix.npy', 'wb') as f:
        np.save(f, q_matrix)
