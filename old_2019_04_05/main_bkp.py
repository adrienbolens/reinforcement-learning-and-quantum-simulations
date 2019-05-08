import systems as sy
import environments as envs
import random
import numpy as np
import sys
#  from math import sqrt

if len(sys.argv) > 1:
    array_seed = sys.argv[1]
else:
    array_seed = None
n_tests = 1
n_episodes = 50000
#  n_episodes = 100

n_sites = 3
n_steps = 3
time_segment = 1.0
bc = 'open'
#  'ferro' or 'random'
initial_state = 'random'
transmat_in_memory = False
seed = 999
J = 1.0
system = sy.SpSmSystem(n_sites, J=J, bc=bc)
#  system = system_3sites()
#  system2 = system_2sites()
#  env.initialize_env(num_steps, system['ham'], system['gates'])

#  always choose even to allow for identity gate
n_onequbit_actions = 13
n_directions = 1
#  always choose odd to allow for identity gate
n_allqubit_actions = 6
#  Assuming all single-qubit gates can act independently (i.e. no symmetry
#  implemented)
#  One action is the sequence "one all-to-all gate + n_sites onequbit gate"
#  n_actions = n_onequbit_actions**n_sites * n_allqubit_actions

#  print(f'\nSpSm System with {n_sites} sites, {bc} B.C., J={J:.1f}, and '
#        f'time_segment={time_segment}.')
#  print(f'Enviroment with "current gate" states, {n_onequbit_actions}
#  one-qubit '
#        f'gates (between -pi and pi) in {n_directions} directions, '
#        f'and {n_allqubit_actions} all-to-all gates '
#        f'(between -Jt/2 and Jt/2)')
#  print(f'# steps = {n_steps}. (One step = one all-to-all gates and a
#  one-qubit '
#        'gate on each site.')
#  print(f'\n# episodes = {n_episodes}.\n# test episode = {n_tests}.')

print('System: SpSm')
params = ['bc', 'initial_state', 'J', 'n_sites', 'time_segment',
          'n_onequbit_actions', 'n_directions', 'n_allqubit_actions',
          'n_steps', 'n_episodes', 'transmat_in_memory', 'seed']
for p in params:
    print(f'{p}: {eval(p)}')

env = envs.CurrentGateStateEnv(system, n_steps,
                               time_segment,
                               initial_state,
                               n_onequbit_actions,
                               n_directions,
                               n_allqubit_actions,
                               transmat_in_memory,
                               seed)
state = env.reset()

# very basic Q-learning algorithm:
print("\nCreating the Q-function of size {} (state) x {} (action) = {}."
      .format(env.observation_space.n, env.action_space.n,
              env.observation_space.n * env.action_space.n))
Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
print(f'The Q-function takes {sys.getsizeof(Q)/1e6:.1f} MB of memory.\n')
G = 0  # accumulated reward
alpha = 0.618  # learning rate
print(f'Learning rate: {alpha}.')
epsilon = 1.0  # Greed 0% (purely random choice of action)
epsilon_min = 0.005  # maximum greed 99.5%
epsilon_decay = 0.99992  # multiplies epislon after each episode
print(f'Epsilon decay: {epsilon_decay}.')


def run_episode(episode, epsilon, render=False):
    done = False
    G, reward = 0, 0
    state = env.reset()
    if render:
        env.render()
    while not done:
        # With the probability of (1 - epsilon) take the best action in our
        # Q-table
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[state, :])
            # Else take a random action
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # assuming Q[next_state, i] = 0 for all i.
        # Otherwise: += alpha * (reward - Q[state, action]) if done == True
        Q[state, action] += alpha * (reward + np.max(Q[next_state])
                                     - Q[state, action])
        G += reward
        state = next_state
        if render:
            env.render()
    if episode % 10000 == 0:
        print(f'Episode {episode} Total Reward: {G:.2f}, current epsilon: '
              f'{epsilon:.2f}')


random.seed(array_seed)
for episode in range(n_episodes):
    run_episode(episode, epsilon)
    # Reducing our epsilon each episode (Exploration-Exploitation trade-off)
    if epsilon >= epsilon_min:
        epsilon *= epsilon_decay

print(f"Final epsilon: {epsilon:.2f}.\n")

#  test_state = [1, 4, 5, 1, 2, 3]
#  solution = [3, 1, 2, 1]

#  print("reward of solution = ", env.reward(env.encode(solution)))

r_av = 0.0
#  r2 = 0.0
for i in range(n_tests):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state, :])
        ns, r, done, _ = env.step(action)
        state = ns
    r_av += r
    #  r2 += r**2
    if i >= n_tests - 1:
        env.render()

r_av /= n_tests
#  std = r2/(n_tests - 1) - r_av**2*(n_tests/(n_tests-1))
#  std = sqrt(abs(std))

print(f'\nFidelity of run with final Q: {r_av:.4f}')
#  print(f'\nAverage fidelity of final agent (after {n_tests} runs) =
#  {r_av:.4f}')
#  print(f'Standard deviation = {std:.4f}')

#  run_episode(True)
