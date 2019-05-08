import systems as sy
import environments as envs
#  import random
#  import gym
#  import numpy as np

n_sites = 2
n_steps = 2
time_segment = 1.0
system = sy.SpSmSystem(n_sites, J=1.0)
n_onequbit_actions = 8
n_allqubit_actions = 3

env = envs.CurrentGateStateEnv(system, n_steps,
                               time_segment,
                               n_onequbit_actions,
                               n_allqubit_actions)

#  An exact solution shoulb be
#  [sz0(0.25*pi)][sz1(0.25*pi)][sxsx(-0.5)]
#  [sz0(-0.25*pi)][sz1(-0.25*pi)][sxsx(-0.5)]

#  action_sequence corresponding to the solution:

a1 = 5 + 5*n_onequbit_actions + 2*n_onequbit_actions**2
a2 = 3 + 3*n_onequbit_actions + 2*n_onequbit_actions**2
print(a1)
action_sequence = [a1, a2]
env.action_sequence = action_sequence
print(env.reward())
env.render()
