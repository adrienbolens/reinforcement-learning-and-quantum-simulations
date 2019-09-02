#  import physics as ph
import sys
import systems as sy
import numpy as np
#  from discrete import Discrete
from math import pi
import warnings
#  import random
#  from scipy.linalg import expm
from scipy.sparse.linalg import expm


class QuantumEnv:
    """Basic environement. Contains only information concerning the system and
    the evolution (n_steps and time_segment), and basic methods.
    Information concerning states and actions are in the subclasses.

    -Description:
        Given an initial state, a Hamiltonian and a final time T, the purpose
        of the agent is to find a sequence of N gates that reproduces the time
        evolution after a time T = time_segment.

    -Obsevations: see subclasses.
        Main categories are:
            Discrete (for tabular Q-learning)
            Continuous (for Deep QL)

    -Actions:
        Actions correspond to choosing a gate (or a set of gates)

    -Rewards:
        There is a reward of R at the end of the sequence where R physically
        quantifies how close the final state is from the target state (e.g.
        fidelity = |<phi_target|phi_final>|^2.
        Only the state at the final time T is relevant.

    ------------------------------------------
    Physical states:
        Not used explicitly in the QL algorithm, but needed for the
        environement to properly return the rewards.
        => related calculations are in the system.py module

    (1) The initial state.
    (2) The final state calculated with the exact Hamiltonian.
    (3) The final state after applying the sequence of gates.

    (1) and (2) are evaluated before the Q-learning, and (3) when asking
    for the reward at t=T.
    """

    def __init__(self, n_steps,
                 time_segment,
                 initial_state,
                 n_directions,
                 seed_initial_state=None,
                 initiate_target_state=True,
                 **other_params):

        # self.system defined in subclass, as they have different
        # store_gates values depending

        self.n_steps = n_steps
        self.time_segment = time_segment
        self.unitary_evolution = \
            expm(-1j * self.time_segment * self.system.hamiltonian)

        self.set_initial_state(seed_initial_state,
                               initial_state,
                               initiate_target_state)

        # use single-qubit gates in one (z-only) or two (z and x) directions
        self.n_directions = n_directions
        self.action_sequence = None
        self.state_sequence = None
        self.lastaction = None
        self.s = None

    def set_initial_state(self, seed=None, initial_state='ferro',
                          initiate_target_state=True):
        if initial_state is None:
            pass
        elif initial_state == 'random_product_state':
            self.system.random_state(product_state=True, seed=seed,
                                     inplace=True)
        elif initial_state == 'random':
            self.system.random_state(product_state=False, seed=seed,
                                     inplace=True)
        elif initial_state == 'ferro':
            self.system.ferro_state(inplace=True)
        elif initial_state == 'antiferro':
            self.system.antiferro_state(inplace=True)
        else:
            raise ValueError(f'Initial state of type {initial_state} '
                             'not implemented.')

        self.initial_state = self.system.state
        #  print(f'The initial state is {self.initial_state}')
        if initiate_target_state:
            self.target_state = \
                self.system.evolve(unitary=self.unitary_evolution)

    def get_gate_sequence(self, action_sequence=None):
        if action_sequence is None:
            action_sequence = self.action_sequence
        gates = []
        for a in action_sequence:
            gates += self.decode_action(a)
        return gates

    def render(self, outfile=sys.stdout, action_sequence=None):
        gates = self.get_gate_sequence(action_sequence)
        m = 1 + self.system.n_sites * self.n_directions
        for n in range(self.n_steps):
            outfile.write(f'Step {n}:\n {str(gates[n*m:(n+1)*m])}\n\n')

    def step(self, a, calculate_reward=True):
        if self.action_sequence is None:
            raise NotImplementedError('action_sequence not initialized'
                                      'reset() should have been called.')
        if self.state_sequence is None:
            raise NotImplementedError('state_sequence not initialized'
                                      'reset() should have been called.')
        self.state_sequence.append(self.s)
        self.action_sequence.append(a)
        next_state, done = self.get_transition(self.s, a)
        self.s = next_state
        self.lastaction = a
        if done and calculate_reward:
            reward = self.reward(self.action_sequence)
        else:
            #  reward = 0.0 if a == 0 else -0.001
            reward = 0.0
        if calculate_reward:
            self.reward_sequence.append(reward)
        return (next_state, reward, done, {})

    def get_transition(self, state, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def decode_action(self, a):
        raise NotImplementedError

    def reward(self, action_sequence=None, n_rewards=None):
        """
        Does the following:
            - Evolve `system.state` from the `initial_state` using
            `action_sequence`
            - Compare it with `target_state` of the exact evolution

        if n_rewards > 1, repeat n_rewards times
        and give two sets of rewards: comparing with
          - target_state evolved from initial_state
          - target_state evolved starting from last state of previous step
        """

        if action_sequence is None:
            action_sequence = self.action_sequence
        self.system.state = self.initial_state
        sets_of_gates = [self.decode_action(a) for a in action_sequence]

        if n_rewards is None:
            for gates in sets_of_gates:
                self.system.apply_gates(gates, inplace=True)
            return self.system.compare_states(self.target_state)
        else:
            target_state1 = self.initial_state
            r1, r2 = [], []
            for n in range(n_rewards):
                target_state1 = self.system.evolve(
                    unitary=self.unitary_evolution,
                    #  time=self.time_segment,
                    starting_state=target_state1
                )
                target_state2 = self.system.evolve(
                    unitary=self.unitary_evolution
                    #  time=self.time_segment
                )
                for gates in sets_of_gates:
                    self.system.apply_gates(gates, inplace=True)
                r1.append(self.system.compare_states(target_state1))
                r2.append(self.system.compare_states(target_state2))
            return (r1, r2)


class DiscreteQuantumEnv(QuantumEnv):
    """
    The number of available gates is fixed
    <=> The parameters take discrete values.

    Use one of the following subclass:
        -FullSequenceState
        -CurrentGateState

    FullSequenceState:
        There are 1 + M + M**2 + ... + M**N = [1-M**(N+1)]/[1-M] states, where
        M is the number of available gates, since a state are completely
        characterized by the sequence of gates, given a fixed initial state.
        (N=n_steps)

    CurrentGateState:
        There are M * n_steps states, where M is the number of available gates
        and n_steps is the number of steps (the "time" variable).
    """
    def __init__(self,
                 n_oqbgate_parameters,
                 n_allqubit_actions,
                 system_class='LongRangeIsing',
                 transmat_in_memory=False,
                 **other_params):

        self.system_class = system_class
        if system_class == 'SpSm':
            self.system = sy.SpSm(store_gates=True, **other_params)
        elif system_class == 'LongRangeIsing':
            self.system = sy.LongRangeIsing(store_gates=True, **other_params)
        else:
            ValueError('System specified not implemented')
        super().__init__(**other_params)

        self.transmat_in_memory = transmat_in_memory
        self.n_oqbgate_parameters = n_oqbgate_parameters
        #  # acting with a single-qubit gate in one or the other direction:
        #  self.n_onequbit_actions = n_oqbgate_parameters * n_directions

        #  # acting with a single-qubit gate in both directions:
        self.n_onequbit_actions = n_oqbgate_parameters ** self.n_directions
        self.n_allqubit_actions = n_allqubit_actions
        #  self.nA = (self.n_onequbit_actions ** self.system.n_sites) * \
        #      n_allqubit_actions

        #  The same gates for each site:
        self.nA = self.n_onequbit_actions * n_allqubit_actions

        self.action_space = Discrete(self.nA)
        self.observation_space = None
        self.transition_matrix = None

        #  K and Delta are the discretized parameters used for the gate
        #  U_all = e^(-i K[i] ham_all), U_one = e^(-i Delta[i] ham_one)
        if self.n_allqubit_actions % 2 == 0:
            warnings.warn('The all-bit identity gate is not available.')
        Jt = self.time_segment * self.system.J
        self.K = np.linspace(-2 * Jt / self.n_steps, 2 * Jt / self.n_steps,
                             self.n_allqubit_actions, endpoint=True)

        if self.system_class == 'LongRangeIsing':
            if self.n_oqbgate_parameters % 2 == 0:
                warnings.warn('The single-bit identity gate is not available.')
            ht = self.time_segment * self.system.ham_params['h']
            self.Delta = np.linspace(-2 * ht / self.n_steps,
                                     2 * ht / self.n_steps,
                                     self.n_oqbgate_parameters,
                                     endpoint=True)
        else:
            self.Delta = np.linspace(0, 2 * pi, self.n_oqbgate_parameters,
                                     endpoint=False)
        self.reset()

    def reset(self):
        self.s = 0
        self.action_sequence = []
        self.state_sequence = []
        self.reward_sequence = []
        self.lastaction = None
        return self.s

    def get_transition(self, state, action):
        if self.transmat_in_memory:
            return self.transition_matrix[state, action]
        else:
            return self.get_nextstate(state, action)

    def initial_action_sequence(self, inplace=False):
        if self.system_class == 'LongRangeIsing':
            # K[i] = Jt/n_steps with K = [-2Jt/n_steps, ..., 2Jt/n_steps]
            # -> i = 3/4 * (n_allqubit_actions - 1)
            # delta[i] = gt/n_steps with delta = [0, ..., (n_one-1)2pi/n_one]
            # i_x = m_g, where gt = m_g *n_steps/n_one * 2*pi
            # same with ht (and i_z = m_h)
            # Now assuming m_g = m_h = 1
            #  Update: now single-bits gate are set as the all-bit:
            #  Delta = [-2ht/n_steps, ..., 2ht/n_steps]
            if (self.n_allqubit_actions - 1) % 4 != 0:
                raise ValueError('Trotter decomposition not possible with the '
                                 'current value of n_allqubit_actions')
            i_all = 3 * (self.n_allqubit_actions - 1) // 4

            #  gt = self.system.ham_params['g']*self.time_segment
            #  ht = self.system.ham_params['h']*self.time_segment
            #  i_x =
            #  int(round(0.5*gt*self.n_oqbgate_parameters/self.n_steps/pi))
            #  i_z =
            #  int(round(0.5*ht*self.n_oqbgate_parameters/self.n_steps/pi))
            #  if not i_x == 1 == i_z:
            #      raise ValueError('Unexpected values for the g & h
            #      parameters.')

            if (self.n_oqbgate_parameters - 1) % 4 != 0:
                raise ValueError('Trotter decomposition not possible with the '
                                 'current value of n_oqbgate_parameters')
            i_z = 3 * (self.n_oqbgate_parameters - 1) // 4
            if self.n_directions == 2:
                if self.system.ham_params['g'] != self.system.ham_params['h']:
                    raise ValueError('Unexpected values for the g & h '
                                     'parameters. The Trotter decomp assumes '
                                     'they are equal.')
                i_x = i_z
            if self.n_directions == 1:
                if self.system.ham_params['g'] != 0:
                    raise ValueError('Unexpected values for the g '
                                     'parameter. The Trotter decomp assumes '
                                     'g == 0.')
                i_x = (self.n_oqbgate_parameters - 1) // 2

            # the corresponding action is
            # a = sum_{l=0}^{n_sites-1} i_l * n_one**l + i_all * n_one**n_sites
            # see decode_action
            # if g != 0 (sigma_x), then n_directions == 2 and
            # i_l = i_one (same for all sites in this case)
            # i_one = 0, ..., n_1params ** 2 - 1: i_one = ix * n_1params + iz
            # i_one = 0, ..., n_one
            # ix, iz = 0, ..., n_params - 1
            if self.n_directions == 2:
                i_one = i_x * self.n_oqbgate_parameters + i_z
            elif self.n_directions == 1:
                i_one = i_z
            # 1 + n + ... + n**(n_sites-1) == (n**n_sites -1)/(n-1)

            #  action = i_one *\
            #      (self.n_onequbit_actions**self.system.n_sites - 1) // \
            #      (self.n_onequbit_actions - 1) + \
            #      i_all * self.n_onequbit_actions**self.system.n_sites

            #  Assume same 1-gate on all sites: a = i_one + n_one * i_all
            action = i_one + i_all * self.n_onequbit_actions
            if inplace:
                self.action_sequence = [action] * self.n_steps
            return [action] * self.n_steps
        else:
            #  return [random.randint(0, self.nA - 1)
            #          for _ in range(self.n_steps)]
            return np.random.randint(0, self.nA - 1, size=self.n_steps)
            #  raise NotImplementedError

    def decode_allqubit_gate(self, i):
        """
        Gates are of the form e^{-iK sum_ij s_i s_j} where K takes values
        #  between -Jt/2 and +Jt/2.
        between -2 Jt/n_steps and +2 Jt/n_steps
        """
        return self.system.allqubit_gate(self.K[i], 'sxsx')

    def decode_onequbit_gate(self, site, i):
        """ Gates are of the form e^{-i Delta s_i} where
        #  Delta is between -pi and pi. update: 0 and pi for SmSp
        update: now gate between -2 ht/n_steps and +2 ht/n_steps (included)
        for LongRangeIsing"""
        if self.n_directions == 1:
            return [self.system.onequbit_gate(site, self.Delta[i], 'sz')]
        #  # acting with a single-qubit gate in one or the other direction:
        #  elif self.n_directions == 2:
        #      _s = 'sz' if i//self.n_oqbgate_parameters == 0 else 'sx'
        #      i = i % self.n_oqbgate_parameters
        #      return [self.system.onequbit_gate(site, Delta[i], _s)]

        # acting with a single-qubit gate in both directions:
        elif self.n_directions == 2:
            # i = 0, ..., n_params ** 2 - 1: i = ix * n_params + iz
            # ix, iz = 0, ..., n_params - 1
            ix = i // self.n_oqbgate_parameters
            iz = i % self.n_oqbgate_parameters
            return [self.system.onequbit_gate(site, self.Delta[iz], 'sz'),
                    self.system.onequbit_gate(site, self.Delta[ix], 'sx')]
        else:
            raise ValueError('one-qubit gates for #directions other than 1 '
                             'or 2 are not implemented')

    def decode_action(self, i):
        """Given an integer 0<= i < nA, return the appropriate set of n_sites
        onequbit_gate and single allqubit_gate."""
        #  i -> (i_0, i_1, ..., i_n_sites-1, i_alltoall)
        #  i = sum_{j=0}^{n_sites-1}  i_j * n_one**j + i_alltoall *
        #                                              n_one ** n_sites
        #  i in range(n_one ** n_sites * n_all)
        #  i_j in range(n_one)
        #  i_alltoall in range(n_all)
        #  n_one = n_onequbit_actions
        #  n_all = n_allqubit_actions
        list_gates = []

        #  #  different gates on each site:
        #  for site in range(self.system.n_sites):
        #      gate = self.decode_onequbit_gate(site, i %
        #      self.n_onequbit_actions)
        #   CHECK ORDER: site then x/z or x/z then site
        #      list_gates.extend(gate)
        #      i = i // self.n_onequbit_actions

        #  same gates on each site:
        gates_z, gates_x = [], []
        for site in range(self.system.n_sites):
            gate = self.decode_onequbit_gate(site, i % self.n_onequbit_actions)
            gates_z.append(gate[0])
            if self.n_directions == 2:
                gates_x.append(gate[1])

        list_gates.extend(gates_z)
        if self.n_directions == 2:
            list_gates.extend(gates_x)

        i = i // self.n_onequbit_actions
        gate = self.decode_allqubit_gate(i)
        list_gates.insert(0, gate)
        #  list_gates.append(gate)
        return list_gates


class FullSequenceStateEnv(DiscreteQuantumEnv):
    """The state is represetend by the full sequence of gates"""

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)

        self.nS = (1 - self.nA**(self.n_steps + 1)) // (1 - self.nA)
        self.observation_space = Discrete(self.nS)

        if self.transmat_in_memory:
            print(f'\nCreating the transition matrix, an array of {self.nS} x '
                  f'{self.nA} = {self.nS*self.nA} tuples of the form (state, '
                  f'done)')
            self.transition_matrix = np.empty((self.nS, self.nA), dtype=object)
            for state in range(self.nS):
                for action in range(self.nA):
                    self.transition_matrix = self.get_nextstate(state, action)
            print(f'The transition matrix takes '
                  f'{sys.getsizeof(self.transition_matrix)/1e6:.1f} '
                  f'MB of memory.')

    def get_nextstate(self, state, action):
        ns_decoded = self.decode_state(state)
        ns_decoded.append(action)
        next_state = self.encode_state(ns_decoded)
        return (next_state, len(ns_decoded) == self.n_steps)

    def encode_state(self, gates):
        """gates is a list of int: [g1, g2, g3,...gm] where m<=n_steps."""
        # [] -> i = 0 | "dim" = 1
        # [g] -> i = 1 + g | "dim" = M
        # [g1, g2] -> i = (1 + M) + M*g2 + g1 | "dim" = M**2
        # [g1, g2, g3] -> i = (1 + M + M**2) + M**2*g3 + M*g2 + g1, etc
        n_gates = len(gates)
        offset = (self.nA**n_gates - 1) // (self.nA - 1)
        i = 0
        for _ in reversed(gates):
            i *= self.nA
            i += _
        i += offset
        return i

    def decode_state(self, i):
        n_gates = -1
        z = i * (self.nA - 1) + 1
        while z != 0:
            z = z // self.nA
            n_gates += 1
        out = []
        i -= (self.nA**n_gates - 1) // (self.nA - 1)
        for _ in range(n_gates):
            out.append(i % self.nA)
            i = i // self.nA
        return out


class CurrentGateStateEnv(DiscreteQuantumEnv):
    """The state is represetend by the last gate in the current sequence of
    gates and the current time"""

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.nS = self.n_steps * self.nA + 1
        self.observation_space = Discrete(self.nS)
        if self.transmat_in_memory:
            print(f'Creating the transition matrix, an array of {self.nS} x '
                  f'{self.nA} = {self.nS*self.nA} tuples of the form (state, '
                  'done).')
            self.transition_matrix = np.empty((self.nS, self.nA), dtype=object)
            for state in range(self.nS):
                for action in range(self.nA):
                    self.transition_matrix[state, action] = \
                        self.get_nextstate(state, action)
            print(f'The transition matrix takes '
                  f'{sys.getsizeof(self.transition_matrix)/1000:.1f} '
                  'MB of memory.')

    def get_nextstate(self, state, action):
        state_decoded = self.decode_state(state)
        if state_decoded == 'empty':
            done = (self.n_steps <= 1)
            if done:
                print('n_steps is <= 1...')
            return (action + 1, done)
        else:
            _, step = state_decoded
            done = (step >= self.n_steps - 2)
            next_state = (step + 1) * self.nA + action + 1
            return (next_state, done)

    def encode_state(self, state):
        if state == 'empty':
            return 0
        else:
            action, time = state
            return self.nA * time + action + 1

    def decode_state(self, i):
        """return (action, time) or 'empty'."""
        if i == 0:
            return 'empty'
        else:
            i -= 1
            return (i % self.nA, i // self.nA)


class ContinuousQuantumEnv(QuantumEnv):
    """
    Actions are lists [a_all, a_0, ..., a_(dir*n_sites -1)] characterizing
    all the gates (one all-to-all and n_directions * n_sites single-qubit gates

    a_i takes values between -1 and 1: a_i == 0 is the identity (e^(-i*0*ham))
    the gate parameter at -1 and 1 correspond to +- range_one or range_all
    U = e^(-i* a*range * ham)
    """

    def __init__(self,
                 system_class='LongRangeIsing',
                 range_one=None,
                 range_all=None,
                 **other_params):

        self.system_class = system_class
        if system_class == 'SpSm':
            self.system = sy.SpSm(store_gates=False, **other_params)
        elif system_class == 'LongRangeIsing':
            self.system = sy.LongRangeIsing(store_gates=False, **other_params)
        else:
            ValueError('System specified not implemented')
        super().__init__(**other_params)

        if range_one is None:
            if system_class == 'LongRangeIsing':
                range_one = (2 * self.system.ham_params['h']
                             * self.time_segment / self.n_steps)
            else:
                range_one = pi
        if range_all is None:
            range_all = (2 * self.system.ham_params['J']
                         * self.time_segment / self.n_steps)

        self.range_one = range_one
        self.range_all = range_all

        # actions are lists of length:
        self.action_len = self.n_directions * self.system.n_sites + 1
        self.reset()

    def initial_action_sequence(self):
        #  return [ [ai,...], [...], ...], the list of action corresponding to
        #  Trotter or random...
        #  = [[a_trotter1, a_2, ...]]*n_steps WARNING: all element same object
        #  all pointing to same memory
        #  gate = e^(-i * a * ham)

        if self.system_class == 'LongRangeIsing':
            if self.n_directions == 2:
                a_all = self.system.ham_params['J'] * self.time_segment \
                    / self.n_steps / self.range_all
                a_onex = self.system.ham_params['g'] * self.time_segment \
                    / self.n_steps / self.range_one
                a_onez = self.system.ham_params['h'] * self.time_segment \
                    / self.n_steps / self.range_one
                action = [a_all] + [a_onez] * self.system.n_sites \
                    + [a_onex] * self.system.n_sites
                return [action] * self.n_steps
            else:
                raise NotImplementedError('Trotter sequence only implemented'
                                          ' for n_directions = 2.')
        else:
            #  return [[random.uniform(-1, 1) for _ in range(self.action_len)]
            #          for _ in range(self.n_steps)]
            return np.random.uniform(-1, 1, size=(self.n_steps,
                                                  self.action_len))

    def decode_action(self, action):
        """ Given [..., ai, ...], return list of gates [, ... Gi, ...]"""
        list_gates = []
        n = self.system.n_sites
        list_gates.append(self.decode_allqubit_gate(action[0]))
        for site, a in enumerate(action[1:n + 1]):
            list_gates.append(self.decode_onequbit_gate(site, a, 'sz'))

        for site, a in enumerate(action[n + 1:2 * n + 1]):
            list_gates.append(self.decode_onequbit_gate(site, a, 'sx'))

        return list_gates

    def decode_onequbit_gate(self, site, a, kind):
        """ Given -1 < a < 1, return the gate"""
        a_scaled = a * self.range_one
        return self.system.onequbit_gate(site, a_scaled, kind)

    def decode_allqubit_gate(self, a):
        #  given -1< a< 1, return the gate
        a_scaled = a * self.range_all
        return self.system.allqubit_gate(a_scaled, 'sxsx')

    def random_action(self):
        return np.random.uniform(-1, 1, size=self.action_len)


class ContinuousCurrentGateEnv(ContinuousQuantumEnv):

    """ use the form (step, [a_all, a_z0, ..., a_zn-1, a_x0, ..., a_xn-1])
    for the states (also self.s), where [...] is the lastaction.
    step = 0, ..., n_steps - 1.
    the initial state before any action is (-1, [0, 0, ..., 0])
    """

    def get_initial_state(self):
        return (-1, np.zeros(self.action_len))

    def reset(self):
        s0 = self.get_initial_state()
        self.s = s0
        self.action_sequence = []
        self.state_sequence = []
        self.reward_sequence = []
        self.lastaction = None
        self.system.state = self.initial_state
        return s0
        #  return self.process_state(self.s)

    def get_transition(self, state, action):
        #  return (next_state, done)
        step, _ = state
        step += 1
        #  the last states have step = n_steps - 1
        done = (step >= self.n_steps - 1)
        return ((step, action), done)

    def get_n_inputs(self) -> int:
        """return the number of input neurons for the NN"""
        return self.n_steps + 2 * self.action_len

    def get_state_len(self) -> int:
        return self.n_steps + self.action_len

    def get_action_len(self) -> int:
        return self.action_len

    def process_state_action(self, state, action=None, reshape=True):
        if action is None:
            action = np.zeros(self.action_len)
        #  Give a state feedable to the NN
        #  one-hote encode the time
        step, current_action = state
        one_hot_step = np.zeros(self.n_steps, dtype=np.float32)
        if step == self.n_steps - 1:
            # there is no need to calculate the Q function for terminal states.
            pass
        else:
            # here, step from -1 to n_steps - 2
            one_hot_step[step + 1] = 1.0
        network_input = np.concatenate(
            (one_hot_step, np.array(current_action, dtype=np.float32),
             np.array(action, dtype=np.float32))
        )
        if reshape:
            return network_input.reshape(1, -1)
        else:
            return network_input

    def process_action(self, processed_sa, action):
        """
        Change the input corresponding to the action of an already
        processed 'state-action' network input
        """
        processed_sa[0, self.n_steps + self.action_len:] = action
        return processed_sa

    def inputs_from_sequence(self, states, actions):
        #  states = self.get_states_from_action_sequence(action_sequence)
        #  states = np.insert(list(enumerate(action_sequence[:-1])), 0,
        #                     (-1, [0] * self.action_len))

        n_inputs = self.get_n_inputs()
        network_input = np.zeros((self.n_steps, n_inputs))
        for i, s, a in zip(range(len(actions)), states, actions):
            network_input[i, :] = self.process_state_action(a, s,
                                                            reshape=False)
        return network_input

    def get_states_from_action_sequence(self, action_sequence=None):
        if action_sequence is None:
            action_sequence = self.action_sequence
        return np.insert(list(enumerate(action_sequence[:-1])), 0,
                         (-1, [0] * self.action_len))

    #  def pad_list_of_inputs(self, inputs):
    #      """
    #      Add trivial inputs to the list `inputs` until it has length
    #      `self.n_steps`.
    #      This is useful for LSTM networks, which require the full sequence of
    #      individual inputs (i.e. all steps in the episodes).
    #      """
    #      for i in range(self.n_steps - len(inputs)):
    #          inputs.append(np.zeros(


class ContinuousCurrentStateEnergyEigensolver(ContinuousCurrentGateEnv):

    def __init__(self, **other_params):
        super().__init__(initiate_target_state=False,
                         **other_params)

    def reset(self):
        s0 = self.get_initial_state()
        self.s = s0
        self.action_sequence = []
        self.state_sequence = []
        self.reward_sequence = []
        self.lastaction = None
        self.system.state = self.initial_state
        return s0

    def reward(self,
               action_sequence=None,
               action=None,
               from_initial_state=False):
        """
        Does the following:
            - Evolve `system.state` from the `initial_state` using
            `action_sequence`.
            - calculate the energy of the final state
        """

        # not really needed: system.sate is reset in reset()
        if from_initial_state:
            self.system.state = self.initial_state

        if action is not None and action_sequence is not None:
            raise ValueError

        if action is not None:
            gates = self.decode_action(action)
            self.system.apply_gates(gates, inplace=True)
            return -self.system.get_mean_energy()

        if action_sequence is None:
            action_sequence = self.action_sequence
        self.system.state = self.initial_state
        sets_of_gates = [self.decode_action(a) for a in action_sequence]

        for gates in sets_of_gates:
            self.system.apply_gates(gates, inplace=True)
        return -self.system.get_mean_energy()

    def step(self, a, calculate_reward=True):
        if self.action_sequence is None:
            raise NotImplementedError('action_sequence not initialized'
                                      'reset() should have been called.')
        if self.state_sequence is None:
            raise NotImplementedError('state_sequence not initialized'
                                      'reset() should have been called.')
        if self.reward_sequence is None:
            raise NotImplementedError('reward_sequence not initialized'
                                      'reset() should have been called.')
        self.state_sequence.append(self.s)
        self.action_sequence.append(a)
        next_state, done = self.get_transition(self.s, a)
        self.s = next_state
        self.lastaction = a
        if calculate_reward:
            reward = self.reward(action=a)
            self.reward_sequence.append(reward)
        else:
            reward = 0.0
        return (next_state, reward, done, {})

    def initial_action_sequence(self):
        return np.random.uniform(-1, 1, size=(self.n_steps, self.action_len))


class Discrete(object):
    """
    {0,1,...,n-1}
    inspired by the Discrete class of the gym package.

    Example:
        env.observation_space = Discrete(2)
        env.action_space.sample()
    """
    def __init__(self, n):
        self.n = n
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        return self.np_random.randint(self.n)
