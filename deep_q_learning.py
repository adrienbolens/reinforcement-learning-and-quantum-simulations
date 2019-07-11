import random
import numpy as np
import sys
#  import systems as sy
import environments as envs
import tensorflow as tf
#  import tensorflow.keras as keras
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense
#  import tensorflow.keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import itertools
from collections import namedtuple
print('Modules loaded')


class DeepQLearning(object):
    """Basic class for Deep Q-Learning.
    Two Neural Networks are used for the Q-function.
    One for the behavior-policy, and one for the target-policy.
    (behavior NN and target NN)
    The target NN is frozen and gets periodically updated with the parameters
    of the behavior NN, while the behavior NN is continuously trained.
    """

    def __init__(self,
                 n_episodes,
                 learning_rate,
                 epsilon_max,
                 epsilon_min,
                 epsilon_decay,
                 n_replays,
                 replay_spacing,
                 lam,
                 model_update_spacing,
                 system_class,
                 architecture,
                 exploration='random',
                 max_Q_optimizer='NAG',
                 GD_eta=0.6,
                 GD_gamma=0.9,
                 n_initial_actions=5,
                 tf_seed=None,
                 seed=None,
                 **other_params
                 ):
        params = locals()
        params.pop('self')
        params.pop('other_params')
        params.update(other_params)
        self.env = envs.ContinuousCurrentGateEnv(**params)
        self.system_class = system_class
        self.n_episodes = n_episodes
        self.lam = lam
        self.max_Q_optimizer = max_Q_optimizer
        self.GD_eta = GD_eta
        self.GD_gamma = GD_gamma
        self.n_initial_actions = n_initial_actions
        self.exploration = exploration
        #  self.list_q_chosen_actions = []
        #  self.list_q_discretized_actions = []
        print("self.n_initial_actions = ", self.n_initial_actions)
        if tf_seed is None:
            tf_seed = seed

        #  - one-hot encoded steps (n_steps neurons)
        #  The followings neurons are doubled: for state and action
        #  - all to all gate (1 neuron)
        #  - single-qubit gates (n_directions * n_sites neurons)
        #  n_action_inputs = 1 + self.env.n_directions *
        #  self.env.system.n_sites
        self.n_inputs = self.env.n_steps + 2 * self.env.action_len
        print(f'\nThe neural networks have {self.n_inputs} input neurons.')

        tf.set_random_seed(tf_seed)
        #  self.architecture = [(100, 'tanh'),
        #                       (20, 'relu'),
        #                       (20, 'relu'),
        #                       (1, 'sigmoid')]
        self.architecture = architecture
        print('Tensorflow verion: ', tf.__version__)
        print('Tensorflow file: ', tf.__file__)
        print('Keras verion: ', keras.__version__)
        print('Keras file: ', keras.__file__)
        self.model = Sequential()
        n, act = self.architecture[0]
        self.model.add(Dense(n, activation=act,
                             input_shape=(self.n_inputs, )))

        for n, act in self.architecture[1:]:
            self.model.add(Dense(n, activation=act))
        #  self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.current_weights = np.array(self.model.get_weights())
        self.target_model = keras.models.clone_model(self.model)
        self.update_target_model()
        self.gradient_weights = K.gradients(self.model.outputs,
                                            self.model.trainable_weights)

        self.gradient_action = K.gradients(
            self.model.outputs, self.model.inputs
        )[0][0, self.env.n_steps + self.env.action_len:]

        self.gradient_action_target = K.gradients(
            self.target_model.outputs, self.target_model.inputs
        )[0][0, self.env.n_steps + self.env.action_len:]

        #  if self.max_Q_optimizer == "Newton":
        #      self.hessian_action = [
        #          K.gradients(self.gradient_action[i],
        #                      self.model.inputs)[0][0, self.env.n_steps +
        #                                            self.env.action_len:]
        #          for i in range(self.env.action_len)
        #      ]

        #      self.hessian_action_target = [
        #       #  K.gradients(self.gradient_action_target[i],
        #       #  self.target_model.inputs)[0][0, self.env.n_steps +
        #       #  self.env.action_len:]
        #          for i in range(self.env.action_len)
        #      ]

        self.sess = K.get_session()

        self.learning_rate = learning_rate
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_replays = n_replays
        self.replay_spacing = replay_spacing
        self.model_update_spacing = model_update_spacing
        # random is only used for mini_batch sampling
        # (somehow, np.random.choice does not like the list of Episode's)
        random.seed(seed)
        np.random.seed(seed)
        self.best_encountered_actions = None
        self.best_encountered_reward = None
        print(f'Instance of {type(self).__name__} initialized with '
              f'the following attributes (showing only str, int and float):')
        for attribute, value in self.__dict__.items():
            if type(value) in (str, int, float):
                print(f'{attribute} = {value}')

    def evaluate_gradient_weights(self, network_input):
        return self.sess.run(self.gradient_weights,
                             feed_dict={self.model.input: network_input})

    def evaluate_gradient_action(self, network_input, use_target=False):
        if use_target:
            model, gradient = self.target_model, self.gradient_action_target
        else:
            model, gradient = self.model, self.gradient_action
        return self.sess.run(gradient, feed_dict={model.input: network_input})

    def evaluate_hessian_action(self, network_input, use_target=False):
        # hessian[i] = gradient of gradient[i] -> Nabla doutput/dai
        # hessian[i, j] = d( doutput/dai ) / daj = d^2ouput/ daj dai
        # it is actually the tranpose of the hessian
        if self.max_Q_optimizer != 'Newton':
            raise ValueError('The max_Q_optimizer is '
                             f'{self.max_Q_optimizer}, you shoud not need'
                             'to calculate the Hessian.')
        if use_target:
            model, hessian = self.target_model, self.hessian_action_target
        else:
            model, hessian = self.model, self.hessian_action
        return np.array(
            self.sess.run(hessian, feed_dict={model.input: network_input}),
            dtype=np.float32
        )

    def run(self):
        #  storage_spacing = 250
        rewards = np.zeros(self.n_episodes)
        self.best_encountered_actions = self.env.initial_action_sequence()
        self.best_encountered_reward = \
            self.env.reward(self.best_encountered_actions)
        print('Fidelity of the initial action sequence (e.g. Trotter) is '
              f'{self.best_encountered_reward}')
        for episode in range(self.n_episodes):

            if episode % self.model_update_spacing == 0:
                self.update_target_model()

            verbose = False
            if episode % (self.n_episodes//10) == 0:
                verbose = True
                print(f'Episode {episode}: ')
            #  if episode % storage_spacing == 0:
            #      mode = 'explore_and_store'
            #  else:
            #      mode = 'explore'
            mode = 'explore'
            reward = self.run_episode(verbose, mode=mode)
            if reward > self.best_encountered_reward:
                self.best_encountered_reward = reward
                self.best_encountered_actions = self.env.action_sequence
            rewards[episode] = reward

            if episode % self.replay_spacing == 0 \
                    and episode < self.n_episodes - 1:
                if episode == 0 and self.system_class != 'LongRangeIsing':
                    pass
                else:
                    self.replay_best_episode()
                    #  for _ in range(self.n_replays):
                    #      self.run_episode(mode='replay')

            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        n_extra = 1000
        extra_rewards = np.zeros(3 * n_extra)
        for n in range(n_extra):
            reward = self.run_episode(mode='greedy', update=True)
            extra_rewards[n] = reward
        for n in range(n_extra, 2*n_extra):
            reward = self.run_episode(mode='explore', update=False)
            extra_rewards[n] = reward
        for n in range(2*n_extra, 3*n_extra):
            reward = self.run_episode(mode='greedy', update=False)
            extra_rewards[n] = reward

        print(f"Final epsilon: {self.epsilon:.2f}.\n")
        self.env.render()
        print(f'\nFidelity of run with final Q: {reward:.4f}')
        print(f'\nBest encountered fidelity: '
              f'{self.best_encountered_reward:.4f}')
        return np.append(rewards, extra_rewards)

    def replay_best_episode(self):
        for _ in range(self.n_replays):
            self.run_episode(mode='replay')

    def run_episode(self, verbose=False, mode='explore', update=True):
        raise NotImplementedError('Vanilla DQL has no implementation without'
                                  'ReplayMemory.')

    def get_best_discretized_action(self, state, use_target=False, n_mesh=4,
                                    n_mesh_all=10):
        if use_target:
            model = self.target_model
        else:
            model = self.model
        state = self.env.process_state_action(state)
        # action is chosen on a mesh (a0, a1, ..., a(env.action_len -1))
        # where -1 < a_i <= 1 takes n_mesh values.
        a_discrete = np.linspace(-1, 1, n_mesh, endpoint=False)
        q_max, a_max, q_min = 0, None, 1
        for a_all in np.linspace(-1, -1, n_mesh_all):
            for action in itertools.product(a_discrete,
                                            repeat=self.env.action_len-1):
                a = np.insert(np.array(action), 0, a_all)
                q = model.predict(self.env.process_action(state, a))[0][0]
                if q > q_max:
                    a_max, q_max = a, q
                if q < q_min:
                    q_min = q
        return (a_max, q_max, q_min)

    def get_best_action(self, state, use_target=False,
                        n_iters=20, convergence_threshold=0.0005):
        n_iters = 3
        #  newton method for the function
        #  a -> model.predict(process_state_action(next_state, a))
        if use_target:
            model = self.target_model
        else:
            model = self.model

        #  For now only contain the "state part" and the "action part" will be
        #  modified inplace
        state = self.env.process_state_action(state)
        a0 = np.linspace(-0.8, 0.8, num=self.n_initial_actions, endpoint=True)
        initial_a = [np.full(self.env.action_len, a) for a in a0]
        q_max = 0
        for i, a in enumerate(initial_a):
            #  print(f"{i}-th initial action")
            update_vec = 0 * a
            for _ in range(n_iters):
                if self.max_Q_optimizer == 'NAG':
                    update_vec = self.NAG_action_update(
                        a, state, use_target, update_vec
                    )
                elif self.max_Q_optimizer == 'Newton':
                    update_vec = self.newton_action_update(
                        a, state, use_target
                    )
                else:
                    raise ValueError('max_Q_optimizer not implemented.')
                #  print(f'a is {a}')
                #  print(f'update_vec is {update_vec}')
                a, a_old = a + update_vec, a
                #  print(f'a_old is {a_old}')
                if np.linalg.norm(a - a_old) < convergence_threshold:
                    print(f"Early convergence after {_} iterations.")
                    break
            #  print(f'final a is {a}')
            a[1:] = (a[1:] + 1) % 2 - 1
            #  mask = (a_one < -1) | (a_one > 1)
            #  n_outside = np.sum(mask)
            #  if n_outside > 0:
            #      print(n_outside, end=', ')
            #      print(1)
            #      print("a outside range at positions:", np.nonzero(mask)[0])
            #      print(f'The converged action contains {n_outside}'
            #          f'/{self.env.action_len} values outside the [-1, 1]'
            #          'range')
            #  a[mask] = np.minimum(np.maximum(a[mask], -1), 1)
            a[0] = min(max(a[0], -1), 1)
            #  print(f'final a after is {a}')
            q = model.predict(self.env.process_action(state, a))[0][0]
            if q > q_max:
                a_max, q_max = a, q

        #  return (a, model.predict(self.env.process_state_action(state, a)))
        #  print(f'a_max, q_max = {a_max, q_max}')
        return (a_max, q_max)

    def newton_action_update(self, action, state, use_target):
        """Newton method
        We want to solve f(x) = 0 (here f = grad output, a vector)
        f(x) = f(x0) + M (x - x0)
        M = grad f(x0) is a matrix
        M[i] = grad f[i] = grad dout / dai
        -> M[i][j] = d (dout/dai) / daj = d^2out/dajdai = hess[j][i]
        M is the transpose of the hessian of output
        evaluate_hessian_action already returns the transpose

        For a given a0, we solve f(x0) + M (x - x0) = 0
        1. M y = - grad output (a_i): solve for y
        2. a_i+1 = a_i + y
        """
        input_ = self.env.process_action(state, action)
        grad = self.evaluate_gradient_action(input_, use_target)
        hess = self.evaluate_hessian_action(input_, use_target)
        # solve hess @ y = - grad
        return np.linalg.solve(hess, -grad)

    def NAG_action_update(self, action, state, use_target, update_vec):
        """Nesterov accelerated gradient "Gradient ascent" (max instead of min)
        step: a <- a + v (usually - sign for descent)
        For usual GD: v = eta * grad_a J(a)
        With Momentum: v_t = gamma * v_{t-1} + eta * grad_a J(a)
        typically gamma = 0.0
        for NAG: v_t = gamma * v_{t-1} + eta * grad_a J(a + gamma * v_{t-1})
        """
        update_vec *= self.GD_gamma
        input_ = self.env.process_action(state, action + update_vec)
        grad = self.evaluate_gradient_action(input_, use_target)
        return update_vec + self.GD_eta * grad

    def update_target_model(self):
        self.target_model.set_weights(self.current_weights)

    def choose_action(self, mode, step=0):
        if mode == 'explore' or mode == 'explore_and_store':
            # With the probability of (1 - epsilon) take the best action in our
            # Q-table
            #  if random.uniform(0, 1) > self.epsilon:
            if self.exploration != 'random':
                raise NotImplementedError(
                    'Only random exploration is implemented. Currently '
                    f'exploration is {self.exploration}'
                )
            if np.random.rand() > self.epsilon:
                action, q = self.get_best_action(self.env.s)
                #  if mode == 'explore_and_store':
                #      self.list_q_chosen_actions.append(q_)
                #      self.list_q_discretized_actions.append(
                #          self.get_best_discretized_action(self.env.s)[1:]
                #      )
                #      print("q_max's added to lists.")
                #  action = np.argmax(self.q_matrix[self.env.s])
                # Else take a random action
            else:
                action = self.env.random_action()
                #  action = self.env.action_space.sample()
        elif mode == 'replay':
            action = self.best_encountered_actions[step]
        elif mode == 'greedy':
            action, q = self.get_best_action(self.env.s)
            #  action = np.argmax(self.q_matrix[self.env.s])
        else:
            raise ValueError(f'The action selecting mode {mode} does not '
                             'exist.')
        return action

    def save_best_encountered_actions(self, filename):
        with open(filename, 'w') as f:
            f.write(f'Corresponding reward = '
                    f'{self.best_encountered_reward:.4f}.\n\n\n')
            self.env.render(f, self.best_encountered_actions)

    def save_post_episode_rewards(self, filename, n_rewards,
                                  action_sequence=None):
        if action_sequence is None:
            action_sequence = self.best_encountered_actions
        r1, r2 = self.env.reward(action_sequence, n_rewards)
        with open(filename, 'wb') as f:
            np.save(f, np.array([r1, r2]))

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.current_weights)

    #  def save_lists_q_max(self, filename1, filename2=None):
    #      if filename2 is None:
    #          filename1, filename2 = \
    #              (filename1[:-4] + '_chosen' + filename1[-4:],
    #               filename1[:-4] + '_discretized' + filename1[-4:])
    #      with open(filename1, 'wb') as f:
    #          np.save(f, np.array(self.list_q_chosen_actions))
    #      with open(filename2, 'wb') as f:
    #          np.save(f,
    #          np.array(list(zip(*self.list_q_discretized_actions))))


class DeepQLearningWithTraces(DeepQLearning):
    """Deep Q-Learning with the addition of eligibility traces.
    Those are basically a compromise between n-step TD
    for all n and Monte Carlo methods (with all steps).
    The backward view is implemented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = 0 * np.array(self.current_weights)

    def run_episode(self, verbose=False, mode='explore', update=True):
        if mode == 'replay':
            learning_rate = 1
            calculate_reward = False
        else:
            learning_rate = self.learning_rate
            calculate_reward = True
        done = False
        total_reward, reward = 0, 0
        self.env.reset()
        self.trace *= 0.0
        step = 0
        while not done:
            #  use policy NN
            action = self.choose_action(mode, step)
            state = self.env.s
            next_state, reward, done, _ = self.env.step(action,
                                                        calculate_reward)
            if done and mode == 'replay':
                reward = self.best_encountered_reward
            total_reward += reward
            if not update:
                step += 1
                continue
            self.trace += self.evaluate_gradient_weights(
                self.env.process_state_action(state, action)
            )
            #  use policy NN
            delta = - self.model.predict(
                self.env.process_state_action(state, action)
            )[0][0]

            delta += reward
            if not done:
                # use target NN
                # get_best_action also returns the corresponding Q-value
                delta += self.get_best_action(next_state, use_target=True)[1]
            self.current_weights += learning_rate * delta * self.trace
            self.model.set_weights(self.current_weights)
            if not done:
                self.trace *= self.lam
                step += 1
        if verbose:
            print(f'\n----------Total Reward: {total_reward:.2f}')
        return total_reward

    def choose_action(self, mode, step=0):
        if mode == 'explore' or mode == 'explore_and_store':
            # With the probability of (1 - epsilon) take the best action
            if self.exploration != "random":
                raise ValueError(
                    'QL with eligibility traces only works with random'
                    'exploration. Currently, exploration is '
                    f'{self.exploration}.'
                )
            if np.random.rand() > self.epsilon:
                action, q = self.get_best_action(self.env.s)
                #  if mode == 'explore_and_store':
                #      self.list_q_chosen_actions.append(q_)
                #      self.list_q_discretized_actions.append(
                #          self.get_best_discretized_action(self.env.s)[1:]
                #      )
                #      print("q_max's added to lists.")
            # Else take a random action
            else:
                action = self.env.random_action()
                # ONLY DIFFERENCE WITH METHOD IN SUPER():
                self.trace *= 0
        else:
            action = super().choose_action(mode, step)
        return action


Episode = namedtuple('Episode', ('action_sequence', 'state_sequence',
                                 'total_reward', 'q_target_sequence'))


class ReplayMemory(object):
    """Replay memory that stores full episodes during the Q-learning"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode, n_pushes=1):
        """Saves a transition."""
        for _ in range(n_pushes):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = episode
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        #  return np.random.choice(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQLWithReplayMemory(DeepQLearning):
    """DQN with the addition of a replay memory.
    This implementation is not compatible with eligibility traces.
    """
    import random

    def __init__(self, capacity, sampling_size, NN_optimizer, n_epochs, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model.compile(optimizer=NN_optimizer, loss='logcosh',
                           metrics=['mse', 'mae'])
        self.n_epochs = n_epochs
        #  metrics=['accuracy']
        self.history = {}
        self.memory = ReplayMemory(capacity)
        self.sampling_size = sampling_size
        self.batch_size = sampling_size * self.env.n_steps
        # not setting weights manually anymore (except for target), just
        # use get_weight method
        self.current_weights = None  # to avoid implementation mistakes

    def run_episode(self, verbose=False, mode='explore', update=True):
        if mode == 'replay':
            calculate_reward = False
        else:
            calculate_reward = True
        done = False
        total_reward, reward = 0, 0
        q_target_sequence = []
        state_sequence = []
        self.env.reset()
        step = 0
        while not done:
            action = self.choose_action(mode, step)
            state_sequence.append(self.env.s)
            next_state, reward, done, _ = self.env.step(action,
                                                        calculate_reward)
            if done and mode == 'replay':
                reward = self.best_encountered_reward
            total_reward += reward
            if not done:
                q_target_sequence.append(
                    self.get_best_action(next_state, use_target=True)[1]
                )
            step += 1
        if verbose:
            print(f'\n----------Total Reward: {total_reward:.2f}')
        #  state_sequence = self.env.get_states_from_action_sequence()
        episode = Episode(self.env.action_sequence, state_sequence,
                          total_reward, np.array(q_target_sequence))
        n_pushes = 1
        if mode == 'replay':
            n_pushes = self.n_replays
        self.memory.push(episode, n_pushes)

        if update:
            self.optimize_model()

        return total_reward

    def replay_best_episode(self):
        self.run_episode(mode='replay')

    def optimize_model(self):
        # update policy model using one mini-batch sampled from memory
        # episodes is a list of namedtuple "Episode"
        if(len(self.memory) < self.sampling_size):
            return None
        episodes = self.memory.sample(self.sampling_size)
        batch = Episode(*zip(*episodes))
        #  # tuple of floats with all rewards of the minibatch
        #  reward = batch.total_reward
        #  # tuple of lists with all action_sequences
        #  action_sequence = batch.action_sequence

        # transforms batch = [((a0, ..., aN-1), rtotal), ((...), rtot),...]
        # into something to feed the netowrk
        labels = np.zeros(self.batch_size)
        labels[self.env.n_steps-1::self.env.n_steps] = batch.total_reward
        # ---- just reward, need to get actual target from Q-learning:
        for i, e in enumerate(episodes):
            labels[i*self.env.n_steps:(i+1)*self.env.n_steps-1] = \
                e.q_target_sequence
        #          ([self.get_best_action(s, use_target=True)[1]
        #            for s in e.state_sequence[1:]])
        # idea: store those Q_target values in Replay Memory, and update them
        # when Q_target is updated (if update_time >~ capacity)
        # if update_time >> capacity, no need to update at all, as memory
        # changes too fast

        #  train = np.zeros(shape=(self.batch_size, self.n_inputs))
        train = np.concatenate([self.env.inputs_from_sequence(*seqs)
                                for seqs in zip(batch.action_sequence,
                                                batch.state_sequence)])

        assert len(train) == self.batch_size, \
            'training data was not properly processed'

        self.model.fit(train, labels, epochs=self.n_epochs, verbose=0)
        for key in self.model.history.history:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(self.model.history.history[key][0])

        # if RNN, go back to t=0 after each episode
        # get input with the right shape
        # single target is deltai = Q_policy(si, ai) - (ri + max_a
        # Q_target(si+1, a))
        # ri is rtot for i = N-1 else 0
        # make array of right shape with all targets
        # train model with keras tools (fit)

    def choose_action(self, mode, step=0):
        if mode == 'replay':
            action = self.best_encountered_actions[step]
        elif mode == 'greedy':
            action, q = self.get_best_action(self.env.s)
        elif mode == 'explore' or mode == 'explore_and_store':
            if self.exploration == 'random':
                if np.random.rand() > self.epsilon:
                    action, q = self.get_best_action(self.env.s)
                else:
                    action = self.env.random_action()
            elif self.exploration == 'gaussian':
                action, q = self.get_best_action(self.env.s)
                # add gaussian fluctuation with std = 0.5*ε
                # (ε = 1 -> 2σ = 1 -> 95% inside [-1, 1])
                action += self.epsilon * 0.5 * np.random.randn(*action.shape)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f'The action selecting mode {mode} does not '
                             'exist.')
        return action

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, np.array(self.model.get_weights()))

    def save_history(self, filename):
        keys = ['loss', 'mean_squared_error', 'mean_absolute_error']
        history_array = np.array(
            [self.history[key] for key in keys]
        )
        print('saving history of NN for metrics: ', keys)
        with open(filename, 'wb') as f:
            np.save(f, history_array)


if __name__ == '__main__':
    import time
    start_time = time.time()
    import json
    from pathlib import Path
    #  from parameters import parameters
    if Path('info.json').is_file():
        with open('info.json') as f:
            info = json.load(f)
        parameters = info['parameters']
    else:
        from parameters import parameters, parameters_deep
        parameters.update(parameters_deep)

    print(f"env.n_steps = {parameters['n_steps']}.")

    # The seed here is for the exploration randomness
    if len(sys.argv) < 2:
        array_index = 1
        create_output_files = False
        print("Output files won't be created.")
    else:
        array_index = int(sys.argv[1])
        create_output_files = True
        print("Output files will be created.")

    print(f"Array n. {array_index}")
    seed_qlearning = array_index
    print(f'The seed used for the q_learning algorithm = {seed_qlearning}.')
    #  start_qlearning = time.time()
    if parameters['subclass'] == 'WithReplayMemory':
        q_learning = DQLWithReplayMemory(
            #  environment=env,
            seed=seed_qlearning,
            **parameters
        )
    else:
        raise NotImplementedError(f"subclass {parameters['subclass']} in "
                                  'parameters.py not recognized.')

    initial_action_sequence = q_learning.env.initial_action_sequence()
    initial_reward = q_learning.env.reward(initial_action_sequence)

    #  start_run = time.time()
    rewards = q_learning.run()
    #  end_run = time.time()

    if create_output_files:
        q_learning.save_best_encountered_actions('best_gate_sequence.txt')
        q_learning.save_weights('final_weights.npy')
        #  q_learning.save_lists_q_max('list_q_max.npy')

        n_rewards = 100
        #  start_post = time.time()
        q_learning.save_post_episode_rewards(
            'post_episode_rewards__best.npy',
            n_rewards,
            q_learning.best_encountered_actions
        )

        q_learning.save_post_episode_rewards(
            'post_episode_rewards__final.npy',
            n_rewards,
            q_learning.env.action_sequence
        )

        if parameters['system_class'] == 'LongRangeIsing':
            q_learning.save_post_episode_rewards(
                'post_episode_rewards__trotter.npy',
                n_rewards,
                initial_action_sequence
            )
        if parameters['subclass'] == 'WithReplayMemory':
            q_learning.save_history('NN_history.npy')

        #  end_post = time.time()
        end_time = time.time()

        #  times = [
        #      start, start_qlearning, start_run, end_run, start_post, end_post
        #  ]
        #  times = [t - start for t in times]
        #  diff_times = [times[i+1] - times[i] for i in range(len(times)-1)]
        with open('rewards.npy', 'wb') as f:
            np.save(f, rewards)
        info_dic = {
            #  'parameters': parameters,
            'initial_reward': initial_reward,
            'final_reward': rewards[-1],
            'best_reward': q_learning.best_encountered_reward,
            'total_time': end_time - start_time
            #  'times': times,
            #  'diff_times': diff_times
            }
        #  print("Compare 'best_encountered_reward' = "
        #        f"{q_learning.best_encountered_reward:.5f} with 'max(rewards)
        #        = "
        #        f"{np.max(rewards):.5f}.")
        with open('results_info.json', 'w') as f:
            json.dump(info_dic, f, indent=2)
        print("results_info.json written.")
