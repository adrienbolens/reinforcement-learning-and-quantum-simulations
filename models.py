from collections import namedtuple
import random
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, LSTM, Activation
import tensorflow.keras.backend as K

#  import keras
#  from keras.models import Sequential
#  from keras.layers import Dense, InputLayer, LSTM
#  import keras.backend as K

from math import pi

print('Tensorflow verion: ', tf.__version__)
print('Tensorflow file: ', tf.__file__)
print('Keras verion: ', keras.__version__)
print('Keras file: ', keras.__file__)


class DeepQNetwork():
    """Deep Neural Network built for the deep Q-learning algorithm"""

    def __init__(self,
                 architecture,
                 env,
                 max_q_optimizer=None,
                 tf_seed=None,
                 **other_params):

        self.env = env
        self.n_inputs = env.get_n_inputs()
        self.action_len = env.get_action_len()
        print(f'\nThe neural networks have {self.n_inputs} input neurons.')

        if max_q_optimizer is None:
            self.max_q_optimizer = {
                'algorithm': 'NAG',
                'momentum': 0.9,
                'learning_rate': 0.6,
                'n_initial_actions': 5,
                'clip_action': True
            }
        else:
            self.max_q_optimizer = max_q_optimizer

        tf.set_random_seed(tf_seed)
        self.architecture = architecture

        self.build_model()
        #  self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.current_weights)
        self.target_model.summary()
        self.update_target_model()

        self.gradient_weights = K.gradients(self.model.outputs,
                                            self.model.trainable_weights)

        self.gradient_action = {
            'main': self.get_gradient_action(self.model),
            'target': self.get_gradient_action(self.target_model)
        }

        if max_q_optimizer.get('algorithm') == 'Newton':
            self.hessian_action = {
                'main': self.get_hessian_action(self.model),
                'target': self.get_hessian_action(self.target_model)
            }

        self.sess = K.get_session()

    @property
    def current_weights(self):
        return np.array(self.model.get_weights())

    @current_weights.setter
    def current_weights(self, weights):
        self.model.set_weights(weights)

    def update_target_model(self):
        self.target_model.set_weights(self.current_weights)

    def action_update(self, *args, **kwargs):
        if self.max_q_optimizer['algorithm'] == 'NAG':
            return self.nag_action_update(*args, **kwargs)
        if self.max_q_optimizer['algorithm'] == 'Newton':
            return self.newton_action_update(*args, **kwargs)

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        #  Store history of all the `fit` on the model in:
        self.history = {}


class DenseQNetwork(DeepQNetwork):
    """Using fully connected Dense layers."""

    def build_model(self):
        self.model = Sequential()
        #  The use of `InputLayer` results in a different behavior of
        #  target_model obtained from `clone_model` (e.g.
        #  target_model.summary() raises an error because the weights of
        #  `target_model` aren't created, even after using its `set_weights`
        #  method...) The error disapear when the input_shape is set in the
        #  first Dense layer.

        self.model.add(InputLayer(input_shape=(self.n_inputs,)))
        for n, act in self.architecture:
            self.model.add(Dense(n, activation=act))

        #  n, act = self.architecture[0]
        #  self.model.add(Dense(n, activation=act,
        #                       input_shape=(self.n_inputs,)))
        #  for n, act in self.architecture[1:]:
        #      self.model.add(Dense(n, activation=act))

    def get_gradient_action(self, model):
        return K.gradients(
            model.outputs, model.inputs
        )[0][0, -self.action_len:]
        #  )[0][0, self.env.n_steps + self.env.action_len:]

    def get_hessian_action(self, model):
        gradient_action = self.get_gradient_action(model)
        return [
            K.gradients(
                gradient_action[i], model.inputs
            )[0][0, -self.action_len:]
            for i in range(self.action_len)
        ]

    def evaluate_gradient_weights(self, network_input):
        return self.sess.run(self.gradient_weights,
                             feed_dict={self.model.input: network_input})

    def evaluate_gradient_action(self, network_input, use_target=False):
        if use_target:
            model = self.target_model
            gradient = self.gradient_action['target']
        else:
            model = self.model
            gradient = self.gradient_action['main']
        return self.sess.run(gradient, feed_dict={model.input: network_input})

    def evaluate_hessian_action(self, network_input, use_target=False):
        # hessian[i] = gradient of gradient[i] -> Nabla doutput/dai
        # hessian[i, j] = d( doutput/dai ) / daj = d^2ouput/ daj dai
        # it is actually the tranpose of the hessian
        if self.max_q_optimizer.get('algorithm') != 'Newton':
            raise ValueError('The max_Q_optimizer is '
                             f'{self.max_Q_optimizer}, you shoud not need'
                             'to calculate the Hessian.')
        if use_target:
            model, hessian = self.target_model, self.hessian_action['target']
        else:
            model, hessian = self.model, self.hessian_action['main']
        return np.array(
            self.sess.run(hessian, feed_dict={model.input: network_input}),
            dtype=np.float32
        )

    def get_best_discretized_action(self, state, use_target=False, n_mesh=4,
                                    n_mesh_all=10):
        """
        Get best Q-value with bruteforce: discretize the action space and
        calculate Q for each value.
        This is used to compare the results of max_q_optimizer for small
        action space.

        Arguments
        ---------
        state: raw form defined in self.env

        Returns
        -------
            tuple:
                a_max: best action
                q_max: corresponding best q value
                q_min: minimum q value encountered
        """

        if use_target:
            model = self.target_model
        else:
            model = self.model

        state = self.env.process_state_action(state, action=None)

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

    def get_best_action(self, state, step=None, use_target=False,
                        n_iters=None, convergence_threshold=None):
                        #  n_iters=20, convergence_threshold=0.0005):
        """
        Returns the best action from the deep network using best_q_optimizer

        Arguments
        ---------
            state: raw form of state defined in self.env
            use_target (bool)

        Returns
        -------
            tuple:
                np.array: a_max, the best action
                float: q_max, the corresponding best q value

        """

        if use_target:
            model = self.target_model
        else:
            model = self.model

        if n_iters is None:
            n_iters = self.max_q_optimizer['n_iterations']
        if convergence_threshold is None:
            convergence_threshold = \
                self.max_q_optimizer['convergence_threshold']

        state = self.env.process_state_action(state, action=None)
        a0 = np.linspace(-1, 1, num=self.max_q_optimizer['n_initial_actions'],
                         endpoint=True)
        initial_a = [np.full(self.action_len, a) for a in a0]
        q_max = -np.infty
        a_max = initial_a[0]
        for i, a in enumerate(initial_a):
            update_vec = 0 * a
            for _ in range(self.max_q_optimizer['n_iterations']):
                update_vec = self.action_update(a, state, use_target,
                                                update_vec)
                a, a_old = a + update_vec, a
                if np.linalg.norm(a - a_old) < convergence_threshold:
                    #  print(f"Early convergence after {_} iterations.")
                    break

            # if range_one is math.pi, use periodic parameters
            if self.env.range_one == pi:
                a[1:] = (a[1:] + 1) % 2 - 1
            # otherwise clip into [-1, 1]
            else:
                if self.max_q_optimizer['clip_action']:
                    #  restrict a_one range to be in [-1, 1]):
                    a[1:] = np.minimum(np.maximum(a[1:], -1), 1)

                # #  count the number of clipped values:
                # #  -----------------------------------
                # n_outside = np.sum(mask)
                # if n_outside > 0:
                #     print(n_outside, end=', ')
                #     print(1)
                #     print("a outside range at positions:",
                #            np.nonzero(mask)[0])
                #     print(f'The converged action contains {n_outside}'
                #         f'/{self.env.action_len} values outside the [-1, 1]'
                #         'range')
                # a[mask] = np.minimum(np.maximum(a[mask], -1), 1)

            if self.max_q_optimizer['clip_action']:
                #  restrict a_all range to be in [-1, 1]):
                a[0] = min(max(a[0], -1), 1)

            q = model.predict(self.env.process_action(state, a))[0][0]
            if q > q_max:
                a_max, q_max = a, q

        return (a_max, q_max)

    def newton_action_update(self, action, state, use_target):
        """
        Newton method
        We want to solve f(x) = 0 (here f = ∇output, a vector)
        f(x) = f(x0) + M (x - x0)
        M = ∇f(x0) is a matrix
        M[i] = ∇f[i] = ∇ (d_out / d_ai)
        -> M[i][j] = d (d_out/d_ai) / d_aj = d^2_out/d_aj d_ai = hess[j][i]
        M is the transpose of the hessian of output
        evaluate_hessian_action already returns the transpose

        For a given a0, we solve f(x0) + M (x - x0) = 0
        1. M y = - ∇ output (a_i): solve for y
        2. a_i+1 = a_i + y
        """

        input_ = self.env.process_action(state, action)
        grad = self.evaluate_gradient_action(input_, use_target)
        hess = self.evaluate_hessian_action(input_, use_target)
        return np.linalg.solve(hess, -grad)

    def nag_action_update(self, action, state, use_target, update_vec):
        """
        Nesterov Accelerated Gradient
        Here, we use "Gradient ASCENT" (max instead of min)
        step: a <- a + v (usually, there is an extra - sign for descent)

        For usual GD:
                v = η * ∇J(a)

        With Momentum:
                v_t = γ * v_{t-1} + η * ∇J(a)
                (typically γ = 0.9)

        for NAG:
                v_t = γ * v_{t-1} + η * ∇J(a + γ * v_{t-1})

        γ: momentum, η: learning_rate, J: cost function, v: update vector

        """
        update_vec *= self.max_q_optimizer['momentum']
        input_ = self.env.process_action(state, action + update_vec)
        grad = self.evaluate_gradient_action(input_, use_target)
        return update_vec + self.max_q_optimizer['learning_rate'] * grad

    def update(self, memory, sampling_size, epochs):
        """
        Update policy network using a batch of episodes sampled from memory
        (`episodes` is a list of `Episode` namedtuples.)
        the metrics in model.history.history are stored in `self.history`
        """
        if(len(memory) < sampling_size):
            return None
        episodes = memory.sample(sampling_size)
        batch = Episode(*zip(*episodes))

        # transforms batch = [((a0, ..., aN-1), rtotal), ((...), rtot),...]
        # into something to feed the netowrk

        # One episode contains `n_steps` network inputs.
        batch_size = sampling_size * self.env.n_steps

        labels = np.zeros((sampling_size, self.env.n_steps))
        labels[:, :] = batch.reward_sequence
        # we also need the q values from the target network:
        labels[:, :-1] += batch.q_target_sequence
        labels = labels.reshape(batch_size)

        #  labels = np.zeros(batch_size)
        #  labels[self.env.n_steps-1::self.env.n_steps] = batch.total_reward
        #  # we also need the q values from the target network:
        #  for i, episode in enumerate(episodes):
        #      labels[i*self.env.n_steps:(i+1)*self.env.n_steps-1] = \
        #          episode.q_target_sequence
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

        assert len(train) == batch_size, ('training data was not properly '
                                          'processed')

        self.model.fit(train, labels, epochs=epochs, verbose=0)
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


class LSTMQNetwork(DeepQNetwork):
    """
    Using a LSTM network. Important difference: the input of the network is
    made of all the `(state, action)` in an episode. Whereas it was only
    the a single `(state, action)` for the Dense network.
    """

    def reset(self):
        self.current_network_input = np.zeros((1, self.n_steps, self.n_inputs))

    def update_network_input(self, state, action, step):
        assert step == len(self.env.action_sequence)
        self.current_network_input[0, step, :] = (
            self.env.process_state_action(state, action, reshape=False)
        )

        #  for i_step, (s, a) in enumerate(zip(self.env.state_sequence,
        #                                      self.env.action_sequence)):
        #      network_input[0, i_step, :] = (
        #          self.env.process_state_action(s, a, reshape=False)
        #      )

    def build_model(self):
        # Number of LSTM 'time' steps.
        self.n_steps = self.env.n_steps

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.n_steps, self.n_inputs)))

        for n in self.architecture['LSTM']:
            self.model.add(LSTM(units=n, return_sequences=True, unroll=True))

        activation = self.architecture.get('activation')
        if activation is not None:
            self.model.add(Activation(activation))

    def get_best_action(self, state, step, use_target=False,
                        n_iters=None, convergence_threshold=None):
                        #  n_iters=20, convergence_threshold=0.0005):
        """
        Returns the best action from the LSTM network using best_q_optimizer

        Arguments
        ---------
            state: raw form of state as defined in self.env.

            use_target (bool): decide whether to use the behavior or target
                network

        Returns
        -------
            tuple:
                np.array: a_max, the best action
                float: q_max, the corresponding best q value

        """

        assert step == len(self.env.action_sequence)
        assert step == len(self.env.state_sequence)

        if n_iters is None:
            n_iters = self.max_q_optimizer['n_iterations']
        if convergence_threshold is None:
            convergence_threshold = \
                self.max_q_optimizer['convergence_threshold']

        if use_target:
            model = self.target_model
        else:
            model = self.model

        state = self.env.process_state_action(state, action=None)

        a0 = np.linspace(-1, 1, num=self.max_q_optimizer['n_initial_actions'],
                         endpoint=True)
        initial_a = [np.full(self.action_len, a) for a in a0]
        q_max = -np.infty
        a_max = initial_a[0]
        for i, a in enumerate(initial_a):
            update_vec = 0 * a
            for _ in range(self.max_q_optimizer['n_iterations']):
                update_vec = self.action_update(a, state, use_target,
                                                update_vec, step)
                a, a_old = a + update_vec, a
                if np.linalg.norm(a - a_old) < convergence_threshold:
                    #  print(f"Early convergence after {_} iterations.")
                    break

            # if range_one is math.pi, use periodic parameters
            if self.env.range_one == pi:
                a[1:] = (a[1:] + 1) % 2 - 1
            # otherwise clip into [-1, 1]
            else:
                if self.max_q_optimizer['clip_action']:
                    #  restrict a_one range to be in [-1, 1]):
                    a[1:] = np.minimum(np.maximum(a[1:], -1), 1)

            #  restrict a_all range to be in [-1, 1]):
            if self.max_q_optimizer['clip_action']:
                a[0] = min(max(a[0], -1), 1)

            extra_input = self.env.process_action(state, a)
            network_input = self.get_current_network_input(extra_input, step)
            #  shape of output is (batch_size=1, n_steps, n_outputs=1)
            q = model.predict(network_input)[0, step, 0]
            if q > q_max:
                a_max, q_max = a, q

        return (a_max, q_max)

    def nag_action_update(self, action, state, use_target, update_vec, step):
        """
        Nesterov Accelerated Gradient
        Here, we use "Gradient ASCENT" (max instead of min)
        step: a <- a + v (usually, there is an extra - sign for descent)

        For usual GD:
                v = η * ∇J(a)

        With Momentum:
                v_t = γ * v_{t-1} + η * ∇J(a)
                (typically γ = 0.9)

        for NAG:
                v_t = γ * v_{t-1} + η * ∇J(a + γ * v_{t-1})

        γ: momentum, η: learning_rate, J: cost function, v: update vector

        """
        update_vec *= self.max_q_optimizer['momentum']
        extra_input = self.env.process_action(state, action + update_vec)
        network_input = self.get_current_network_input(extra_input, step)
        grad = self.evaluate_gradient_action(network_input,
                                             step=step,
                                             use_target=use_target)
        return update_vec + self.max_q_optimizer['learning_rate'] * grad

    def get_gradient_action(self, model):

        output_slices = [
            tf.slice(model.output, [0, step, 0], [1, 1, 1])
            for step in range(self.n_steps)
        ]
        grad_slices = [
            #  slice(x, [i, ...], [j, ...]) correspond to x[i:(i+j), ...]
            #  i.e. a slice of length j starting at x[i] and ending at x[i+j-1]
            tf.slice(
                K.gradients(s, model.input)[0],
                [0, i, self.n_inputs - self.action_len],
                [1, 1, self.action_len]
            )[0, 0] for i, s in enumerate(output_slices)
        ]

        return grad_slices
        #  return K.gradients(
        #      model.outputs, model.inputs
        #  )[0][0, :, -self.action_len:]
        #  #  )[0][0, -self.action_len:]

    def evaluate_gradient_action(self, network_input, step, use_target=False):
        if use_target:
            model = self.target_model
            gradient = self.gradient_action['target'][step]
        else:
            model = self.model
            gradient = self.gradient_action['main'][step]
        r = self.sess.run(gradient, feed_dict={model.input: network_input})
        return r

    def get_current_network_input(self, extra_input=None, step=None):
        """
        Create an LSTM_network_input using an extra single-step input appended
        to the sequence of old inputs (created from the sequences of previous
        states and actions stored in `self.env`).
        Trivial states (i.e. zeros) are padded after the extra_input.
        """
        if extra_input is not None:
            assert step == len(self.env.action_sequence)
            assert step == len(self.env.state_sequence)

        network_input = self.current_network_input

        # input should have the shape (batch_size=1, n_steps, n_inputs)
        #  network_input = np.zeros((1, self.n_steps, self.n_inputs))

        #  for i_step, (s, a) in enumerate(zip(self.env.state_sequence,
        #                                      self.env.action_sequence)):
        #      network_input[0, i_step, :] = (
        #          self.env.process_state_action(s, a, reshape=False)
        #      )

        if extra_input is not None:
            # extra_input.shape is (1, n_inputs), i.e. already processed
            network_input[0, step, :] = extra_input[0]
        return network_input

    def update(self, memory, sampling_size, epochs):
        """
        Update policy network using a batch of episodes sampled from memory
        (`episodes` is a list of `Episode` namedtuples.)
        the metrics in model.history.history are stored in `self.history`
        """
        if(len(memory) < sampling_size):
            return None
        episodes = memory.sample(sampling_size)
        batch = Episode(*zip(*episodes))

        # One episode contains one network input
        # The network output is the Q-value for each steps in the epispode.
        batch_size = sampling_size
        labels = np.zeros((batch_size, self.env.n_steps, 1))
        labels[:, :, 0] = batch.reward_sequence
        # we also need the q values from the target network:
        labels[:, :-1, 0] += batch.q_target_sequence

        # idea: store those Q_target values in Replay Memory, and update them
        # when Q_target is updated (if update_time >~ capacity)
        # if update_time >> capacity, no need to update at all, as memory
        # changes too fast

        # train = np.zeros(shape=(batch_size, self.env.n_steps, self.n_inputs))

        train = np.array([self.env.inputs_from_sequence(*seqs) for seqs in
                          zip(batch.action_sequence, batch.state_sequence)])

        assert train.shape == (batch_size, self.env.n_steps, self.n_inputs)

        assert len(train) == batch_size, ('training data was not properly '
                                          'processed')

        self.model.fit(train, labels, epochs=epochs, verbose=0)
        for key in self.model.history.history:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(self.model.history.history[key][0])


# An episode consists of a sequence of actions, states, and rewards
#
# To speed up the algorithm, it also contains the list of targets for the
# target NN. (there is no target for the final state, therefore its length is
# n_steps-1). Ideally, those should be recalculated incase the target NN has
# been updated when the episode is used, but this would result in a significant
# increase of computation time. (the argmax Q calculation is time-consuming)
Episode = namedtuple('Episode', ('action_sequence', 'state_sequence',
                                 'reward_sequence', 'q_target_sequence'))


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
