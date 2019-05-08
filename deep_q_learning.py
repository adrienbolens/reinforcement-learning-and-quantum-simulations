import random
import numpy as np
#  import sys
#  import systems as sy
#  import environments as envs
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K


class DeepQLearning(object):

    def __init__(self,
                 environment,
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
                 tf_seed=None,
                 seed=None,
                 **other_params
                 ):
        self.env = environment
        self.system_class = system_class
        self.n_episodes = n_episodes
        self.lam = lam

        #  - one-hot encoded steps (n_steps neurons)
        #  The followings neurons are doubled: for state and action
        #  - all to all gate (1 neuron)
        #  - single-qubit gates (n_directions * n_sites neurons)
        self.n_inputs = self.env.n_steps \
            + 2 * (1 + self.env.n_directions * self.env.system.n_sites)
        print(f'\nThe neural networks have {self.n_inputs} input neurons.')

        tf.set_random_seed(tf_seed)
        self.architecture = [(100, 'tanh'),
                             (20, 'relu'),
                             (20, 'relu'),
                             (1, 'sigmoid')]
        print('Tensorflow verion: ', tf.__version__)
        print('Keras verion: ', keras.__version__)
        self.model = Sequential()
        n, act = self.architecture[0]
        self.model.add(Dense(n, activation=act,
                             input_shape=(self.n_inputs, )))

        for n, act in self.architecture[1:]:
            self.model.add(Dense(n, activation=act))
        #  self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.current_weights = self.model.get_weights()
        self.target_model = keras.models.clone_model(self.model)
        self.update_target_model()
        self.gradient_weights = K.gradients(self.model.outputs,
                                            self.model.trainable_weights)

        self.gradient_action = K.gradients(self.model.outputs,
                                           self.model.inputs)
        self.sess = K.get_session()

        self.trace = 0 * np.array(self.current_weights)

        self.learning_rate = learning_rate
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_replays = n_replays
        self.replay_spacing = replay_spacing
        self.model_update_spacing = model_update_spacing
        random.seed(seed)
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

    def evalutate_gradient_action(self, network_input):
        return self.sess.run(self.gradient_action,
                             feed_dict={self.model.input: network_input})

    def run(self):
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
                self.current_episode = episode
                print(f'Episode {episode}: ')
            reward = self.run_episode(verbose, mode='explore')
            if reward > self.best_encountered_reward:
                self.best_encountered_reward = reward
                self.best_encountered_actions = self.env.action_sequence
            rewards[episode] = reward
            if episode % self.replay_spacing == 0 \
                    and episode < self.n_episodes - 1:
                if episode == 0 and self.system_class != 'LongRangeIsing':
                    pass
                else:
                    for _ in range(self.n_replays):
                        self.run_episode(mode='replay')
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print(f"Final epsilon: {self.epsilon:.2f}.\n")
        #  final_reward = self.run_episode(verbose, mode='greedy',
        #  update=False)
        self.env.render()
        print(f'\nFidelity of run with final Q: {reward:.4f}')
        print(f'\nBest encountered fidelity: '
              f'{self.best_encountered_reward:.4f}')
        return rewards

    def run_episode(self, verbose=False, mode='explore', update=True):
        if mode == 'replay':
            learning_rate = 1
        else:
            learning_rate = self.learning_rate
        done = False
        total_reward, reward = 0, 0
        self.env.reset()
        self.trace *= 0.0
        step = 0
        while not done:
            action = self.choose_action(mode, step)
            state = self.env.s
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if not update:
                step += 1
                continue
            # gradient instead of 1
            self.trace += self.evaluate_gradient(
                self.env.process_state_action(state, action)
            )
            #  self.trace[state, action] += 1
            # weights instead of q_matrix
            #  USE BEHAVIOUR NN
            delta = - self.model.predict(
                self.env.process_state_action(state, action)
            )
            #  delta = - self.q_matrix[state, action]

            delta += reward
            if not done:
                #  newton method instead of max WITH TARGET NN
                #  USE TARGET NN
                #  delta += predict of next_state with best_action
                delta += self.get_best_action(next_state, self.target_model)[1]
                #  delta += np.max(self.q_matrix[next_state])

            #  modify weight of NN
            #  USE BEHAVIOUR NN
            self.current_weights += learning_rate * delta * self.trace
            self.model.set_weights(self.current_weights)
            #  self.q_matrix += learning_rate * delta * self.trace
            if not done:
                self.trace *= self.lam
                step += 1
        if verbose:
            print(f'----------Total Reward: {total_reward:.2f}, current '
                  f'epsilon: {self.current_episode:.2f}')
        return total_reward

    def get_best_action(self, state, model):
        #  newton method for the function
        #  a -> model.predict(process_state_action(next_state, a))

        a = [0]*self.env.action_len
        return (a, model.predict(self.env.process_state_action(state, a)))

    def update_target_model(self):
        self.target_model.set_weights(self.current_weights)

    def choose_action(self, mode, step=0):
        if mode == 'explore':
            # With the probability of (1 - epsilon) take the best action in our
            # Q-table
            if random.uniform(0, 1) > self.epsilon:
                action = self.get_best_action(self.env.s, self.model)[0]
                #  action = np.argmax(self.q_matrix[self.env.s])
                # Else take a random action
            else:
                action = self.env.random_action()
                #  ADD FUNCTION TO ENV
                #  action = self.env.action_space.sample()
                self.trace *= 0
                #  self.trace.fill(0.0)
        elif mode == 'replay':
            action = self.best_encountered_actions[step]
        elif mode == 'greedy':
            action = self.get_best_action(self.env.s, self.model)[0]
            #  action = np.argmax(self.q_matrix[self.env.s])
        return action
