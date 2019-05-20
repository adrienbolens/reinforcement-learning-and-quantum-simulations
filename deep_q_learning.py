#  import random
import numpy as np
import sys
#  import systems as sy
import environments as envs
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
print('Modules loaded')


class DeepQLearning(object):

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
                 optimization_method='NAG',
                 GD_eta=0.6,
                 GD_gamma=0.9,
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
        self.optimization_method = optimization_method
        self.GD_eta = GD_eta
        self.GD_gamma = GD_gamma
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

        if self.optimization_method == "Newton":
            self.hessian_action = [
                K.gradients(self.gradient_action[i],
                            self.model.inputs)[0][0, self.env.n_steps +
                                                  self.env.action_len:]
                for i in range(self.env.action_len)
            ]

            self.hessian_action_target = [
                K.gradients(self.gradient_action_target[i],
                            self.target_model.inputs)[0][0, self.env.n_steps +
                                                         self.env.action_len:]
                for i in range(self.env.action_len)
            ]

        self.sess = K.get_session()

        self.trace = 0 * np.array(self.current_weights)

        self.learning_rate = learning_rate
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_replays = n_replays
        self.replay_spacing = replay_spacing
        self.model_update_spacing = model_update_spacing
        #  random.seed(seed)
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
        if self.optimization_method != 'Newton':
            raise ValueError('The optimization_method is '
                             f'{self.optimization_method}, you shoud not need'
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
            self.trace += self.evaluate_gradient_weights(
                self.env.process_state_action(state, action)
            )
            #  self.trace[state, action] += 1
            # weights instead of q_matrix
            #  USE BEHAVIOUR NN
            delta = - self.model.predict(
                self.env.process_state_action(state, action)
            )[0][0]
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
            print(f'\n----------Total Reward: {total_reward:.2f}, current '
                  f'epsilon: {self.current_episode:.2f}')
        return total_reward

    def get_best_action(self, state, use_target=False, n_initial_actions=3,
                        n_iters=15, convergence_threshold=0.001):
        #  newton method for the function
        #  a -> model.predict(process_state_action(next_state, a))
        if use_target:
            model = self.target_model
        else:
            model = self.model

        state = self.env.process_state_action(state)
        #  print(f'the state is {state}.')
        a0 = np.linspace(-0.1, 0.1, num=n_initial_actions, endpoint=True)
        initial_a = [np.full(self.env.action_len, a) for a in a0]
        q_max = 0
        for i, a in enumerate(initial_a):
            #  print(f"{i}-th initial action")
            update_vec = 0 * a
            for _ in range(n_iters):
                if self.optimization_method == 'NAG':
                    update_vec = self.NAG_action_update(
                        a, state, use_target, update_vec
                    )
                elif self.optimization_method == 'Newton':
                    update_vec = self.newton_action_update(
                        a, state, use_target
                    )
                else:
                    raise ValueError('optimization_method not implemented.')
                a, a_old = a + update_vec, a
                if np.linalg.norm(a - a_old) < convergence_threshold:
                    print(f"Early convergence after {_} iterations.")
                    break
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
            q = model.predict(self.env.process_action(state, a))[0][0]
            if q > q_max:
                a_max, q_max = a, q

        #  return (a, model.predict(self.env.process_state_action(state, a)))
        return (a_max, q_max)

    def newton_action_update(self, action, state, use_target):
        # Newton method
        # We want to solve f(x) = 0 (here f = grad output, a vector)
        # f(x) = f(x0) + M (x - x0)
        # M = grad f(x0) is a matrix
        # M[i] = grad f[i] = grad dout / dai
        # -> M[i][j] = d (dout/dai) / daj = d^2out/dajdai = hess[j][i]
        # M is the transpose of the hessian of output
        # evaluate_hessian_action already returns the transpose

        # For a given a0, we solve f(x0) + M (x - x0) = 0
        # 1. M y = - grad output (a_i): solve for y
        # 2. a_i+1 = a_i + y
        input_ = self.env.process_action(state, action)
        grad = self.evaluate_gradient_action(input_, use_target)
        hess = self.evaluate_hessian_action(input_, use_target)
        # solve hess @ y = - grad
        return np.linalg.solve(hess, -grad)

    def NAG_action_update(self, action, state, use_target, update_vec):
        # Nesterov accelerated gradient "Gradient ascent" (max instead of min)
        # step: a <- a + v (usually - sign for descent)
        # For usual GD: v = eta * grad_a J(a)
        # With Momentum: v_t = gamma * v_{t-1} + eta * grad_a J(a)
        # typically gamma = 0.0
        # for NAG: v_t = gamma * v_{t-1} + eta * grad_a J(a + gamma * v_{t-1})
        update_vec *= self.GD_gamma
        input_ = self.env.process_action(state, action + update_vec)
        grad = self.evaluate_gradient_action(input_, use_target)
        return update_vec + self.GD_eta * grad

    def update_target_model(self):
        self.target_model.set_weights(self.current_weights)

    def choose_action(self, mode, step=0):
        if mode == 'explore':
            # With the probability of (1 - epsilon) take the best action in our
            # Q-table
            #  if random.uniform(0, 1) > self.epsilon:
            if np.random.rand() > self.epsilon:
                action = self.get_best_action(self.env.s)[0]
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
        from parameters import parameters

    print(f"env.n_steps = {parameters['n_steps']}.")

    # The seed here is for the exploration randomness
    if len(sys.argv) < 2:
        array_index = 1
        create_output_files = False
    else:
        array_index = int(sys.argv[1])
        create_output_files = True

    print(f"Array n. {array_index}")
    seed_qlearning = array_index
    print(f'The seed used for the q_learning algorithm = {seed_qlearning}.')
    #  start_qlearning = time.time()
    q_learning = DeepQLearning(
        #  environment=env,
        seed=seed_qlearning,
        **parameters
    )

    initial_action_sequence = q_learning.env.initial_action_sequence()
    initial_reward = q_learning.env.reward(initial_action_sequence)

    #  start_run = time.time()
    rewards = q_learning.run()
    #  end_run = time.time()

    if create_output_files:
        q_learning.save_best_encountered_actions('best_gate_sequence.txt')
        q_learning.save_weights('final_weights.npy')

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
