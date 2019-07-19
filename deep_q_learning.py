import random
import numpy as np
import environments as envs
import models as models
from models import Episode


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
                 epsilon_max,
                 epsilon_min,
                 epsilon_decay,
                 n_replays,
                 replay_spacing,
                 model_update_spacing,
                 system_class,
                 exploration='uniform',
                 n_extra_episodes=0,
                 verify_argmax_q=False,
                 seed=None,
                 **other_params):

        self.env = envs.ContinuousCurrentGateEnv(system_class=system_class,
                                                 **other_params)
        self.system_class = system_class
        self.n_episodes = n_episodes
        self.exploration = exploration
        if verify_argmax_q:
            self.list_q_chosen_actions = []
            self.list_q_discretized_actions = []
            self.storage_spacing = 250
        self.verify_argmax_q = verify_argmax_q

        #  - one-hot encoded steps (n_steps neurons)
        #  The followings neurons are doubled: for state and action
        #  - all to all gate (1 neuron)
        #  - single-qubit gates (n_directions * n_sites neurons)
        #  n_action_inputs = 1 + self.env.n_directions *
        #  self.env.system.n_sites

        self.model = models.DeepQNetwork(env=self.env,
                                         tf_seed=seed,
                                         **other_params)

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_replays = n_replays
        self.replay_spacing = replay_spacing
        self.model_update_spacing = model_update_spacing
        # random is only used for mini_batch sampling
        # (np.random.choice does not like the list of Episode namedtuples)
        random.seed(seed)
        np.random.seed(seed)
        self.best_encountered_actions = None
        self.best_encountered_reward = None
        self.n_extra_episodes = n_extra_episodes
        print(f'Instance of {type(self).__name__} initialized with '
              f'the following attributes (showing only str, int and float):')
        for attribute, value in self.__dict__.items():
            if type(value) in (str, int, float):
                print(f'{attribute} = {value}')

    def run(self):
        rewards = np.zeros(self.n_episodes)
        self.best_encountered_actions = self.env.initial_action_sequence()
        self.best_encountered_reward = \
            self.env.reward(self.best_encountered_actions)
        print('Fidelity of the initial action sequence (e.g. Trotter) is '
              f'{self.best_encountered_reward}')
        for episode in range(self.n_episodes):

            if episode % self.model_update_spacing == 0:
                self.model.update_target_model()

            verbose = False
            if episode % (self.n_episodes//10) == 0:
                verbose = True
                print(f'Episode {episode}: ')

            mode = 'explore'
            if self.verify_argmax_q and episode % self.storage_spacing == 0:
                mode = 'explore_and_store'

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

            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print(f"Final epsilon: {self.epsilon:.2f}.\n")
        self.env.render()
        print(f'\nFidelity of run with final Q: {reward:.4f}')
        print(f'\nBest encountered fidelity: '
              f'{self.best_encountered_reward:.4f}')

        if self.n_extra_episodes == 0:
            return rewards

        extra_rewards = np.zeros(self.n_extra_episodes)
        n_extra = self.n_extra_episodes//3
        for n in range(n_extra):
            reward = self.run_episode(mode='greedy', update=True)
            extra_rewards[n] = reward
        for n in range(n_extra, 2*n_extra):
            reward = self.run_episode(mode='explore', update=False)
            extra_rewards[n] = reward
        for n in range(2*n_extra, self.n_extra_episodes):
            reward = self.run_episode(mode='greedy', update=False)
            extra_rewards[n] = reward
        return np.append(rewards, extra_rewards)

    def replay_best_episode(self):
        for _ in range(self.n_replays):
            self.run_episode(mode='replay')

    def run_episode(self, verbose=False, mode='explore', update=True):
        raise NotImplementedError('Vanilla DQL has no implementation without'
                                  'ReplayMemory.')

    def choose_action(self, mode, step=0):
        if mode == 'replay':
            action = self.best_encountered_actions[step]
        elif mode == 'greedy':
            action, q = self.model.get_best_action(self.env.s)
        elif mode == 'explore' or mode == 'explore_and_store':
            if self.exploration == 'uniform':
                if np.random.rand() > self.epsilon:
                    action, q = self.get_best_action(self.env.s)
                else:
                    action, q = self.env.random_action(), np.nan
            elif self.exploration == 'gaussian':
                action, q = self.model.get_best_action(self.env.s)
                # add gaussian fluctuation with std = 0.5*ε
                # (ε = 1 -> 2σ = 1 -> 95% inside [-1, 1])
                action += self.epsilon * 0.5 * np.random.randn(*action.shape)
            else:
                raise NotImplementedError

            if mode == 'explore_and_store':
                self.list_q_chosen_actions.append(q)
                self.list_q_discretized_actions.append(
                    self.model.get_best_discretized_action(self.env.s)[1:]
                )
        else:
            raise ValueError(f'The action selecting mode {mode} does not '
                             'exist.')
        return action

    def save_best_encountered_actions(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write(f'Corresponding reward = '
                        f'{self.best_encountered_reward:.4f}.\n\n\n')
                self.env.render(f, self.best_encountered_actions)
        except Exception:
            print(f'`{filename}` could not be saved.')

    def save_post_episode_rewards(self, filename, n_rewards,
                                  action_sequence=None):
        try:
            if action_sequence is None:
                action_sequence = self.best_encountered_actions
            r1, r2 = self.env.reward(action_sequence, n_rewards)
            with open(filename, 'wb') as f:
                np.save(f, np.array([r1, r2]))
        except Exception:
            print(f'`{filename}` could not be saved.')

    def save_weights(self, filename):
        try:
            with open(filename, 'wb') as f:
                np.save(f, self.model.current_weights)
        except Exception:
            print(f'`{filename}` could not be saved.')

    def save_lists_q_max(self, filename1, filename2=None):
        if not self.verify_argmax_q:
            return
        try:
            if filename2 is None:
                filename1, filename2 = \
                    (filename1[:-4] + '_chosen' + filename1[-4:],
                     filename1[:-4] + '_discretized' + filename1[-4:])
            with open(filename1, 'wb') as f:
                np.save(f, np.array(self.list_q_chosen_actions))
            with open(filename2, 'wb') as f:
                np.save(f,
                        np.array(list(zip(*self.list_q_discretized_actions))))
        except Exception:
            print(f'Could not save `{filename1}`')


class DeepQLearningWithTraces(DeepQLearning):
    """Deep Q-Learning with the addition of eligibility traces.
    Those are basically a compromise between n-step TD
    for all n and Monte Carlo methods (with all steps).
    The backward view is implemented.
    """

    def __init__(self, trace_decay_rate, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        #  λ:
        self.trace_decay_rate = trace_decay_rate
        self.trace = 0 * self.model.current_weights

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
            self.trace += self.model.evaluate_gradient_weights(
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
                delta += self.model.get_best_action(next_state,
                                                    use_target=True)[1]
            self.model.current_weights += learning_rate * delta * self.trace
            if not done:
                #  multiply by λ
                self.trace *= self.trace_decay_rate
                step += 1
        if verbose:
            print(f'\n----------Total Reward: {total_reward:.2f}')
        return total_reward

    def choose_action(self, mode, step=0):
        if self.exploration != "uniform":
            raise ValueError(
                'QL with eligibility traces only works with uniform '
                'exploration. Currently, exploration is '
                f'{self.exploration}.'
            )
        if mode == 'explore' or mode == 'explore_and_store':
            if np.random.rand() > self.epsilon:
                action, q = self.model.get_best_action(self.env.s)
            else:
                action, q = self.env.random_action(), np.nan
                # Only difference with super().choose_action:
                self.trace *= 0

            if mode == 'explore_and_store':
                self.list_q_chosen_actions.append(q)
                self.list_q_discretized_actions.append(
                    self.model.get_best_discretized_action(self.env.s)[1:]
                )
        else:
            action = super().choose_action(mode, step)
        return action


class DQLWithReplayMemory(DeepQLearning):
    """DQN with the addition of a replay memory.
    This implementation is not compatible with eligibility traces.
    """
    import random

    def __init__(self, capacity, sampling_size, NN_optimizer, n_epochs,
                 loss='logcosh', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.compile(optimizer=NN_optimizer, loss=loss,
                           metrics=['mse', 'mae'])
        self.loss = loss
        self.n_epochs = n_epochs
        #  metrics=['accuracy']
        self.memory = models.ReplayMemory(capacity)
        self.sampling_size = sampling_size
        self.batch_size = sampling_size * self.env.n_steps

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
                    self.model.get_best_action(next_state, use_target=True)[1]
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
            self.model.update(memory=self.memory,
                              sampling_size=self.sampling_size,
                              batch_size=self.batch_size,
                              epochs=self.n_epochs)

        return total_reward

    def replay_best_episode(self):
        self.run_episode(mode='replay')

    def save_history(self, filename):
        try:
            import pandas as pd
            #  keys = ['loss', 'mean_squared_error', 'mean_absolute_error']
            keys = self.model.history.keys()
            history_df = pd.DataFrame(self.model.history)
            history_df.rename(columns={'loss': self.loss}, inplace=True)
            print('saving history of NN for metrics: ', keys)
            with open(filename, 'w') as f:
                history_df.to_csv(f, index=False)
                #  np.save(f, history_array)
                #  np.savetxt(f, history_array, fmt='%.8e', delimiter=',',
                #             header=','.join(keys))
        except Exception:
            print(f'`{filename}` could not be saved.')
