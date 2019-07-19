import random
import numpy as np
import sys
import environments as envs


class QLearning(object):

    def __init__(self,
                 n_episodes,
                 learning_rate,
                 epsilon_max,
                 epsilon_min,
                 epsilon_decay,
                 n_replays,
                 replay_spacing,
                 lam,
                 system_class,
                 seed=None,
                 **other_params
                 ):
        self.env = envs.CurrentGateStateEnv(system_class=system_class,
                                            **other_params)
        self.system_class = system_class
        self.n_episodes = n_episodes
        self.lam = lam
        # Q-learning algorithm with eligibility trace and replay:
        print("Creating the Q-function of size {} (state) x {} (action) = {}."
              .format(self.env.observation_space.n, self.env.action_space.n,
                      self.env.observation_space.n * self.env.action_space.n))
        self.q_matrix = np.zeros((self.env.observation_space.n,
                                  self.env.action_space.n), dtype=np.float32)
        self.trace = np.zeros((self.env.observation_space.n,
                               self.env.action_space.n), dtype=np.float32)
        print(f'The Q-function takes {sys.getsizeof(self.q_matrix)/1e6:.1f} '
              'MB of memory.\n')
        self.learning_rate = learning_rate
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_replays = n_replays
        self.replay_spacing = replay_spacing
        random.seed(seed)
        self.best_encountered_actions = None
        self.best_encountered_reward = None
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
            verbose = False
            if episode % (self.n_episodes//10) == 0:
                verbose = True
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

    def save_q_matrix(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.q_matrix)

    def run_episode(self, verbose=False, mode='explore', update=True):
        #  q_matrix, mode, learning_rate=None, verbose=False,
        #              update_q_matrix=True):
        if mode == 'replay':
            learning_rate = 1
            calculate_reward = False
        else:
            learning_rate = self.learning_rate
            calculate_reward = True
        done = False
        total_reward, reward = 0, 0
        self.env.reset()
        self.trace.fill(0.0)
        step = 0
        while not done:
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
            self.trace[state, action] += 1
            delta = - self.q_matrix[state, action]
            if done:
                delta += reward
                self.q_matrix += learning_rate * delta * self.trace
                continue
            # we could remove last line of q_matrix
            # assuming q_matrix[next_state, i] = 0 for all i. (not anymore)
            # Otherwise: += learning_rate * (reward - q_matrix[state, action])
            # if done == True
            delta += np.max(self.q_matrix[next_state])
            self.q_matrix += learning_rate * delta * self.trace
            self.trace *= self.lam
            step += 1
        if verbose:
            print(f'----------Total Reward: {total_reward:.2f}')
        return total_reward

    def choose_action(self, mode, step=0):
        if mode == 'explore':
            # With the probability of (1 - epsilon) take the best action in our
            # Q-table
            if random.uniform(0, 1) > self.epsilon:
                action = np.argmax(self.q_matrix[self.env.s])
                # Else take a random action
            else:
                action = self.env.action_space.sample()
                self.trace.fill(0.0)
        elif mode == 'replay':
            action = self.best_encountered_actions[step]
        elif mode == 'greedy':
            action = np.argmax(self.q_matrix[self.env.s])
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


class Rerun(QLearning):

    def __init__(self, q_matrix, **other_params):
        print('WARNING: This is a rerun with a preset q_matrix, which will not'
              'be updated')
        super().__init__(**other_params)
        self.q_matrix = q_matrix

    def run(self):
        rewards = np.zeros(self.n_episodes)
        self.best_encountered_actions = self.env.initial_action_sequence()
        self.best_encountered_reward = \
            self.env.reward(self.best_encountered_actions)
        print('Fidelity of the initial action sequence (e.g. Trotter) is '
              f'{self.best_encountered_reward}')
        for episode in range(self.n_episodes):
            verbose = False
            if episode % (self.n_episodes//10) == 0:
                verbose = True
                print(f'Episode {episode}: ')
            reward = self.run_episode(verbose, mode='explore', update=False)
            if reward > self.best_encountered_reward:
                self.best_encountered_reward = reward
                self.best_encountered_actions = self.env.action_sequence
            rewards[episode] = reward
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print(f"Final epsilon: {self.epsilon:.2f}.\n")
        self.env.render()
        print(f'\nFidelity of run with final Q: {reward:.4f}')
        print(f'\nBest encountered fidelity: '
              f'{self.best_encountered_reward:.4f}')
        return rewards
