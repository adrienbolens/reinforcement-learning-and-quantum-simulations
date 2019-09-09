import numpy as np
import environments as envs
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from actor_critic_models import ActorCriticNetworks


class DeepDeterministicGradientPolicy(object):
    """
    Basic class for Deep Q-Learning.
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
                 system_class,
                 network_type,
                 env_type,
                 sampling_size,
                 exploration='gaussian',
                 n_extra_episodes=0,
                 seed=None,
                 **other_params):

        self.env_type = env_type
        if env_type == 'DynamicalEvolution':
            self.env = envs.ContinuousCurrentGateEnv(system_class=system_class,
                                                     **other_params)
        elif env_type == 'EnergyMinimizer':
            self.env = envs.ContinuousCurrentStateEnergyMinimizer(
                system_class=system_class, **other_params
            )

        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        sess = tf.Session()
        K.set_session(sess)

        self.system_class = system_class
        self.n_episodes = n_episodes
        self.exploration = exploration

        self.network_type = network_type
        if network_type == 'Dense':
            self.actor_critic = ActorCriticNetworks(
                sess=sess,
                env=self.env,
                **other_params
            )

        if network_type == 'LSTM':
            pass

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.best_encountered_actions = None
        self.best_encountered_rewards = None

        self.n_extra_episodes = n_extra_episodes

    def run(self):
        #  rewards = np.zeros((self.n_episodes, self.env.n_steps))
        rewards = np.zeros((self.n_episodes, 1))
        self.best_encountered_actions = self.env.initial_action_sequence()
        self.env.reset()
        if self.env_type == 'DynamicalEvolution':
            self.best_encountered_rewards = (
                [0]*(len(self.best_encountered_actions) - 1)
                + [self.env.reward(self.best_encountered_actions)]
            )
        elif self.env_type == 'EnergyMinimizer':
            self.best_encountered_rewards = (
                [self.env.reward(action=a)
                 for a in self.best_encountered_actions]
            )
        #  self.env.reward(action_sequence=self.best_encountered_actions)
        print('Final reward of the initial action sequence (e.g. Trotter) is '
              f'{self.best_encountered_rewards[-1]}')
        for episode in range(self.n_episodes):
            mode = 'explore'
            verbose = False
            if episode % (self.n_episodes//10) == 0:
                verbose = True
                print(f'Episode {episode}: ')
            reward_sequence = self.run_episode(verbose, mode=mode)
            if reward_sequence[-1] > self.best_encountered_rewards[-1]:
                self.best_encountered_rewards = reward_sequence
                self.best_encountered_actions = self.env.action_sequence
            rewards[episode, -1] = reward_sequence[-1]

            #  replay the best episode
            if (self.replay_spacing != 0 and
                    episode % self.replay_spacing == 0 and
                    episode < self.n_episodes - 1):
                if episode == 0 and self.system_class != 'LongRangeIsing':
                    pass
                else:
                    self.replay_best_episode()

            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print(f"Final epsilon: {self.epsilon:.2f}.\n")
        self.env.render()
        print(f'\nReward of run with final Q: ',
              [f'{r:.4f}' for r in reward_sequence])
        print(f'\nBest encountered reward: ',
              [f'{r:.4f}' for r in self.best_encountered_rewards])

        if self.n_extra_episodes == 0:
            return rewards

        #  append extra test episodes at the end of the run
        extra_rewards = np.zeros((self.n_extra_episodes, 1))
        n_extra = self.n_extra_episodes//3
        # greedy: no exploration. with update: NNs still learning
        for n in range(n_extra):
            reward_sequence = self.run_episode(mode='greedy', update=True)
            extra_rewards[n, 1] = reward_sequence[-1]
        # explore: with exploration. no update: NNs frozen
        for n in range(n_extra, 2*n_extra):
            reward_sequence = self.run_episode(mode='explore', update=False)
            extra_rewards[n, -1] = reward_sequence[-1]
        # greedy with no update (reward should stay the same: sanity check
        for n in range(2*n_extra, self.n_extra_episodes):
            reward_sequence = self.run_episode(mode='greedy', update=False)
            extra_rewards[n, -1] = reward_sequence[-1]
        return np.append(rewards, extra_rewards, axis=0)

    def run_episode(self, verbose=False, mode='explore', update=True):
        if mode == 'replay':
            calculate_reward = False
        else:
            calculate_reward = True
        done = False
        current_state = self.env.reset()
        step = 0
        if self.network_type == 'LSTM':
            self.model.reset()
        episode = []
        while not done:
            state_input = self.env.process_state(current_state)
            action = self.choose_action(mode, state_input, step)
            if self.network_type == 'LSTM':
                self.model.update_network_input(self.env.s, action, step)
            #  state_sequence.append(self.env.s)
            next_state, reward, done, _ = self.env.step(action,
                                                        calculate_reward)
            next_state_input = self.env.process_state(next_state)
            if not calculate_reward:
                reward = self.best_encountered_rewards[step]
            episode.append(
                (state_input, action, reward, next_state_input, done)
            )
            current_state = next_state
            step += 1

        if calculate_reward:
            reward_sequence = self.env.reward_sequence
        else:
            reward_sequence = self.best_encountered_rewards

        if verbose:
            print(f'\n----------Total Reward: {reward_sequence[-1]:.2f}')

#          actor_critic.train()
#          cur_state = new_state
        if mode == 'replay':
            for _ in self.n_replays:
                self.actor_critic.remember(episode)
        else:
            self.actor_critic.remember(episode)

        if update:
            self.actor_critic.train(self.sampling_size)

        return reward_sequence

    def replay_best_episode(self):
        self.run_episode(mode='replay')

    def choose_action(self, mode, state_input, step=0):
        if mode == 'replay':
            action = self.best_encountered_actions[step]
        elif mode == 'greedy':
            action = self.actor_critic.act(state_input)
        elif mode == 'explore':
            if self.exploration == 'uniform':
                if np.random.rand() > self.epsilon:
                    action = self.actor_critic.act(state_input)
                else:
                    action = self.env.random_action()
            elif self.exploration == 'gaussian':
                action = self.actor_critic.act(state_input)
                # add gaussian fluctuation with std = 0.5*ε
                # (ε = 1 -> 2σ = 1 -> 95% inside [-1, 1])
                std = 0.5 * self.epsilon
                action += std * np.random.randn(*action.shape)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f'The action selecting mode `{mode}` does not '
                             'exist.')
        return action

    def get_initial_sequence(self):
        initial_action_sequence = self.env.initial_action_sequence()
        reward = self.env.reward(action_sequence=initial_action_sequence)
        return initial_action_sequence, reward

    def save_best_encountered_actions(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write('Corresponding reward = [')
                for a in self.best_encountered_rewards:
                    f.write(f' {a:.4f} ')
                f.write('] \n\n\n')
                self.env.render(f, self.best_encountered_actions)
        except Exception as e:
            print(f'`{filename}` could not be saved.')
            print('--> ', e)

    #  def save_history(self, filename):
    #      try:
    #          import pandas as pd
    #          #  keys = ['loss', 'mean_squared_error', 'mean_absolute_error']
    #          keys = self.model.history.keys()
    #          history_df = pd.DataFrame(self.model.history)
    #          history_df.rename(columns={'loss': self.loss}, inplace=True)
    #          print('saving history of NN for metrics: ', keys)
    #          with open(filename, 'w') as f:
    #              history_df.to_csv(f, index=False)
    #              #  np.save(f, history_array)
    #              #  np.savetxt(f, history_array, fmt='%.8e', delimiter=',',
    #              #             header=','.join(keys))
    #      except Exception as e:
    #          print(f'`{filename}` could not be saved.')
    #          print('--> ', e)
