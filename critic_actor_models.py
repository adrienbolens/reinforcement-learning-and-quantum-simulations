"""
actor-critic models
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers.merge import Add
from tensorflow.keras.optimizers import Adam
#  import tensorflow.keras.backend as K

import random
from collections import deque

print('Tensorflow verion: ', tf.__version__)
print('Tensorflow file: ', tf.__file__)
print('Keras verion: ', keras.__version__)
print('Keras file: ', keras.__file__)


#  Implementation of the Deep Deterministic Policy Gradient (DDPG)
#  based on `Lillicrap, et al., 2015`.

class ActorCriticNetworks:
    def __init__(self,
                 #  architecture,
                 sess,
                 env,
                 #  capacity,
                 tf_seed=None,
                 **other_params):

        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        #  self.epsilon = 1.0
        #  self.epsilon_decay = .995
        self.gamma = 1.0
        #  self.gamma = .95
        self.tau = .125

        self.capacity = 2000

        # ====================================================================
        # Actor Model
        # ====================================================================
        # Network for the policy
        # (In contrast to Q-learning where the actions are just argmax Q)
        # --> compatible with continuous action space
        #
        # Updated using policy gradient: ΔΘA ~ +dQ/dΘA = dQ/da * dA/dΘA
        # A: Actor Model, AΘ: its parameters, a: action (Q = Q(s, a))
        # Q: Critic Model (Q function)

        self.memory = deque(maxlen=self.capacity)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # where we will feed dQ/da (from the critic model)
        self.actor_critic_grads = tf.placeholder(
            tf.float32, [None, self.env.get_action_len()]
        )
        actor_model_weights = self.actor_model.trainable_weights

        # dA/dΘA (from the actor model)
        # (already weighted by -dQ/da through grad_ys)
        # gradient ASCENT -> - sign
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        grad_ys=-self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate
        ).apply_gradients(grads)

        # ====================================================================
        # Critic Model
        # ====================================================================
        # Network for modeling Q
        # similar to DQN: update the network by  minimizing the loss L(Q-y)
        # where yi = ri + γQ'(si+1, A'(si+1))

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        # where we calculate dQ/da needed for the update of ΘA above
        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ====================================================================
    # Network Architecture
    # ====================================================================

    def create_actor_model(self):
        state_input = Input(shape=self.env.get_n_inputs())
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        #  using tanh for actions between -1 and 1
        output = Dense(self.env.get_action_len(), activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        #  adam  = Adam(lr=self.learning_rate)
        #  not really needed as we apply_gradients explicitly
        #  only needed for defining metrics
        #  model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.get_n_inputs())
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.get_action_len())
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        #
        #  now focusing on DynamicalEvolution:
        #  using sigmoid for Q function between 0 and 1 (= fidelity)
        #
        output = Dense(1, activation='sigmoid')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ====================================================================
    # Model Training
    # ====================================================================

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(state)
            critic_grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: state,
                self.actor_critic_grads: critic_grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        #  rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ====================================================================
    # Target Model Updates
    # ====================================================================

    def _update_actor_target(self):
        weights = np.array(self.actor_model.get_weights())
        weights_target = np.array(self.target_actor_model.get_weights())

        self.target_actor_model.set_weights(
            self.tau * weights + (1 - self.tau) * weights_target
        )

    def _update_critic_target(self):
        weights = np.array(self.critic_model.get_weights())
        weights_target = np.array(self.target_critic_model.get_weights())

        self.target_critic_model.set_weights(
            self.tau * weights + (1 - self.tau) * weights_target
        )

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ====================================================================
    # Model Predictions
    # ====================================================================

    def act(self, state, std=None):
        """Choose an action and add Gaussian noise to it"""
        action = self.actor_model.predict(state)
        if std is None:
            return action
        else:
            # add gaussian fluctuation with std = 0.5*ε
            # (ε = 1 -> 2σ = 1 -> 95% inside [-1, 1])
            action += std * np.random.randn(*action.shape)
            return action

#  def main():
#      sess = tf.Session()
#      K.set_session(sess)
#      env = gym.make("Pendulum-v0")
#      actor_critic = ActorCritic(env, sess)

#      num_trials = 10000
#      trial_len  = 500

#      cur_state = env.reset()
#      action = env.action_space.sample()
#      while True:
#          env.render()
#          cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
#          action = actor_critic.act(cur_state)
#          action = action.reshape((1, env.action_space.shape[0]))

#          new_state, reward, done, _ = env.step(action)
#          new_state = new_state.reshape((1, env.observation_space.shape[0]))

#          actor_critic.remember(cur_state, action, reward, new_state, done)
#          actor_critic.train()

#          cur_state = new_state

#  if __name__ == "__main__":
#      main()
