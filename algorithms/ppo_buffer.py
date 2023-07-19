from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class PPOBufferAlgorithm(Algorithm):
    def __init__(self, num_players, discount=1., batch_size=256, epsilon=1e-8, buffer_size=2048,
                 num_learning_per_epoch_actor=6, num_learning_per_epoch_critic=3, clip_eps=0.1):
        # self.batch_size = batch_size
        self.num_players = num_players
        self.discount = discount
        self.optimizer_actor = tf.optimizers.legacy.Adam(3e-4)
        self.optimizer_critic = tf.optimizers.legacy.Adam(6e-4)
        self.num_learning_per_epoch_actor = num_learning_per_epoch_actor
        self.num_learning_per_epoch_critic = num_learning_per_epoch_critic
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.clip_eps = clip_eps

        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.d = []
        self.m = []

    """
    states: List of shape (N, S) where N is the number of transitions and S the state space
    actions: List of shape (N, 1)
    rewards: List of shape (N, 1)
    dones: List of shape (N, 1)
    """
    def store_game(self, states, actions, masks, rewards, dones):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        masks = np.array(masks)

        next_states = np.copy(states)[self.num_players:]
        dones = dones[self.num_players:]

        states = states[:-self.num_players]
        masks = masks[:-self.num_players]
        actions = actions[:-self.num_players]
        rewards = rewards[:-self.num_players]
        for i in range(0, len(states), self.num_players):
            self.s.append(states[i:i + self.num_players][tf.where(states[i:i + self.num_players][:, 0] == 1)[0]])
            self.sn.append(next_states[i:i + self.num_players][tf.where(next_states[i:i + self.num_players][:, 0] == 1)[0]])
            self.a.append(actions[i])
            self.r.append(rewards[i])
            self.d.append(dones[i])
            self.m.append(masks[i])

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            if len(lst[i:i + n]) == n:
                yield lst[i:i + n]

    def learn(self, agent: ACAgent):
        if len(self.s) < self.buffer_size:
            return 0.

        self.s = np.array(self.s).squeeze()
        self.a = np.array(self.a)
        self.r = np.array(self.r)[..., None]
        self.sn = np.array(self.sn).squeeze()
        self.d = np.array(self.d)[..., None]
        self.m = np.array(self.m)
        BS = self.s.shape[0]

        self.assert_same_shape(self.s, (BS, self.s.shape[1]))
        self.assert_same_shape(self.a, (BS, self.a.shape[1]))
        self.assert_same_shape(self.r, (BS, 1))
        self.assert_same_shape(self.sn, self.s)
        self.assert_same_shape(self.d, (BS, 1))
        self.assert_same_shape(self.m, self.a)

        loss = 0.
        iterations = 0.

        """
        Estimating TD error for actor update
        """
        val = agent.value_net(self.s)
        new_val = tf.stop_gradient(agent.value_net(self.sn))
        self.assert_same_shape(val, (BS, 1))
        self.assert_same_shape(new_val, (BS, 1))

        """
        Actor update using Clip-PPO
        """

        initial_probs = agent.policy_net(self.s)
        initial_probs = initial_probs * self.m
        initial_probs = initial_probs / tf.reduce_sum(initial_probs, axis=-1, keepdims=True)
        initial_probs = tf.reduce_sum(initial_probs * self.a, axis=-1, keepdims=True)
        initial_probs = np.array(initial_probs)
        for _ in range(self.num_learning_per_epoch_actor):
            perm = np.random.permutation(np.arange(0, len(self.s)))
            for indexes in self.chunks(perm, self.batch_size):
                s = np.array(self.s)[indexes]
                a = np.array(self.a)[indexes]
                r = np.array(self.r)[indexes]
                d = np.array(self.d)[indexes]
                m = np.array(self.m)[indexes]

                reward_to_go = tf.stop_gradient(r + self.discount * np.array(new_val)[indexes] * (1-d))
                td_error = tf.cast(reward_to_go - np.array(new_val)[indexes], tf.float32)
                self.assert_same_shape(reward_to_go, (self.batch_size, 1))
                self.assert_same_shape(reward_to_go, td_error)
                with tf.GradientTape() as a_tape:
                    probs = agent.policy_net(s)
                    probs = probs * m
                    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                    self.assert_same_shape(probs, a)

                    selected_actions_probs = tf.reduce_sum(probs * a, axis=-1, keepdims=True)
                    self.assert_same_shape(selected_actions_probs, (self.batch_size, 1))

                    importance_sampling_ratio = selected_actions_probs / (tf.cast(initial_probs[indexes], tf.float32) + 1e-8)

                    self.assert_same_shape(td_error, importance_sampling_ratio)

                    loss_actor = tf.minimum(
                        td_error * importance_sampling_ratio,
                        td_error * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    )

                    self.assert_same_shape(loss_actor, (self.batch_size, 1))

                    loss_actor = tf.reduce_mean(-loss_actor)
                    self.assert_same_shape(loss_actor, [])

                grad_actor = a_tape.gradient(loss_actor, agent.policy_net.trainable_weights)
                self.optimizer_actor.apply_gradients(zip(grad_actor, agent.policy_net.trainable_weights))

        """
        Value update using TD(0)
        """
        for _ in range(self.num_learning_per_epoch_critic):
            with tf.GradientTape() as c_tape:
                val = agent.value_net(self.s)
                new_val = tf.stop_gradient(agent.value_net(self.sn))
                self.assert_same_shape(val, (BS, 1))
                self.assert_same_shape(new_val, (BS, 1))

                reward_to_go = tf.stop_gradient(self.r + self.discount * new_val * (1-self.d))
                self.assert_same_shape(reward_to_go, (BS, 1))

                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                self.assert_same_shape(loss_critic, (BS, 1))

                loss_critic = tf.reduce_mean(loss_critic)
                self.assert_same_shape(loss_critic, [])

            grad_critic = c_tape.gradient(loss_critic, agent.value_net.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, agent.value_net.trainable_weights))

            loss += loss_critic
            iterations += 1

        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.m = []
        self.d = []

        return loss / (iterations + 1e-10)
