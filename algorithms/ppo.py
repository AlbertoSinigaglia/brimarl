from brimarl_masked.algorithms.ema import ExponentialMovingStdDev
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class PPOAlgorithm(Algorithm):
    def __init__(self, num_players, discount=1., min_samples=128, epsilon=1e-8,
                 num_learning_per_epoch_actor=6, num_learning_per_epoch_critic=3, clip_eps=0.1,
                 lr_actor=1e-4, lr_critic=3e-4):
        # self.batch_size = batch_size
        super().__init__()
        self.num_players = num_players
        self.discount = discount
        self.optimizer_actor = tf.optimizers.legacy.Adam(lr_actor)
        self.optimizer_critic = tf.optimizers.legacy.Adam(lr_critic)
        self.num_learning_per_epoch_actor = num_learning_per_epoch_actor
        self.num_learning_per_epoch_critic = num_learning_per_epoch_critic
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.clip_eps = clip_eps
        self.sigma_ema = ExponentialMovingStdDev()

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

    def learn(self, agent: ACAgent):
        if len(self.s) < self.min_samples:
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

        self.sigma_ema.update(self.r)
        self.r = self.r / (self.sigma_ema.get_stddev() + self.epsilon)

        loss = 0.
        iterations = 0.

        """
        Estimating TD error for actor update
        """
        val = agent.value_net(self.s)
        new_val = tf.stop_gradient(agent.value_net(self.sn))
        self.assert_same_shape(val, (BS, 1))
        self.assert_same_shape(new_val, (BS, 1))

        reward_to_go = tf.stop_gradient(self.r + self.discount * new_val * (1-self.d))
        td_error = reward_to_go - val
        self.assert_same_shape(reward_to_go, (BS, 1))
        self.assert_same_shape(reward_to_go, td_error)

        """
        Actor update using Clip-PPO
        """
        initial_probs = None
        for _ in range(self.num_learning_per_epoch_actor):
            with tf.GradientTape() as a_tape:
                probs = agent.policy_net(self.s)
                probs = probs * self.m
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                self.assert_same_shape(probs, self.a)

                selected_actions_probs = tf.reduce_sum(probs * self.a, axis=-1, keepdims=True)
                self.assert_same_shape(selected_actions_probs, (BS, 1))

                if initial_probs is None:
                    initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                importance_sampling_ratio = selected_actions_probs / (initial_probs + 1e-8)

                self.assert_same_shape(td_error, importance_sampling_ratio)

                loss_actor = tf.minimum(
                    td_error * importance_sampling_ratio,
                    td_error * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                )

                self.assert_same_shape(loss_actor, (BS, 1))

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





