from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class A2CClippedAlgorithm(Algorithm):
    def __init__(self, num_players, discount=1., min_samples=128, epsilon=1e-8,
                 num_learning_per_epoch_actor=10, num_learning_per_epoch_critic=3, clip_eps=0.1):
        # self.batch_size = batch_size
        self.num_players = num_players
        self.discount = discount
        self.optimizer_actor = tf.optimizers.legacy.Adam(3e-4)
        self.optimizer_critic = tf.optimizers.legacy.Adam(6e-4)
        self.num_learning_per_epoch_actor = num_learning_per_epoch_actor
        self.num_learning_per_epoch_critic = num_learning_per_epoch_critic
        self.epsilon = epsilon
        self.min_samples = min_samples
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
        for iteration in range(self.num_learning_per_epoch_actor):
            with tf.GradientTape() as a_tape:
                probs = agent.policy_net(self.s)
                probs = probs * self.m
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                self.assert_same_shape(probs, self.a)

                selected_actions_probs = tf.reduce_sum(probs * self.a, axis=-1, keepdims=True)
                self.assert_same_shape(selected_actions_probs, (BS, 1))

                if initial_probs is None:
                    initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                self.assert_same_shape(selected_actions_probs, initial_probs)

                remove_extreme_p = (
                        1-tf.cast(tf.logical_or(
                            tf.logical_and(selected_actions_probs > 0.95, td_error > 0),
                            tf.logical_and(selected_actions_probs < 0.05, td_error < 0)
                        ), tf.float32) ) + \
                        self.d # for the last step (p will always be 1)
                self.assert_same_shape(remove_extreme_p, selected_actions_probs)

                r_theta = selected_actions_probs / (initial_probs + self.epsilon)

                self.assert_same_shape(r_theta, selected_actions_probs)

                clipped_r_theta = tf.clip_by_value(r_theta, 1 - self.clip_eps, 1 + self.clip_eps)

                self.assert_same_shape(clipped_r_theta, r_theta)

                non_masked_probs = tf.cast(r_theta == clipped_r_theta, tf.float32)
                ratio = tf.reduce_mean(non_masked_probs)
                if ratio < 0.5:
                    print(f"breaking at {iteration}")
                    break

                self.assert_same_shape(non_masked_probs, selected_actions_probs)
                self.assert_same_shape(non_masked_probs, val)
                self.assert_same_shape(ratio, [])

                loss_actor = -td_error * clipped_r_theta * ratio * remove_extreme_p
                self.assert_same_shape(loss_actor, selected_actions_probs)

                loss_actor = tf.reduce_mean(loss_actor)
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
        """print()
        print(np.array(probs[0:2]).round(2))
        print(np.array(val[0:20]).squeeze().round(2))"""

        return loss / (iterations + 1e-10)
