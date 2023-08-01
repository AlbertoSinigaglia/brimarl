from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from brimarl_masked.algorithms.ema import ExponentialMovingStdDev
from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class PPOAlgorithmParallel(Algorithm):
    def __init__(self, num_players, discount=1., min_samples=128, epsilon=1e-8,
                 num_learning_per_epoch_actor=6, num_learning_per_epoch_critic=3, clip_eps=0.1,
                 lr_actor=1e-4, lr_critic=3e-4):
        # self.batch_size = batch_size
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


        self.games = []
        self.thread_pool_executor = ProcessPoolExecutor(max_workers=10)

    def store_game(self, states, actions, masks, rewards, dones):
        self.games.append((
            states, actions, rewards, dones, masks
        ))

    def learn(self, agent: ACAgent):
        if len(self.games) * (40 / self.num_players) < self.min_samples:
            return 0.

        s, sn, a, d, m, r = zip(
            *self.thread_pool_executor.map(perform_single, zip(
                self.games, repeat(self.num_players)
            ))
        )

        s = np.array(s).squeeze()
        a = np.array(a)
        r = np.array(r)[..., None]
        sn = np.array(sn).squeeze()
        d = np.array(d)[..., None]
        m = np.array(m)

        dim_1 = s.shape[0] * s.shape[1]

        s = s.reshape((dim_1, -1))
        sn = sn.reshape((dim_1, -1))
        a = a.reshape((dim_1, -1))
        m = m.reshape((dim_1, -1))
        r = r.reshape((dim_1, 1))
        d = d.reshape((dim_1, 1))

        BS = s.shape[0]

        self.assert_same_shape(s, (BS, s.shape[1]))
        self.assert_same_shape(a, (BS, a.shape[1]))
        self.assert_same_shape(r, (BS, 1))
        self.assert_same_shape(sn, s)
        self.assert_same_shape(d, (BS, 1))
        self.assert_same_shape(m, a)

        self.sigma_ema.update(r)
        r = r / (self.sigma_ema.get_stddev() + self.epsilon)

        loss = 0.
        iterations = 0.

        """
        Estimating TD error for actor update
        """
        val = agent.value_net(s)
        new_val = tf.stop_gradient(agent.value_net(sn))
        self.assert_same_shape(val, (BS, 1))
        self.assert_same_shape(new_val, (BS, 1))

        reward_to_go = tf.stop_gradient(r + self.discount * new_val * (1-d))
        td_error = reward_to_go - val
        self.assert_same_shape(reward_to_go, (BS, 1))
        self.assert_same_shape(reward_to_go, td_error)

        """
        Actor update using Clip-PPO
        """
        initial_probs = None
        for _ in range(self.num_learning_per_epoch_actor):
            with tf.GradientTape() as a_tape:
                probs = agent.policy_net(s)
                probs = probs * m
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                self.assert_same_shape(probs, a)

                selected_actions_probs = tf.reduce_sum(probs * a, axis=-1, keepdims=True)
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
                val = agent.value_net(s)
                new_val = tf.stop_gradient(agent.value_net(sn))
                self.assert_same_shape(val, (BS, 1))
                self.assert_same_shape(new_val, (BS, 1))

                reward_to_go = tf.stop_gradient(r + self.discount * new_val * (1-d))
                self.assert_same_shape(reward_to_go, (BS, 1))

                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                self.assert_same_shape(loss_critic, (BS, 1))

                loss_critic = tf.reduce_mean(loss_critic)
                self.assert_same_shape(loss_critic, [])

            grad_critic = c_tape.gradient(loss_critic, agent.value_net.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, agent.value_net.trainable_weights))

            loss += loss_critic
            iterations += 1

        self.games = []

        return loss / (iterations + 1e-10)



def perform_single(params):
    data, num_players = params
    states, actions, rewards, dones, masks = data
    next_states = np.copy(states)[num_players:]
    dones = np.array(dones)[num_players:]

    states = np.array(states)[:-num_players]
    masks = np.array(masks)[:-num_players]
    actions = np.array(actions)[:-num_players]
    rewards = np.array(rewards)[:-num_players]

    ret_states = []
    ret_next_states = []
    ret_actions = []
    ret_dones = []
    ret_masks = []
    ret_rewards = []

    for i in range(0, len(states), num_players):
        s_1 = states[i:i + num_players][tf.where(states[i:i + num_players][:, 0] == 1)[0]]
        sn_1 = next_states[i:i + num_players][tf.where(next_states[i:i + num_players][:, 0] == 1)[0]]
        ret_states.append(s_1)
        ret_next_states.append(sn_1)
        ret_actions.append(actions[i])
        ret_rewards.append(rewards[i])
        ret_dones.append(dones[i])
        ret_masks.append(masks[i])

    return ret_states, ret_next_states, ret_actions, ret_dones, ret_masks, ret_rewards
