from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.gae import GAEAlgorithm


class PPOGAEAlgorithm(GAEAlgorithm):
    def __init__(self, num_players, discount, num_learning_per_epoch, min_samples=128, entropy_beta=0.0, epsilon=0.,
                ppo_clip_eps=0.1):
        super().__init__(discount)
        self.num_players = num_players
        self.discount = discount
        self.optimizer_critic = tf.optimizers.legacy.Adam(1e-3/2)
        self.optimizer_actor = tf.optimizers.legacy.Adam(1e-3/2)
        self.num_learning_per_epoch = num_learning_per_epoch
        self.entropy_beta = entropy_beta
        self.epsilon = epsilon
        self.min_samples = min_samples

        self.mu = tf.metrics.Mean()
        self.sigma = tf.metrics.Mean()
        self.ppo_clip_eps = ppo_clip_eps

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

        self.s.append([])
        self.sn.append([])
        self.a.append([])
        self.r.append([])
        self.d.append([])
        self.m.append([])

        for i in range(0, len(states), self.num_players):
            self.s[-1].append(states[i:i + self.num_players][tf.where(states[i:i + self.num_players][:, 0] == 1)[0]])
            self.sn[-1].append(next_states[i:i + self.num_players][tf.where(next_states[i:i + self.num_players][:, 0] == 1)[0]])
            self.a[-1].append(actions[i])
            self.m[-1].append(masks[i])
            self.r[-1].append(rewards[i])
            self.d[-1].append(dones[i])


    def learn(self, agent: ACAgent):
        if len(self.s) * len(self.s[0]) < self.min_samples:
            return None

        self.s = np.array(self.s).squeeze()
        self.a = np.array(self.a)
        self.r = np.array(self.r)[..., None]
        self.sn = np.array(self.sn).squeeze()
        self.d = np.array(self.d)[..., None]
        self.m = np.array(self.m)

        BS = self.s.shape[0]
        T = self.s.shape[1]

        self.assert_same_shape(self.s, (BS, T, self.s.shape[-1]))
        self.assert_same_shape(self.a, (BS, T, self.a.shape[-1]))
        self.assert_same_shape(self.r, (BS, T, 1))
        self.assert_same_shape(self.sn, self.s)
        self.assert_same_shape(self.d, (BS, T, 1))
        self.assert_same_shape(self.m, self.a)

        flat_s = self.s.reshape((-1, self.s.shape[-1]))
        flat_sn = self.sn.reshape((-1, self.sn.shape[-1]))
        flat_a = self.a.reshape((-1, self.a.shape[-1]))
        flat_m = self.m.reshape((-1, self.m.shape[-1]))
        flat_r = self.r.reshape((-1, 1))
        flat_d = self.d.reshape((-1, 1))

        self.assert_same_shape(flat_s, (BS*T, self.s.shape[-1]))
        self.assert_same_shape(flat_sn, flat_s)
        self.assert_same_shape(flat_a, (BS*T, self.a.shape[-1]))
        self.assert_same_shape(flat_m, flat_a)
        self.assert_same_shape(flat_r, (BS*T, 1))
        self.assert_same_shape(flat_d, (BS*T, 1))

        loss = 0.
        iterations = 0.
        initial_probs = None
        for _ in range(self.num_learning_per_epoch):
            with tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
                flat_val = agent.value_net(flat_s)
                val = tf.reshape(flat_val, (BS, T, flat_val.shape[-1]))
                flat_new_val = tf.stop_gradient(agent.value_net(flat_sn))
                new_val = tf.reshape(flat_new_val, (BS, T, flat_new_val.shape[-1]))

                self.assert_same_shape(flat_val, (BS*T, 1))
                self.assert_same_shape(val, (BS, T, 1))
                self.assert_same_shape(flat_new_val, (BS*T, 1))
                self.assert_same_shape(new_val, (BS, T, 1))

                reward_to_go = self.r + self.discount * new_val * (1-self.d)
                flat_reward_to_go = tf.reshape(reward_to_go, (BS*T, 1))
                gae_ = tf.stop_gradient(self.gae(reward_to_go - val))

                self.assert_same_shape(reward_to_go, val)
                self.assert_same_shape(flat_reward_to_go, flat_val)
                self.assert_same_shape(gae_, val)

                self.mu.update_state(np.mean(gae_))
                self.sigma.update_state(np.std(gae_))

                gae_ = (gae_ - self.mu.result()) / (self.sigma.result() + 1e-6)
                flat_gae = tf.reshape(gae_, (-1, 1))

                self.assert_same_shape(flat_gae, (BS*T, 1))

                probs = agent.policy_net(flat_s)
                probs = probs * flat_m
                probs = probs / (tf.reduce_sum(probs, axis=-1, keepdims=True) + self.epsilon)

                selected_actions_probs = tf.reduce_sum(probs * flat_a, axis=-1, keepdims=True)

                self.assert_same_shape(probs, flat_a)
                self.assert_same_shape(selected_actions_probs, flat_gae)

                initial_probs = tf.stop_gradient(initial_probs if initial_probs is not None else selected_actions_probs)
                if np.any(selected_actions_probs.numpy() == 0): raise Exception("action prob is zero lol, something is messed up")

                importance_sampling_ratio = selected_actions_probs / (initial_probs + self.epsilon)
                clipped_r_theta = tf.clip_by_value(importance_sampling_ratio, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps)

                self.assert_same_shape(flat_gae, importance_sampling_ratio)
                self.assert_same_shape(flat_gae, clipped_r_theta)

                ratio = (importance_sampling_ratio == clipped_r_theta).numpy().sum() / importance_sampling_ratio.shape[0]

                self.assert_same_shape(np.array(ratio), [])

                loss_actor = ratio * -tf.minimum(
                    flat_gae * importance_sampling_ratio,
                    flat_gae * clipped_r_theta
                )
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[..., None]

                self.assert_same_shape(loss_actor, (BS * T, 1))
                self.assert_same_shape(loss_critic, (BS, T, 1))

                loss_critic = tf.reduce_mean(loss_critic)
                loss_actor = tf.reduce_mean(loss_actor)

                self.assert_same_shape(loss_critic, [])
                self.assert_same_shape(loss_actor, [])

            grad_actor = a_tape.gradient(loss_actor, agent.policy_net.trainable_weights)
            grad_critic = c_tape.gradient(loss_critic, agent.value_net.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, agent.policy_net.trainable_weights))
            self.optimizer_critic.apply_gradients(zip(grad_critic, agent.value_net.trainable_weights))

            loss += loss_critic
            iterations += 1

        """examples = np.random.randint(0, len(probs), 2)
        print(probs.numpy()[examples].round(2))
        print(gae_.numpy()[examples].round(2).squeeze())
        print(
            gae_.numpy()[examples].round(2).squeeze().min(),
            gae_.numpy()[examples].round(2).squeeze().max(),
            gae_.numpy().mean()
        )"""
        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.d = []
        self.m = []

        return loss / (iterations + 1e-10)
