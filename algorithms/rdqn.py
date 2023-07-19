from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.recurrent_q_agent import RecurrentDeepQAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class RQLearningAlgorithm(Algorithm):
    def __init__(self, num_players, replay_memory_capacity, train_on_full_game, sequence_len, batch_size, discount, replace_every,num_learning_per_epoch):
        self.num_players = num_players
        self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.discount = discount
        self.training_iterations = 0
        self.replace_every = replace_every
        self.target_net = None
        self.optimizer = tf.optimizers.legacy.Adam(1e-4)
        self.train_on_full_game = train_on_full_game
        self.num_learning_per_epoch = num_learning_per_epoch

    """
    states: List of shape (N, S) where N is the number of transitions and S the state space
    actions: List of shape (N, 1)
    rewards: List of shape (N, 1)
    dones: List of shape (N, 1)
    """

    def store_game(self, states, actions, masks, rewards, dones):
        if not self.train_on_full_game:
            # for i in range(0, len(states) - self.sequence_len):
            for i in range(0, len(states) - self.sequence_len, self.num_players):
                self.replay_memory.push((
                    states[i:(i + self.sequence_len)],
                    actions[i:(i + self.sequence_len)],
                    rewards[i:(i + self.sequence_len)],
                    dones[i:(i + self.sequence_len)],
                    masks[i:(i + self.sequence_len)],
                ))
        else:
            self.replay_memory.push((
                states,
                actions,
                rewards,
                dones,
                masks,
            ))

    def learn(self, agent: RecurrentDeepQAgent):
        """Samples batch of experience from replay_memory,
        if it has enough samples, and trains the policy network on it
        """
        avg_loss = 0.
        if len(self.replay_memory.memory) >= self.batch_size:
            for _ in range(self.num_learning_per_epoch):
                if self.training_iterations % self.replace_every == 0:
                    print("resetting target net rdqn")
                    self.target_net = agent.clone().policy_net

                self.training_iterations += 1

                # Otherwise sample a batch
                batch = self.replay_memory.sample(self.batch_size)

                states = []
                actions = []
                rewards = []
                done = []
                next_masks = []
                for seq in batch:
                    states.append(seq[0])
                    actions.append(seq[1])
                    rewards.append(seq[2])
                    done.append(seq[3])
                    next_masks.append(seq[4])

                states = tf.convert_to_tensor(np.array(states), dtype=tf.float32, )
                actions = tf.convert_to_tensor(np.array(actions), dtype=tf.float32, )
                rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32, )[...,None]
                done = tf.convert_to_tensor(np.array(done), dtype=tf.float32, )[...,None]
                next_masks = tf.convert_to_tensor(np.array(next_masks), dtype=tf.float32, )
                next_states = tf.identity(states)

                # remove last turn
                states = states[:, :-self.num_players, :]
                actions = actions[:, :-self.num_players]
                rewards = rewards[:, :-self.num_players]

                # remove first turn
                next_states = next_states[:, self.num_players:, :]
                done = done[:, self.num_players:]
                next_masks = next_masks[:, self.num_players:]

                BS = states.shape[0]
                T = states.shape[1]

                self.assert_same_shape(states, (BS, T, states.shape[2]))
                self.assert_same_shape(actions, (BS, T, actions.shape[2]))
                self.assert_same_shape(rewards, (BS, T, 1))
                self.assert_same_shape(next_states, states)
                self.assert_same_shape(done, (BS, T, 1))
                self.assert_same_shape(next_masks, actions)

                mask_for_states_actions_rewards = tf.where(states[:, :, 0] == 1)
                actions = tf.reshape(tf.gather_nd(actions, mask_for_states_actions_rewards), (BS, -1, actions.shape[-1]))
                rewards = tf.reshape(tf.gather_nd(rewards, mask_for_states_actions_rewards), (BS, -1, rewards.shape[-1]))

                mask_for_next_states_done = tf.where(next_states[:, :, 0] == 1)
                done = tf.reshape(tf.gather_nd(done, mask_for_next_states_done), (tf.shape(done)[0], -1))[..., None]
                next_masks = tf.reshape(tf.gather_nd(next_masks, mask_for_next_states_done), (tf.shape(next_masks)[0], -1, tf.shape(next_masks)[-1]))

                NT = actions.shape[1]

                self.assert_same_shape(actions, (BS, NT, actions.shape[2]))
                self.assert_same_shape(rewards, (BS, NT, 1))
                self.assert_same_shape(done, (BS, NT, 1))
                self.assert_same_shape(next_masks, actions)

                with tf.GradientTape() as tape:
                    q_values, *_ = agent.policy_net(states)
                    q_values = tf.reshape(tf.gather_nd(q_values, mask_for_states_actions_rewards), (BS, -1, q_values.shape[-1]))
                    self.assert_same_shape(q_values, actions)
                    q_values = tf.reduce_sum(q_values * actions, axis=-1, keepdims=True)
                    self.assert_same_shape(q_values, (BS, NT, 1))

                    with tape.stop_recording():
                        target_q_values, *_ = self.target_net(next_states)
                        target_q_values = tf.reshape(tf.gather_nd(target_q_values, mask_for_next_states_done), (BS, -1, target_q_values.shape[-1]))
                        self.assert_same_shape(target_q_values, (BS, NT, target_q_values.shape[-1]))
                        target_q_values = target_q_values + (1-next_masks) * np.finfo(np.float32).min
                        self.assert_same_shape(target_q_values, (BS, NT, target_q_values.shape[-1]))
                        target_q_values = tf.reduce_max(target_q_values, axis=-1, keepdims=True)

                    self.assert_same_shape(q_values, target_q_values)
                    target = rewards + (self.discount * target_q_values * (1 - done))
                    self.assert_same_shape(q_values, target)
                    loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.stop_gradient(target), q_values))

                grad = tape.gradient(loss, agent.policy_net.trainable_weights)
                self.optimizer.apply_gradients(zip(grad, agent.policy_net.trainable_weights))
                avg_loss += loss

            agent.update_epsilon()
        return avg_loss / self.num_learning_per_epoch
