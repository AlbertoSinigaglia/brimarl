from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_q_agent import ACQAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class A2CQAlgorithm(Algorithm):
    def __init__(self, num_players, discount, num_learning_per_epoch,
                 min_samples=128, entropy_beta=0.0, epsilon=0.,
                 batch_size=128, replace_every=500,
                 replay_memory_capacity=100_000
                 ):

        self.batch_size = batch_size
        self.training_iterations = 0
        self.replace_every = replace_every
        self.num_players = num_players
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_memory = ReplayMemory(replay_memory_capacity)

        self.discount = discount
        self.optimizer_actor = tf.optimizers.legacy.Adam(1e-4)
        self.optimizer_critic = tf.optimizers.legacy.Adam(1e-4)
        self.num_learning_per_epoch = num_learning_per_epoch
        self.entropy_beta = entropy_beta
        self.epsilon = epsilon
        self.min_samples = min_samples

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
        # used for Actor
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        masks = np.array(masks)

        next_states = np.copy(states)[self.num_players:]
        dones = dones[self.num_players:]

        states = states[:-self.num_players]
        actions = actions[:-self.num_players]
        rewards = rewards[:-self.num_players]
        masks = masks[:-self.num_players]

        for i in range(0, len(states), self.num_players):
            self.s.append(states[i:i + self.num_players][tf.where(states[i:i + self.num_players][:, 0] == 1)[0]])
            self.sn.append(next_states[i:i + self.num_players][tf.where(next_states[i:i + self.num_players][:, 0] == 1)[0]])
            self.a.append(actions[i])
            self.r.append(rewards[i])
            self.d.append(dones[i])
            self.m.append(masks[i])

        # used for q net
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        masks = np.array(masks)
        dones = np.array(dones)

        next_states = np.copy(states)[self.num_players:]
        next_masks = np.copy(masks)[self.num_players:]
        dones = dones[self.num_players:]

        states = states[:-self.num_players]
        actions = actions[:-self.num_players]
        rewards = rewards[:-self.num_players]
        for i in range(0, len(states), self.num_players):
            self.replay_memory.push((
                states[i:i + self.num_players][tf.where(states[i:i + self.num_players][:, 0] == 1)[0]],
                next_states[i:i + self.num_players][tf.where(next_states[i:i + self.num_players][:, 0] == 1)[0]],
                actions[i],
                rewards[i],
                dones[i],
                masks[i],
                next_masks[i],
            ))

    def learn(self, agent: ACQAgent):
        if len(self.s) < self.min_samples:
            return None

        # actor training
        self.s = np.array(self.s).squeeze()
        self.a = np.array(self.a)
        self.r = np.array(self.r)
        self.sn = np.array(self.sn).squeeze()
        self.d = np.array(self.d)
        self.m = np.array(self.m)

        initial_selected_probs = None
        for _ in range(self.num_learning_per_epoch):
            with tf.GradientTape() as a_tape:
                probs = agent.policy_net(self.s)
                probs = probs * self.m
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                q = agent.q_net(self.s)
                td_error = tf.stop_gradient(tf.reduce_sum(q * self.a, axis=-1) - tf.reduce_sum(q * probs, axis=-1))
                selected_actions_probs = tf.reduce_sum(probs * self.a, axis=-1)

                if initial_selected_probs is None: initial_selected_probs = tf.stop_gradient(selected_actions_probs)
                entropy = self.entropy_beta * tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + self.epsilon + 1-self.m) * self.m, axis=-1))
                if np.any(selected_actions_probs.numpy() == 0):
                    raise Exception("action prob is zero lol, something is messed up")
                if tf.math.is_nan(entropy) or entropy == 0 and self.entropy_beta > 0:
                    raise Exception(f"entropy messed up: {entropy}")

                # mathematically they should be the same, but you never know
                if self.num_learning_per_epoch > 1:
                    loss_actor = tf.reduce_mean(-td_error * selected_actions_probs/initial_selected_probs)# + entropy
                else:
                    loss_actor = tf.reduce_mean(-td_error * tf.math.log(selected_actions_probs))

            grad_actor = a_tape.gradient(loss_actor, agent.policy_net.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, agent.policy_net.trainable_weights))

        print(np.array(probs[0:2]).round(2))

        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.d = []
        self.m = []


        # q net training

        avg_loss = 0.
        if len(self.replay_memory.memory) >= self.batch_size:
            for _ in range(self.num_learning_per_epoch):
                if self.training_iterations % self.replace_every == 0:
                    print("resetting target net dqn")
                    self.target_net = agent.clone().policy_net

                self.training_iterations += 1

                # Otherwise sample a batch
                batch = self.replay_memory.sample(self.batch_size)

                states = []
                actions = []
                rewards = []
                done = []
                next_states = []
                masks = []
                next_masks = []
                for el in batch:
                    states.append(el[0])
                    next_states.append(el[1])
                    actions.append(el[2])
                    rewards.append(el[3])
                    done.append(el[4])
                    masks.append(el[5])
                    next_masks.append(el[6])

                states = tf.convert_to_tensor(np.array(states).squeeze(), dtype=tf.float32, )
                actions = tf.convert_to_tensor(np.array(actions), dtype=tf.float32, )
                rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32, )
                done = tf.convert_to_tensor(np.array(done), dtype=tf.float32, )
                next_states = tf.convert_to_tensor(np.array(next_states).squeeze(), dtype=tf.float32, )
                masks = tf.convert_to_tensor(np.array(masks), dtype=tf.float32, )
                next_masks = tf.convert_to_tensor(np.array(next_masks), dtype=tf.float32, )

                with tf.GradientTape() as tape:
                    q_values = agent.policy_net(states)
                    q_values = tf.reduce_sum(q_values * actions, axis=-1)
                    with tape.stop_recording():
                        target_q_values = self.target_net(next_states) + (1-next_masks) * np.finfo(np.float32).min
                        target_q_values = tf.reduce_max(target_q_values, axis=-1)

                    target = tf.stop_gradient(rewards + (self.discount * target_q_values * (1 - done)))
                    loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.stop_gradient(target), q_values))

                grad = tape.gradient(loss, agent.policy_net.trainable_weights)
                self.optimizer_critic.apply_gradients(zip(grad, agent.policy_net.trainable_weights))
                avg_loss += loss

        return avg_loss / self.num_learning_per_epoch
