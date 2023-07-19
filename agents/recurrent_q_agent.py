from typing import List

import numpy as np
from brimarl_masked.networks.dqrnet import DQRN
import tensorflow as tf
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class RecurrentDeepQAgent(Agent):
    """Q-learning agent using a recurrent network"""

    def __init__(self, n_actions: int,  epsilon: float, minimum_epsilon: float,
                 epsilon_decay: float, hidden_size: int = 256,
                 fully_connected_layers: int = 256, training=True,
                 name="RecurrentDeepQLearningAgent") -> None:
        """"""
        super().__init__(name)
        self.name = name

        # Network parameters
        self.n_actions = n_actions

        self.policy_net = DQRN(
            hidden_size,
            n_actions,
            fully_connected_layers,
        )
        self.policy_net.build((None, None, 40))

        self.hidden_size = hidden_size
        self.fully_connected_layers = fully_connected_layers
        self.h, self.c = self.policy_net.init_hidden(1)

        # Reinforcement learning parameters
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay
        self.training = training

    def state(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
        """
        Each card is encoded as a vector of length 6:
        first entry: numerical value of the card.
        second entry: boolean value (1 if briscola 0 otherwise).
        last four entries: one hot encoding of the seeds.
        For example "Asso di bastoni" is encoded as:
        [0, 1, 1, 0, 0, 0] if the briscola is "bastoni".
        State:
        [
            my turn 0/1
            my team turn 0/1
            my points (maybe also other team points?)
            move counter
            first card in my hand | 0
            second card in my hand | 0
            third card in my hand | 0
            first card on the table | 0
            second card on the table | 0
            third card on the table | 0
        ]
        """
        state = np.zeros(40)
        value_offset = 4
        seed_offset = 2
        features_per_card = 6
        state[0] = player.id == current_player.id
        state[1] = player.id == current_player.id or env.get_teammate(player.id) == current_player.id
        state[2] = player.points / 60
        state[3] = env.counter / 10

        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + value_offset + seed_offset + card.seed
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1

        for i, card in enumerate(env.played_cards):
            number_index = (i + 3) * features_per_card + value_offset
            seed_index = (i + 3) * features_per_card + seed_offset + card.seed + value_offset
            state[number_index] = card.number
            state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            state[seed_index] = 1
        return state

    def action(self, game: BriscolaGame, player: BriscolaPlayer):# , available_actions: List[int]):
        """Selects action according to an epsilon-greedy policy"""
        """# Select a random action with probability epsilon
        action = np.random.choice(available_actions)
        if np.random.uniform() > self.epsilon or not self.training:
            # Select a greedy action with probability 1 - epsilon
            output, (self.h, self.c) = self.policy_net(
                tf.reshape(self.state(game, player, player), (1, 1, -1)),
                self.h,
                self.c,
            )
            # consider only the first element (ignore the batch dimension)
            output = output.numpy()[0][0]
            sorted_actions = (-output).argsort()
            for predicted_action in sorted_actions:
                if predicted_action in available_actions:
                    action = predicted_action
                    break"""
        assert len(player.hand)
        mask = np.zeros(40)
        for card in player.hand: mask[card.id] = 1

        game_action = np.random.choice(list(range(len(player.hand))))
        if np.random.uniform() > self.epsilon or not self.training:
            output, (self.h, self.c) = self.policy_net(
                tf.reshape(self.state(game, player, player), (1, 1, -1)),
                self.h,
                self.c,
            )
            # consider only the first element (ignore the batch dimension)
            output = output.numpy()[0][0]
            masked_output = (1-mask) * np.finfo(np.float32).min + output
            action = np.argmax(masked_output)
            game_action = np.argwhere([c.id == action for c in player.hand]).squeeze()
            assert player.hand[game_action].id == action

        action = np.zeros(40)
        action[player.hand[game_action].id] = 1
        return game_action, action, mask

    def update_epsilon(self):
        if self.epsilon > self.minimum_epsilon:
            self.epsilon -= self.epsilon_decay
        if self.epsilon < self.minimum_epsilon:
            self.epsilon = self.minimum_epsilon

    def reset(self):
        self.h, self.c = self.policy_net.init_hidden(1)

    def save_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.save_weights(path + self.name + "/q_network")

    def load_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.load_weights(path + self.name + "/q_network")

    def clone(self, training=False):
        assert type(self) is RecurrentDeepQAgent
        new_agent = RecurrentDeepQAgent(
            n_actions=self.n_actions,
            epsilon=self.epsilon,
            minimum_epsilon=self.minimum_epsilon,
            epsilon_decay=self.epsilon_decay,
            hidden_size=self.hidden_size,
            fully_connected_layers=self.fully_connected_layers,
            training=training
        )
        new_agent.policy_net.set_weights([np.copy(weight) for weight in self.policy_net.get_weights()])
        return new_agent

