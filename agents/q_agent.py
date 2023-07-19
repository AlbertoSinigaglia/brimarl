from typing import List

import numpy as np
from brimarl_masked.networks.dqnet import DQN
import tensorflow as tf
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class DeepQAgent(Agent):
    """Q-learning agent using a recurrent network"""

    def __init__(self, epsilon: float, minimum_epsilon: float,
                 epsilon_decay: float, hidden_size: int = 256,
                 fully_connected_layers: int = 256, training=True,
                 name="DeepQAgent") -> None:
        """"""
        super().__init__(name)
        self.name = name

        self.policy_net = DQN(40)
        self.policy_net.build((None, 244))

        self.hidden_size = hidden_size
        self.fully_connected_layers = fully_connected_layers

        # Reinforcement learning parameters
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay
        self.training = training
        self.current_game_state = np.zeros((40, 2))

    def state(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
        turn_state = np.zeros(4)
        turn_state[0] = player.id == current_player.id
        turn_state[1] = player.id == current_player.id or env.get_teammate(player.id) == current_player.id
        turn_state[2] = player.points / 60
        turn_state[3] = env.counter / 10

        player_hand_cards = np.zeros((40,2))
        for i, card in enumerate(player.hand):
            player_hand_cards[card.id, 0] = 1
            player_hand_cards[card.id, 1] = 1 if card.seed == env.briscola.seed else 0

        played_cards = np.zeros((40,2))
        for i, card in enumerate(env.played_cards):
            played_cards[card.id] = 1
            played_cards[card.id, 1] = 1 if card.seed == env.briscola.seed else 0
            self.current_game_state[card.id, 0] = 1
            self.current_game_state[card.id, 1] = 1 if card.seed == env.briscola.seed else 0

        return np.concatenate((
            turn_state.reshape((-1)),
            player_hand_cards.reshape((-1)),
            played_cards.reshape((-1)),
            self.current_game_state.reshape((-1))
        ))

    def action(self, game: BriscolaGame, player: BriscolaPlayer):
        assert len(player.hand)
        mask = np.zeros(40)
        for card in player.hand: mask[card.id] = 1

        game_action = np.random.choice(list(range(len(player.hand))))
        if np.random.uniform() > self.epsilon or not self.training:
            output = self.policy_net(
                tf.reshape(self.state(game, player, player), (1, -1)),
            )
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
        self.current_game_state = np.zeros_like(self.current_game_state)

    def save_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.save_weights(path + self.name + "/q_network")

    def load_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.load_weights(path + self.name + "/q_network")

    def clone(self, training=False):
        assert type(self) is DeepQAgent
        new_agent = DeepQAgent(
            epsilon=self.epsilon,
            minimum_epsilon=self.minimum_epsilon,
            epsilon_decay=self.epsilon_decay,
            hidden_size=self.hidden_size,
            fully_connected_layers=self.fully_connected_layers,
            training=training
        )
        new_agent.policy_net.set_weights([np.copy(weight) for weight in self.policy_net.get_weights()])
        return new_agent

