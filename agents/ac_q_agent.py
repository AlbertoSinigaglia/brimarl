import sys
from typing import List

import numpy as np
from brimarl_masked.networks.ac_q_nets import Actor, QCritic
import tensorflow as tf
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class ACQAgent(Agent):

    def __init__(self, training=True, name="ACQAgent",) -> None:
        """"""
        super().__init__(name)
        self.name = name

        self.policy_net = Actor()
        self.policy_net.build((None, 80))
        self.q_net = QCritic()
        self.q_net.build((None, 80))

        self.training = training
        self.current_game_state = np.zeros(40)

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
        turn_state = np.zeros(40)
        value_offset = 4
        seed_offset = 2
        features_per_card = 6
        turn_state[0] = player.id == current_player.id
        turn_state[1] = player.id == current_player.id or env.get_teammate(player.id) == current_player.id
        turn_state[2] = player.points / 60
        turn_state[3] = env.counter / 10

        for i, card in enumerate(player.hand):
            number_index = i * features_per_card + value_offset
            seed_index = i * features_per_card + value_offset + seed_offset + card.seed
            turn_state[number_index] = card.number
            turn_state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            turn_state[seed_index] = 1

        for i, card in enumerate(env.played_cards):
            number_index = (i + 3) * features_per_card + value_offset
            seed_index = (i + 3) * features_per_card + seed_offset + card.seed + value_offset
            turn_state[number_index] = card.number
            turn_state[number_index + 1] = 1 if card.seed == env.briscola.seed else 0
            turn_state[seed_index] = 1
            self.current_game_state[card.id] = 1
        return np.concatenate((turn_state, self.current_game_state))

    def action(self, game: BriscolaGame, player: BriscolaPlayer):
        assert len(player.hand)
        mask = np.zeros(40)
        for card in player.hand: mask[card.id] = 1
        s = self.state(game, player, player)[None, ...]
        original_probs = self.policy_net(s)
        probs = original_probs[0]
        probs *= mask
        probs = probs / tf.reduce_sum(probs)
        action = tf.random.categorical(
            tf.math.log(probs[None, ...]), 1
        )[0, 0]
        # if not self.training:
        #    action = np.argmax(self.q_net(s)[0] - 1e+20 * (1-mask))
        game_action = np.argwhere([c.id == action for c in player.hand]).squeeze()
        assert player.hand[game_action].id == action
        action = np.zeros(40)
        action[player.hand[game_action].id] = 1
        return game_action, action, mask

    def reset(self):
        self.current_game_state = np.zeros_like(self.current_game_state)

    def save_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.save_weights(path + self.name + "/policy")
        self.q_net.save_weights(path + self.name + "/qnet")

    def load_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.load_weights(path + self.name + "/policy")
        self.q_net.load_weights(path + self.name + "/qnet")

    def clone(self, training=False):
        assert type(self) is ACQAgent
        new_agent = ACQAgent(
            training=training
        )
        new_agent.policy_net.set_weights([np.copy(weight) for weight in self.policy_net.get_weights()])
        new_agent.q_net.set_weights([np.copy(weight) for weight in self.q_net.get_weights()])
        return new_agent



