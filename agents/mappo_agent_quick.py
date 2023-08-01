import sys
from typing import List

import numpy as np
from brimarl_masked.networks.acnets import Actor, Critic
import tensorflow as tf
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class MappoAgentQuick(Agent):

    def __init__(self, training=True, name="MappoAgentQuick") -> None:
        """"""
        super().__init__(name)
        self.name = name

        self.training = training
        self.current_game_state = np.zeros((40, 2))

    @property
    def policy_net(self):
        if getattr(self, '_POLICY_NET', None) is None:
            self._POLICY_NET = Actor()
            self._POLICY_NET.build((None, 244))  # 80))
        return self._POLICY_NET

    @property
    def value_net(self):
        if getattr(self, '_VALUE_NET', None) is None:
            self._VALUE_NET = Critic()
            self._VALUE_NET.build((None, 244*2))  # 80))
        return self._VALUE_NET

    @policy_net.setter
    def policy_net(self, policy: tf.keras.Model):
        self._POLICY_NET = policy
        if not policy.built:
            self._POLICY_NET.build((None, 244))  # 80))

    @value_net.setter
    def value_net(self, value):
        self._VALUE_NET = value
        if not value.built:
            self._VALUE_NET.build((None, 244))  # 80))

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
        return self.new_way(env, player, current_player)

    def new_way(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
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

    def old_way(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
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
        original_probs = self.policy_net(s)[0]
        masked_probs = original_probs * mask
        probs = masked_probs / tf.reduce_sum(masked_probs)
        action = tf.random.categorical(
            tf.math.log(probs[None, ...]), 1
        )[0, 0]
        # if not self.training:
        #     action = np.argmax(probs)
        try:
            game_action = np.argwhere([c.id == action for c in player.hand]).squeeze()
            assert player.hand[game_action].id == action
            action = np.zeros(40)
            action[player.hand[game_action].id] = 1
        except Exception as e:
            print(e)
            print(action)
            print(original_probs)
            print(masked_probs)
            print(tf.reduce_sum(masked_probs))
            print(probs)
            print([c.id == action for c in player.hand])
            print(np.argwhere([c.id == action for c in player.hand]).squeeze())
            print([c.id for c in player.hand])
            exit(99)
            raise Exception()
        return game_action, action, mask

    def reset(self):
        self.current_game_state = np.zeros_like(self.current_game_state)

    def save_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.save_weights(path + self.name + "/policy")
        self.value_net.save_weights(path + self.name + "/value")

    def load_model(self, path: str):
        if not path.endswith("/"):
            path += "/"
        self.policy_net.load_weights(path + self.name + "/policy")
        self.value_net.load_weights(path + self.name + "/value")

    def clone(self, training=False, deep=False):
        assert type(self) is MappoAgentQuick
        new_agent = MappoAgentQuick(
            training=training
        )
        if deep:
            new_agent.policy_net.set_weights([np.copy(weight) for weight in self.policy_net.get_weights()])
            new_agent.value_net.set_weights([np.copy(weight) for weight in self.value_net.get_weights()])
        else:
            new_agent.policy_net = self.policy_net
            new_agent.value_net = self.value_net
        return new_agent



