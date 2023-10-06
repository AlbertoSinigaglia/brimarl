import sys
from typing import List

import numpy as np
from brimarl_masked.networks.acnets import Actor, Critic
import tensorflow as tf
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class OracleAgentQuick(Agent):
    ID = 0

    def __init__(self, training=True, name="OracleAgentQuick") -> None:
        """"""
        super().__init__(name)
        self.name = name
        self.id = OracleAgentQuick.ID
        OracleAgentQuick.ID += 1

        self.training = training
        self.current_game_state = np.zeros((40, 2))

    @property
    def policy_net(self):
        if getattr(self, '_POLICY_NET', None) is None:
            self._POLICY_NET = Actor()
            #self._POLICY_NET.build((None, 364))  # 80))
        return self._POLICY_NET

    @property
    def value_net(self):
        if getattr(self, '_VALUE_NET', None) is None:
            self._VALUE_NET = Critic()
            #self._VALUE_NET.build((None, 364))  # 80))
        return self._VALUE_NET

    @policy_net.setter
    def policy_net(self, policy: tf.keras.Model):
        self._POLICY_NET = policy
        # if not policy.built:
            #self._POLICY_NET.build((None, 364))  # 80))

    @value_net.setter
    def value_net(self, value):
        self._VALUE_NET = value
        # if not value.built:
            #self._VALUE_NET.build((None, 364))  # 80))

    def state(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
        return self.new_way(env, player, current_player)

    @staticmethod
    def encode_hand(player: BriscolaPlayer, env: BriscolaGame):
        player_hand_cards = np.zeros((40,2))
        for i, card in enumerate(player.hand):
            player_hand_cards[card.id, 0] = 1
            player_hand_cards[card.id, 1] = 1 if card.seed == env.briscola.seed else 0
        return player_hand_cards

    def new_way(self, env: BriscolaGame, player: BriscolaPlayer, current_player: BriscolaPlayer):
        turn_state = np.zeros(4)
        turn_state[0] = player.id == current_player.id
        turn_state[1] = player.id == current_player.id or env.get_teammate(player.id) == current_player.id
        turn_state[2] = player.points / 60
        turn_state[3] = env.counter / 10

        played_cards = np.zeros((40,3))
        initial_card = np.zeros((40,))
        if len(env.played_cards):
            initial_card[env.played_cards[0].id] = 1
        # if "my" team started -> our card - enemy's card - our card - ...
        our_turn = 1 if env.players_order[0] in [player.id, env.get_teammate(player.id)] else 0
        for i, card in enumerate(env.played_cards):
            played_cards[card.id,0] = 1
            played_cards[card.id, 1] = 1 if card.seed == env.briscola.seed else 0
            # if we started the turn, the odds position cards are our
            played_cards[card.id, 2] = (i + our_turn) % 2
            self.current_game_state[card.id, 0] = 1
            self.current_game_state[card.id, 1] = 1 if card.seed == env.briscola.seed else 0

        """played_cards = np.zeros((3, 40))
        for i, card in enumerate(env.played_cards):
            if i < 3:
                played_cards[i, card.id] = 1
            self.current_game_state[card.id, 0] = 1
            self.current_game_state[card.id, 1] = 1 if card.seed == env.briscola.seed else 0"""

        player_hand_cards = self.encode_hand(player, env)
        teammate_hand_cards = self.encode_hand(env.players[env.get_teammate(player.id)], env)
        en1 = self.encode_hand(env.players[(player.id + 1) % 4], env)
        en2 = self.encode_hand(env.players[(env.get_teammate(player.id)+1) % 4], env)

        return np.concatenate((
            turn_state.reshape((-1)),
            player_hand_cards.reshape((-1)),
            teammate_hand_cards.reshape((-1)),
            played_cards.reshape((-1)),
            initial_card.reshape((-1)),
            self.current_game_state.reshape((-1)),
            en1.reshape((-1)),
            en2.reshape((-1))
        ))

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
        return
        if not path.endswith("/"):
            path += "/"
        self.policy_net.save_weights(path + self.name + "/policy")
        self.value_net.save_weights(path + self.name + "/value")

    def load_model(self, path: str):
        return
        if not path.endswith("/"):
            path += "/"
        self.policy_net.load_weights(path + self.name + "/policy")
        self.value_net.load_weights(path + self.name + "/value")

    def clone(self, training=False, deep=False):
        assert type(self) is OracleAgentQuick
        new_agent = OracleAgentQuick(
            training=training
        )
        if deep:
            new_agent.policy_net.set_weights([np.copy(weight) for weight in self.policy_net.get_weights()])
            new_agent.value_net.set_weights([np.copy(weight) for weight in self.value_net.get_weights()])
        else:
            new_agent.policy_net = self.policy_net
            new_agent.value_net = self.value_net
        return new_agent



