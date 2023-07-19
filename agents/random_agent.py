import random
from typing import List

import brimarl_masked.environment.environment as brisc
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class RandomAgent(brisc.Agent):
    """Agent selecting random available actions."""

    def __init__(self):
        super().__init__("RandomAgent")
        self.name = 'RandomAgent'

    def state(self, *_):
        pass

    def action(self, game: BriscolaGame, player: BriscolaPlayer):
        return random.choice(list(range(len(player.hand)))), None, None

    def update(self, reward):
        pass

    def make_greedy(self):
        pass

    def restore_epsilon(self):
        pass

    def reset(self):
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass

    def clone(self, training=False):
        pass
