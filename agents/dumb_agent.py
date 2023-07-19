import random
from typing import List

import brimarl_masked.environment.environment as brisc
from brimarl_masked.environment.environment import BriscolaGame, BriscolaPlayer, Agent


class DumbAgent(brisc.Agent):
    """Agent selecting random available actions."""

    def __init__(self):
        super().__init__("DumbAgent")
        self.name = 'DumbAgent'

    def state(self, *_):
        return []

    def action(self, game: BriscolaGame, player: BriscolaPlayer):
        return 0, None, None

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
        return DumbAgent()
