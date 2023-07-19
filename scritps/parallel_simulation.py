from multiprocessing import Pool as PPool
from multiprocessing.dummy import Pool as TPool
from multiprocessing import cpu_count
from typing import List
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.environment.emulate import play_episode
from brimarl_masked.environment.environment import BriscolaGame, Agent
from itertools import repeat
from time import time
import os
import sys
import tensorflow as tf
import logging
class SuppressOutput:
    def __init__(self):
        self.devnull = open(os.devnull, 'w')
        self.saved_stdout = sys.stdout
        self.saved_tf_log_level = tf.get_logger().getEffectiveLevel()

    def __enter__(self):
        sys.stdout = self.devnull
        tf.get_logger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.saved_stdout
        tf.get_logger().setLevel(self.saved_tf_log_level)
        self.devnull.close()
#def mute():
#    sys.stdout = open(os.devnull, 'w')
#    sys.stderr = open(os.devnull, 'w')






def worker(params):
    num_players, win_extra_points, agents = params
    game = BriscolaGame(num_players, win_extra_points=win_extra_points)
    game.reset()
    return play_episode(game, agents, train=True)
class GamePool:
    def __init__(self, thread=True):
        if thread:
            self.pool = TPool(cpu_count())
        else:
            self.pool = PPool(cpu_count())

    def play(self, game: BriscolaGame, agents: List[List[Agent]]):
        with SuppressOutput(), tf.device('/CPU:0'):
            return self.pool.map(
                worker,
                zip(repeat(game.num_players), repeat(game.win_extra_points), agents))

if __name__ == "__main__":
    agents = [
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
        [ACAgent(),ACAgent(),ACAgent(),ACAgent()],
    ]
    game = BriscolaGame(4, win_extra_points=0)
    pool = GamePool()
    t1 = time()
    pool.play(game, agents)
    t2 = time()
    t = time()
    pool.play(game, agents)
    print(t2-t1)
    print(time() - t)
    pool.pool.close()

