from brimarl_masked.algorithms.replay_memory import ReplayMemory
import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class GAEAlgorithm(Algorithm):
    def __init__(self, discount, lambda_coefficient=0.9):
        self.discount = discount
        self.lambda_coefficient = lambda_coefficient

    def gae(self, advantages):
        t = tf.range(advantages.shape[1], dtype=tf.float32)
        kernel = (self.discount * self.lambda_coefficient)**t
        kernel = kernel[:, None, None]
        advantages = tf.pad(advantages, [[0,0], [0, advantages.shape[1]-1], [0,0]])
        res = tf.nn.conv1d(advantages, kernel, 1, padding="VALID")
        return res

    def test(self):
        self.discount = 1
        print(self.gae(tf.convert_to_tensor([
            [3,2,1],
            [5,6,7],
        ], dtype=tf.float32)[..., None]))


# GAEAlgorithm(1,1).test()

