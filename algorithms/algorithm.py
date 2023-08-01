from brimarl_masked.environment.environment import Agent
import numpy as np
import tensorflow as tf

class Algorithm:
    def store_game(self, states, actions, masks, rewards, dones):
        raise NotImplementedError()

    def learn(self, agent: Agent):
        raise NotImplementedError()

    def assert_same_shape(self, shape1, shape2):
        return assert_same_shape(shape1, shape2)

def assert_same_shape(shape1, shape2):
    def convert_to_numpy_shape(shape):
        if isinstance(shape, list):
            return np.array(shape)
        elif isinstance(shape, tf.Tensor):
            return convert_to_numpy_shape(shape.shape)
        elif isinstance(shape, np.ndarray):
            return convert_to_numpy_shape(shape.shape)
        elif isinstance(shape, tf.TensorShape):
            return np.array(shape.as_list())
        return shape

    shape1 = convert_to_numpy_shape(shape1)
    shape2 = convert_to_numpy_shape(shape2)

    assert np.array_equal(shape1, shape2), f"shape1: {shape1} - shape2: {shape2}"
