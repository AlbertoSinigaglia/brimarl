import tensorflow as tf
from tensorflow import Tensor


class DQN(tf.keras.models.Model):
    def __init__(self,
                 output_dim: int = 40,
                 fc_activation=tf.nn.leaky_relu,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.fc_activation = fc_activation

        self.fc1 = tf.keras.layers.Dense(256, activation=fc_activation)
        self.fc2 = tf.keras.layers.Dense(256, activation=fc_activation)
        self.out = tf.keras.layers.Dense(output_dim, activation="linear",
                                         kernel_initializer=tf.initializers.RandomNormal(),
                                         bias_initializer=tf.initializers.RandomNormal())

    def call(self, x: Tensor):
        return self.out(self.fc2(self.fc1(x)))
