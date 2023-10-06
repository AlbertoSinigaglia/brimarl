import tensorflow as tf
from tensorflow import Tensor


class ActorReg(tf.keras.models.Model):
    def __init__(self,
                 output_dim: int = 40,
                 fc_activation=tf.nn.tanh,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
    def call(self, x: Tensor):
        return self.model(x)

class CriticReg(tf.keras.models.Model):
    def __init__(self,
                 fc_activation=tf.nn.tanh,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation="linear"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation=fc_activation),
            tf.keras.layers.Dense(1, activation="linear")
        ])
    def call(self, x: Tensor):
        return self.model(x)

