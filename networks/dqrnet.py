import tensorflow as tf
from tensorflow import Tensor


class DQRN(tf.keras.models.Model):
    def __init__(self,
                 hidden_size: int = 256,
                 output_dim: int = 40,
                 fc_size_post_lstm: int = 128,
                 fc_size_pre_lstm: int = 32,
                 fc_activation=tf.nn.leaky_relu,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.fc_activation = fc_activation

        self.rnn = tf.keras.layers.LSTM(
            hidden_size,
            return_state=True,
            return_sequences=True
        )

        self.fc1 = tf.keras.layers.Dense(fc_size_pre_lstm, activation=fc_activation)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.fc2 = tf.keras.layers.Dense(fc_size_post_lstm, activation=fc_activation)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.out = tf.keras.layers.Dense(output_dim, activation="linear")

    def call(self, x: Tensor, h: Tensor = None, c: Tensor = None):
        x = self.ln1(self.fc1(x))
        if c is not None and h is not None:
            x, *hidden = self.rnn(x, (h, c))
        else:
            x, *hidden = self.rnn(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = self.out(x)
        return x, hidden

    def init_hidden(
            self,
            batch_size: int,
    ):
        h = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
        c = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
        return h, c
