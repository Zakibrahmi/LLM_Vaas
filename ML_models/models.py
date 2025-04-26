import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.models import *

class TrafficModels:
    def __init__(self, input_width, out_steps, num_features=None):
        self.input_width = input_width
        self.out_steps = out_steps
        self.num_features = num_features
    
    def dense_model(self):
        """
        Creates a Dense model for traffic prediction.

        Returns:
            tf.keras.Model: The compiled Dense model.
        """
        model = tf.keras.Sequential([
            Lambda(lambda x: x[:, -1:, :]),
            Dense(512, activation='relu'),
            Dense(self.out_steps)
        ])
        return model

    def conv_model(self, conv_width=3):
        """
        Creates a Convolutional model for traffic prediction.

        Args:
            conv_width (int, optional): Width of the convolutional window. Defaults to 3.

        Returns:
            tf.keras.Model: The compiled Convolutional model.
        """
        model = tf.keras.Sequential([
            Lambda(lambda x: x[:, -conv_width:, :]),
            Conv1D(256, activation='relu', kernel_size=(conv_width)),
            Dense(512, activation='relu'),
            Dense(self.out_steps)
        ])
        return model

    def lstm_model(self):
        """
        Creates an LSTM model for traffic prediction.

        Returns:
            tf.keras.Model: The compiled LSTM model.
        """
        model = tf.keras.Sequential([
            LSTM(32, return_sequences=False),
            Dense(self.out_steps * self.num_features, kernel_initializer=tf.initializers.zeros()),
            Dense(self.out_steps)
        ])
        return model

    def bidirectional_lstm_model(self):
        """
        Creates a Bidirectional LSTM model for traffic prediction.

        Returns:
            tf.keras.Model: The compiled Bidirectional LSTM model.
        """
        forward_layer = LSTM(512, return_sequences=True)
        backward_layer = LSTM(512, return_sequences=True, go_backwards=True)
        model = tf.keras.Sequential([
            Bidirectional(forward_layer, backward_layer=backward_layer),
            Bidirectional(LSTM(512, return_sequences=False, stateful=False)),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(self.out_steps)
        ])
        return model

    # Assuming a baseline model is simply the average of the last 'input_width' values
    def baseline_model(self):  
        """
        Creates a simple baseline model (average of previous values).

        Returns:
            tf.keras.Model: The compiled baseline model.
        """
        model = tf.keras.Sequential([
            Lambda(lambda x: tf.reduce_mean(x[:, -self.input_width:, :], axis=1)),  # Average of last input_width values
            Dense(self.out_steps) 
        ])
        return model