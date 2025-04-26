
#Instead of using ReduceLROnPlateau directly, 
# you can subclass it and override the on_epoch_end method to use the updated learning rate access
import tensorflow as tf
from keras import backend

class MyReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)  
        super().on_epoch_end(epoch, logs)
