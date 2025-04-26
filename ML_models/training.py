
import pandas as pd
#import seaborn as sns
import tensorflow as tf
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MyReduceLROnPlateau import *

from  tensorflow.python.keras import layers, models, losses, regularizers, optimizers, callbacks

def compile_and_fit(model, window, val_df, scaled_tv, patience=5, max_epochs=30,  model_label='model', log_dict=None):
    #plot_metrics = TrainingPlot()

    rlr = MyReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.001, patience=patience, mode='min', verbose=1 )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True )

    model.compile( optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()] )

    history = model.fit( window.train, epochs=max_epochs, validation_data=window.val, callbacks=[early_stopping, rlr] )

    if log_dict:
        log_dict['models'][model_label] = model

        IPython.display.clear_output()

        print(f'Training vs. Validation:\n')
        #plot_train_validation(history, window.val)

        log_dict['multi_val_performance'][model_label] = model.evaluate(window.val, verbose=0)
        log_dict['multi_performance'][model_label] = model.evaluate(window.train, verbose=0)

    #Save the time serie model 
    model.save(f'./models/{model_label}.h5')
    
    predictions = model.predict(window.val)
    evaluate_predictions(val_df.traffic_volume, predictions, scaled_tv=scaled_tv)
    return history

def evaluate_predictions(y_true, y_pred, scaled_tv, plot_start_index=-500):
    print(f'\n\nPredictions Evaluation \n')
    y_p = np.asarray(scaled_tv.inverse_transform(y_pred.reshape(-1,1))).ravel()
    n_predictions = len(y_p)
    y = np.asarray(y_true[-n_predictions:]).ravel()
    print('Predictions:', n_predictions)

    # Use tf.keras.metrics.mean_absolute_error as a function instead
    mae = float(tf.keras.losses.mean_absolute_error(y, y_p))
    # Alternatively, call it directly as mae = float(tf.keras.losses.mean_absolute_error(y, y_p))

    mae_scaled = float(scaled_tv.transform(np.array([[mae]])))
    print(f'MAE: {mae:.2f} ({mae_scaled:.4f})')

    plt.subplots(figsize=(15,2))
    plt.plot(y[plot_start_index:], marker='.', label='true')
    plt.plot(y_p[plot_start_index:], marker='.', label='predicted')
    plt.legend()
    plt.show()