
import pandas as pd
#import seaborn as sns
import tensorflow as tf
import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from windowGenerator import *
from models import TrafficModels
from training import *
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # Enable deterministic ops
if __name__ == "__main__":
    

    # To store resuts:
    multi_val_performance = {}
    multi_performance = {}
    models = {}

    my_log = {
         'multi_val_performance':multi_val_performance,
         'multi_performance':multi_performance,
         'models':models
    }

    # 1. Load dataset
    df = pd.read_csv('./data/MITVolume_cleaned.csv', parse_dates=['date_time'], index_col='date_time')
    num_features = df.shape[1]

    # 2. Data Splitting and Resampling
    # Applying the resample method with the argument '1H' to  resample the data to have an hourly frequency.
    train_df = df[:-15000].resample('1h').mean().ffill()
    val_df = df[-15000:-5000].resample('1h').mean().ffill()
    test_df = df[-5000:].resample('1h').mean().ffill()

    # 3. Data Normalization
    scaler = MinMaxScaler().fit(train_df)
    s_train_df = pd.DataFrame(scaler.transform(train_df), index=train_df.index, columns=train_df.columns)
    s_val_df = pd.DataFrame(scaler.transform(val_df), index=val_df.index, columns=val_df.columns)
    s_test_df = pd.DataFrame(scaler.transform(test_df), index=test_df.index, columns=test_df.columns)
    scaled_tv = MinMaxScaler().fit(train_df[['traffic_volume']])

    # 4. Create window
    INPUT_WIDTH = 6
    OUT_STEPS = 1
    SHIFT = 1
    window = WindowGenerator(input_width=INPUT_WIDTH,
                               label_width=OUT_STEPS,
                               shift=SHIFT, 
                               train_df=s_train_df, val_df=s_val_df,
                               test_df= s_test_df,
                               label_columns=['traffic_volume'],
                               batch_size=32
                               )
   #base_window.plot()
   # 5. Train a model
    modelsObject = TrafficModels(input_width=INPUT_WIDTH,out_steps=OUT_STEPS)
    model_label = 'bidirectionalLSTM'
    model_BLSTM = modelsObject.bidirectional_lstm_model()
    # Compile the model. Result is stored in my_log variable
    history = compile_and_fit(model_BLSTM, window, val_df=val_df, scaled_tv=scaled_tv,
                          model_label=model_label, log_dict=my_log
                        )
    print(my_log)