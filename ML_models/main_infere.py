import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from windowGenerator import *
import datetime

def load_and_preprocess_data(data_path='./data/forcast_data.csv'):
    df = pd.read_csv(data_path, parse_dates=['date_time'], index_col='date_time')
    """
    # Feature Engineering 
    df = pd.get_dummies(df_raw, columns=['weather_main'], prefix='weather')
    df = df.drop(columns=['weather_description'], inplace=False)
    df['is_holiday'] = df['holiday'].apply(lambda x: 1 if x != 'None' else 0)
    df['is_weekend'] = df.index.day_name().map(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    df = df.drop(columns=['holiday'], inplace=False)

    # Time features
    timestamp_s = df.index.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
     # Date and time components
    df['dayofweek'] = df.index.dayofweek
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_hour'] = df.index.hour

    # 4. Reorder columns
    column_order = ['traffic_volume', 'Day sin', 'Day cos', 'Year sin', 'Year cos', 'temp', 'clouds_all', 'rain_1h', 'snow_1h', 'is_weekend', 'is_holiday'] + [col for col in df.columns if col not in ['traffic_volume', 'Day sin', 'Day cos', 'Year sin', 'Year cos', 'temp', 'clouds_all', 'rain_1h', 'snow_1h', 'is_weekend', 'is_holiday']]
    df = df[column_order]
    """
    # Load the saved scaler 
   
    return df

def main_infer(model_path, data_path):

    """
    Extracts past samples from a CSV file for a given target sample.

    Args:
        data_path (str): Path to the CSV file that contains samples to test.
        target_sample (list): The target sample to forcast it trafic volume.
        input_width (int): Number of past samples to extract (default: 6).
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    data = load_and_preprocess_data(data_path)
    scaler = MinMaxScaler().fit(data) # or load from file
    # Assume my_scaler was fitted on the training data

    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    # Sort the DataFrame by 'date_time' in ascending order
    input_features = data.columns.tolist()
    data = data.sort_index()
   
    # Extract the past samples and the target sample
    #past_samples = data.loc[start_index:target_index[0]] 
    # Reshape the data
    
    input_data = np.array(data).reshape(1, 5, len(data.columns.tolist()))
     
    # Make the prediction()
    forecast = model.predict(input_data)
    # 3. Inverse transform
    dummy_array = np.zeros((1, len(input_features)))
    dummy_array[0, 0] = forecast[0,0]  # Insert prediction in 'volume' position
    rescaled = scaler.inverse_transform(dummy_array)

    final_forecast = rescaled[0, 0]  # Extract your rescaled volume prediction
    return final_forecast


if __name__ == "__main__":
    
    model_path ="./models/bidirectionalLSTM.h5"
    data_path= "./data/predictions.csv"
   
    f = main_infer(model_path, data_path)
     # Print the forecast
    print(f"Forecasted traffic volume: {f:.2f}")