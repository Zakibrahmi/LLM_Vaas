import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
#for warning
from warnings import filterwarnings
filterwarnings("ignore")

def clean_and_prepare_data(data_path, output_file=None):
    """
    Cleans and prepares the Metro Interstate Traffic Volume data for time series forecasting.

    Args:
        data_path (str): Path or URL to the raw data file.

    Returns:
        tuple: A tuple containing the scaled training, validation, and testing DataFrames.
    """

    # 1. Load the raw data
    df_raw = pd.read_csv(data_path, parse_dates=['date_time'], index_col='date_time')

    # 2. Data Cleaning
    # Temperature
    df_raw = df_raw.reset_index()
    df_raw = df_raw.set_index('date_time')
    df_raw.loc[df_raw['temp'] == 0, 'temp'] = np.nan
    day1_mean = df_raw.loc['2014-01-31'].temp.mean()
    day2_mean = df_raw.loc['2014-02-02'].temp.mean()
    df_raw.loc[df_raw['temp'].isnull() & (df_raw.index.date == datetime.date(2014, 1, 31)), 'temp'] = day1_mean
    df_raw.loc[df_raw['temp'].isnull() & (df_raw.index.date == datetime.date(2014, 2, 2)), 'temp'] = day2_mean

    # Rain
    df_raw.loc['2016-07-11 17:00:00', 'rain_1h'] = np.nan
    df_raw.loc['2016-07-11 17:00:00', 'rain_1h'] = df_raw['rain_1h'].mean()

    # 3. Feature Engineering
     #. 1. We transform the weather_main into one-hot encoded variables, and to drop the weather_description.
     # 2. We will create a new feature is_holiday and drop the old feature holiday. We don't need to keep track which weekend it is. 
     #  So, we will create a new feature is_weekend.

    df = pd.get_dummies(df_raw, columns=['weather_main'], prefix='weather')
    df = df.drop(columns=['weather_description'], inplace=False)
    df['is_holiday'] = df['holiday'].apply(lambda x: 1 if x != 'None' else 0)
    df['is_weekend'] = df.index.day_name().map(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    df = df.drop(columns=['holiday'], inplace=False)

    # Time features
    timestamp_s = df.index.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    # The traffic flow is seasonal and periodic based on daily and yearly periodicity.
    # Hence, to convert it to a usable signal is to use sin and cos to convert the time to clear "Time of day" and "Time of year" signals:
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

    # Save the cleaned data
    if output_file:
        df.to_csv(output_file, index=True)
      
    return df

if __name__ == "__main__":


    output_file = "./data/MITVolume_cleaned.csv"
    data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    clean_and_prepare_data(data_path=data_path, output_file=output_file)