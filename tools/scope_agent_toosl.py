from crewai import Agent, Task, Crew
from crewai_tools import CSVSearchTool
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from langchain.tools import tool
import pandas as pd
from crewai.tools import BaseTool
from typing import Optional, Dict, Any
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os
from utils.models import *

class ExtractSamplesTool(BaseTool):
    name: str = "extract_samples_tool"
    description: str = "Extract datetime-based samples from a CSV file"

    def _run(self, reference_datetime: str, datetime_column: str = 'date_time', num_samples: int = 5):
        try:
            csv = load_config('./config/config.yaml')
            df = pd.read_csv(csv["data_path"])

            if datetime_column not in df.columns:
                return {
                    "status": "error",
                    "message": f"Column '{datetime_column}' not found",
                    "available_columns": list(df.columns)
                }

            df[datetime_column] = pd.to_datetime(df[datetime_column])
            df = df.sort_values(by=datetime_column)

            ref_dt = pd.to_datetime(reference_datetime)
            mask = df[datetime_column] >= ref_dt

            if not mask.any():
                return {
                    "status": "warning",
                    "message": "Reference datetime too late",
                    "latest_record": str(df[datetime_column].max())
                }

            ref_index = mask.idxmax()
            result = df.iloc[max(0, ref_index - num_samples):ref_index]

            return {
                #"status": "success",
                "data": result.to_dict(orient='records'),
                #"metadata": {
                #    "file": csv_path,
                #    "time_range": {
                #        "start": str(result[datetime_column].min()),
                #        "end": str(result[datetime_column].max())
                #    }
                #}
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "solution": "Check knowledge source for file details"
            }
        
class MLModelTool(BaseTool):
    name: str = "ml_model_tool"
    description: str = "Make predictions using a trained ML model on time series data"

    def _run(self, time_series_data: list):
        conf = load_config('./config/config.yaml')
        model_path = conf["MLModel_path"]
        # Load the saved model
        model = tf.keras.models.load_model(model_path)
        data = pd.DataFrame(time_series_data)
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
       