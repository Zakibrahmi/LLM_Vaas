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
os.environ["OPENAI_API_KEY"] = "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"
# Initialize the CSV tool

# 1. Define the knowledge source (proper CSV integration)
file_paths='MITVolume_cleaned.csv',  # List of files
    
# 2. Create the extraction tool that leverages knowledge

class ExtractSamplesTool(BaseTool):
    name: str = "extract_samples_tool"
    description: str = "Extract datetime-based samples from a CSV file"

    def _run(self, reference_datetime: str, datetime_column: str = 'date_time', num_samples: int = 5):
        try:
            csv_path = 'MITVolume_cleaned.csv'
            df = pd.read_csv(csv_path)

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
        model_path = "bidirectionalLSTM.h5"
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
       
      
extract_samples_tool = ExtractSamplesTool()
ml_model_tool = MLModelTool()

# 3. Create agent with BOTH tool and knowledge
data_analyst = Agent(
    role="Time Series Analyst",
    goal="Extract precise datetime-based samples",
    backstory="Specializes in temporal data analysis with CSV expertise",
    tools=[extract_samples_tool, ml_model_tool],
    verbose=True
)

# 4. Create task that leverages knowledge
analysis_task = Task(
    description="""
    Analyze our time series data by extracting {num_samples} records 
    immediately before {target_datetime}. Use the '{column}' column for timestamps.
    """,
    expected_output="Structured results with only data formatted as Dataframe format",
    agent=data_analyst,
    inputs={
        "datetime_column": "{column}",  # Dynamic column selection
        "reference_datetime": "{target_datetime}",
        "num_samples": "{num_samples}"
    }
)
# Create the new ML prediction task
ml_prediction_task = Task(
    description="""
    Take the time series data from the previous analysis formatted as DataFrame. 
    The input will be the output from the analysis_task.
    """,
    expected_output="forcated trafic volume",
    agent=data_analyst,
    async_execution=False,  # Set to True if you want parallel execution
    #output_file="predictions.csv",  # Optional: save predictions to file
    context=[analysis_task]  # This task depends on analysis_task
)
# 5. Execute with all parameters
crew = Crew(agents=[data_analyst], tasks=[analysis_task, ml_prediction_task])
result = crew.kickoff(inputs={
    "target_datetime": "2018-08-03 20:00:00",
    "num_samples": 5,
    "column": "date_time"  # Can be changed to other timestamp columns
})

print("Analysis Results:")
print(result)