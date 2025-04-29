import pandas as pd

def extract_samples_before_reference(csv_path, datetime_column, reference_datetime, num_samples):
    """
    Extract X samples immediately preceding a reference datetime from a CSV file.
    
    Parameters:
        csv_path (str): Path to the CSV file
        datetime_column (str): Name of the datetime column
        reference_datetime (str/datetime): The reference point ("YYYY-MM-DD HH:MM:SS" or datetime object)
        num_samples (int): Number of samples to extract before the reference
    
    Returns:
        pd.DataFrame: DataFrame containing the requested samples
        str: Status message (found/not found)
    """
    # Read and prepare data
    df = pd.read_csv(csv_path)
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.sort_values(by=datetime_column)
    
    # Convert reference to datetime if it's a string
    if isinstance(reference_datetime, str):
        reference_datetime = pd.to_datetime(reference_datetime)
    
    # Find the index of the first row >= reference datetime
    mask = df[datetime_column] >= reference_datetime
    if not mask.any():
        return None, "Reference datetime not found - it's later than all samples"
    
    ref_index = mask.idxmax()
    
    # Calculate start index for the X samples before
    start_index = max(0, ref_index - num_samples)
    
    # Extract and return the samples
    result = df.iloc[start_index:ref_index]
    return result, f"Found {len(result)} samples before {reference_datetime}"

# Example usage:
samples, message = extract_samples_before_reference(
     csv_path='./knowledge/MIT_Volume_test.csv',
     datetime_column='date_time',
     reference_datetime='2018-08-03 20:00:00',
     num_samples=3
 )
print(message)
print(samples)