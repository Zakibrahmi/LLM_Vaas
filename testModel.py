import requests
import yaml
from huggingface_hub import InferenceClient


# Load the YAML file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get the configurations
token = config["HUGGINGFACE"]["TOKEN"]
model = config["HUGGINGFACE"]["MODEL"]
timeout = config["HUGGINGFACE"]["TIMEOUT"]

# Initialize the client
client = InferenceClient(model=model, token=token, timeout=timeout)

# Define the prompt
prompt = "write a python function that prints hello"

# Make a request
try:
    response = client.text_generation(prompt=prompt)
    print("Response:", response)
except Exception as e:
    print(f"Error: {e}")