
from crewai import LLM
import yaml

def get_LLM_model(model_name="OpenAI")-> LLM:

    """Factory method to create any configured LLM from the config.yaml file"""

    config = load_config("./config/config.yaml")
    
    llm= LLM(model=config[model_name]["model"], api_key=config[model_name]["key"],base_url="https://api.openai.com/v1" ,temperature=config[model_name]["temperature"])
    return llm

def load_config(config_path: str):
        """Load YAML config with error handling"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Config loading failed: {str(e)}")

