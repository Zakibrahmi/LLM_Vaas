from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict
import json
import re

# --------- QueryAnalysisOutput ---------
class QueryAnalysisOutput(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_time: Optional[str] = None
    constraints: Optional[List[dict]] = []

    @field_validator('constraints', mode='before')
    def validate_constraints(cls, value):
        """Ensure constraints is always a list"""
        if value is None:
            return []
        if isinstance(value, dict):  # Handle single constraint object
            return [value]
        return value

    @classmethod
    def parse_raw_json(cls, json_str: str):
        """Robust JSON sanitization and parsing"""
        try:
            # Remove all markdown code blocks
            cleaned = re.sub(r'```json|```', '', json_str).strip()
            
            # Fix common JSON errors
            cleaned = (
                cleaned
                .replace("'", '"')          # Replace single quotes
                .replace("None", "null")    # Python None -> JSON null
                .replace("True", "true")    # Python True -> JSON true
                .replace("False", "false")  # Python False -> JSON false
            )

            # Remove trailing commas
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            # Fix missing closing brackets
            open_braces = cleaned.count('[') - cleaned.count(']')
            open_curlies = cleaned.count('{') - cleaned.count('}')
            
            if open_braces > 0:
                cleaned += ']' * open_braces
            if open_curlies > 0:
                cleaned += '}' * open_curlies

            return cls(**json.loads(cleaned))
        except json.JSONDecodeError as e:
            error_msg = f"JSON Error: {str(e)}\nProblematic JSON:\n{cleaned}"
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Validation Error: {str(e)}") from e

class UserFeedbackOutput(BaseModel):
    missing_or_unrealistic_fields: List[str]
    messages: List[str]
    responses: Dict[str, str]

    @classmethod
    def parse_raw_json(cls, json_str: str):
        """Sanitize and parse raw JSON from LLM"""
        cleaned = re.sub(r'```json|```', '', json_str).strip()
        cleaned = (
            cleaned
            .replace("'", '"')
            .replace("None", "null")
            .replace("True", "true")
            .replace("False", "false")
        )

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nCleaned content:\n{cleaned}")

        return cls(**data)

class FinalConstraint(BaseModel):
    type: str
    value: str

    @field_validator("value")
    def no_null_or_unrealistic(cls, v):
        if v in ("null", "unrealistic"):
            raise ValueError("Final constraint value must not be 'null' or 'unrealistic'")
        return v.strip()

class FinalQueryOutput(BaseModel):
    origin: str
    destination: str
    departure_time: str
    constraints: List[FinalConstraint]

    @field_validator("origin", "destination", "departure_time")
    def must_be_valid_string(cls, v):
        if v in ("null", "unrealistic"):
            raise ValueError("This field must not be 'null' or 'unrealistic'")
        return v.strip()
