from pydantic import BaseModel
from typing import Dict, Optional, List, Set, Tuple, Union

# Define the Pydantic model for the query
class output_query(BaseModel):
    origin: str
    destination: str
    departure_time:str
    constrains: List[str]