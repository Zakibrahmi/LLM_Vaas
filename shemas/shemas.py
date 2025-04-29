
from pydantic import BaseModel
from typing import Optional, List

class QueryAnalysisOutput(BaseModel):
    origin: Optional[str]
    destination: Optional[str]
    departure_time: Optional[str]
    constraints: Optional[List[dict]]