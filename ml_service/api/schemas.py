from pydantic import BaseModel
from datetime import datetime


class EventOut(BaseModel):
    timestamp: datetime
    label: str
    confidence: float
    latitude: float
    longitude: float
