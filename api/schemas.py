from pydantic import BaseModel, Field
from typing import Optional

class CaptionResponse(BaseModel):
    caption: str = Field(..., description="Generated image caption")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    style: str = Field(..., description="Caption style used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "caption": "a dog sitting on grass in a park",
                "confidence": 0.8542,
                "processing_time": 1.234,
                "style": "factual"
            }
        }

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
