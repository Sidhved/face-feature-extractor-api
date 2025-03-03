# ---------------------------------------------API Request Models-------------------------------------------- #

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional
from fastapi import UploadFile

class ImageUrlRequest(BaseModel):
    """Request model for image URL."""
    url: HttpUrl = Field(..., description="URL of the image to process")
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/image.jpg"
            }
        }

# ---------------------------------------------------------------------------------------------------- #