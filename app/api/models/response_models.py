# ---------------------------------------------API Response Models-------------------------------------------- #

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ImageInfo(BaseModel):
    """Information about the processed image."""
    image_id: str = Field(..., description="Unique identifier for the image")
    filename: str = Field(..., description="Original filename")
    dimensions: Dict[str, int] = Field(..., description="Image dimensions (width, height)")
    format: str = Field(..., description="Image format (JPEG, PNG, etc.)")

class BoundingBox(BaseModel):
    """Bounding box for a facial region."""
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")

class Region(BaseModel):
    """Region of interest with points and bounding box."""
    points: List[List[float]] = Field(..., description="Points defining the region boundary")
    bounding_box: BoundingBox = Field(..., description="Bounding box for the region")

class FacialFeatureResponse(BaseModel):
    """Response model for facial feature extraction."""
    status: str = Field(..., description="Status of the request (success/error)")
    message: str = Field(..., description="Status message or error description")
    image_info: Optional[ImageInfo] = Field(None, description="Information about the processed image")
    face_detected: bool = Field(..., description="Whether a face was detected in the image")
    landmarks: Optional[List[List[float]]] = Field(None, description="Facial landmarks as (x, y) coordinates")
    regions_of_interest: Optional[Dict[str, Region]] = Field(None, description="Extracted regions of interest")
    masked_image: Optional[str] = Field(None, description="Base64 encoded image with landmarks and regions")
    processing_time: float = Field(..., description="Processing time in seconds")
    result_file: Optional[str] = Field(None, description="Path to the saved result JSON file")
    image_file: Optional[str] = Field(None, description="Path to the saved processed image file")
    masked_image_file: Optional[str] = Field(None, description="Path to the saved masked image file")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Image processed successfully",
                "image_info": {
                    "image_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "filename": "example.jpg",
                    "dimensions": {"width": 640, "height": 480},
                    "format": "JPEG"
                },
                "face_detected": True,
                "landmarks": [[100.5, 120.3], [110.2, 130.7], "..."],
                "regions_of_interest": {
                    "t_zone": {
                        "points": [[120.5, 80.3], [150.2, 100.7], "..."],
                        "bounding_box": {"x": 120, "y": 80, "width": 100, "height": 120}
                    },
                    "left_cheek": {
                        "points": [[80.5, 150.3], [100.2, 170.7], "..."],
                        "bounding_box": {"x": 80, "y": 150, "width": 60, "height": 70}
                    },
                    "right_cheek": {
                        "points": [[180.5, 150.3], [200.2, 170.7], "..."],
                        "bounding_box": {"x": 180, "y": 150, "width": 60, "height": 70}
                    }
                },
                "masked_image": "base64_encoded_string",
                "processing_time": 1.254
            }
        }

# ---------------------------------------------------------------------------------------------------- #