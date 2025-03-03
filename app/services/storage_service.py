# ---------------------------------------------Storage Service-------------------------------------------- #

import os
import json
from pathlib import Path
import uuid
import datetime
import traceback

from app.core.config import settings
from app.core.logging import logger

class StorageService:
    """Service for storing API results to disk."""
    
    def __init__(self):
        """Initialize storage service."""
        self.results_dir = Path(settings.RESULTS_STORAGE_PATH)
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure that storage directories exist."""
        try:
            # Create main results directory
            self.results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create subdirectories for different types of data
            (self.results_dir / "json").mkdir(exist_ok=True)
            (self.results_dir / "images").mkdir(exist_ok=True)
            
            logger.info(f"Storage directories created at {self.results_dir}")
        except Exception as e:
            logger.error(f"Failed to create storage directories: {str(e)}")
            logger.error(traceback.format_exc())
    
    def save_result(self, result_data, image_id=None):
        """
        Save API result to JSON file, with masked image saved separately.
        
        Args:
            result_data (dict): API response data
            image_id (str, optional): Unique ID for the image. If None, uses the ID from result_data
            
        Returns:
            tuple: (success, file_path, image_path)
        """
        try:
            # Get image ID from result data if not provided
            if image_id is None and "image_info" in result_data and result_data["image_info"]:
                image_id = result_data["image_info"].get("image_id", str(uuid.uuid4()))
            elif image_id is None:
                image_id = str(uuid.uuid4())
            
            # Create timestamp for filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Extract masked image if present, and save separately
            masked_image_path = None
            result_copy = result_data.copy()
            
            if "masked_image" in result_copy and result_copy["masked_image"]:
                # Save masked image as separate file
                try:
                    # Decode base64 image
                    import base64
                    from PIL import Image
                    import io
                    
                    img_data = base64.b64decode(result_copy["masked_image"])
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Create filename for image
                    img_filename = f"{image_id}_{timestamp}_masked.jpg"
                    img_path = self.results_dir / "images" / img_filename
                    
                    # Save image
                    img.save(str(img_path))
                    masked_image_path = str(img_path)
                    
                    # Replace base64 in JSON with file path reference
                    result_copy["masked_image_path"] = masked_image_path
                    del result_copy["masked_image"]  # Remove base64 data to keep JSON small
                    
                    logger.info(f"Saved masked image to {masked_image_path}")
                except Exception as e:
                    logger.error(f"Failed to save masked image: {str(e)}")
                    # Keep the base64 data if saving failed
            
            # Create filename for JSON
            json_filename = f"{image_id}_{timestamp}.json"
            json_path = self.results_dir / "json" / json_filename
            
            # Save JSON data
            with open(json_path, 'w') as f:
                json.dump(result_copy, f, indent=2)
            
            logger.info(f"Saved result to {json_path}")
            return True, str(json_path), masked_image_path
        
        except Exception as e:
            logger.error(f"Failed to save result: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None, None
        
    def save_image(self, image, image_id=None):
        """
        Save image to disk.
        
        Args:
            image (numpy.ndarray): BGR image
            image_id (str, optional): Unique ID for the image. If None, generates a new UUID
            
        Returns:
            tuple: (success, file_path)
        """
        import cv2
        
        try:
            # Generate image ID if not provided
            if image_id is None:
                image_id = str(uuid.uuid4())
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{image_id}_{timestamp}.jpg"
            
            # Create full file path
            file_path = self.results_dir / "images" / filename
            
            # Save image
            cv2.imwrite(str(file_path), image)
            
            logger.info(f"Saved image to {file_path}")
            return True, str(file_path)
        
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None

# Create singleton instance
storage_service = StorageService()

# ---------------------------------------------------------------------------------------------------- #