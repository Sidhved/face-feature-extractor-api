# ---------------------------------------------Image Utility Service-------------------------------------------- #

import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
import traceback
from typing import Optional

from app.core.config import settings
from app.core.logging import logger

class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def read_image_from_bytes(file_bytes) -> Optional[np.ndarray]:
        """
        Read image from bytes.
        
        Args:
            file_bytes (bytes): Image bytes
            
        Returns:
            numpy.ndarray: BGR image or None if failed
        """
        try:
            # Convert bytes to numpy array
            image = np.frombuffer(file_bytes, np.uint8)
            # Decode the image
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image bytes")
                return None
                
            return image
        except Exception as e:
            logger.error(f"Error reading image from bytes: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def read_image_from_url(url) -> Optional[np.ndarray]:
        """
        Read image from URL.
        
        Args:
            url (str): Image URL
            
        Returns:
            numpy.ndarray: BGR image or None if failed
        """
        try:
            # Download image from URL
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to numpy array
            image_bytes = response.content
            return ImageUtils.read_image_from_bytes(image_bytes)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching image from URL: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing image from URL: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def resize_image(image, target_size=None):
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image (numpy.ndarray): BGR image
            target_size (tuple): Target size (width, height)
            
        Returns:
            numpy.ndarray: Resized image
        """
        if target_size is None:
            target_size = settings.TARGET_IMAGE_SIZE
            
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        
        # Calculate new dimensions
        if w > h:
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
            
        # Ensure dimensions don't exceed target size
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    @staticmethod
    def convert_to_base64(image):
        """
        Convert image to base64 string.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            # Convert from BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            
            # Get base64 string
            base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return base64_string
        
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def check_image_quality(image):
        """
        Check if the image quality is good enough for landmark detection.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            tuple: (is_good, message) where is_good is a boolean and message is a string
        """
        # Check image dimensions
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False, "Image is too small, minimum dimensions are 100x100 pixels"
        
        # Check if image is not grayscale (converted to color)
        if len(image.shape) < 3 or image.shape[2] != 3:
            return False, "Image must be in color format"
        
        # Check for excessive blur using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:  # Threshold for blur detection
            return False, "Image is too blurry for accurate landmark detection"
        
        return True, "Image quality is good"

# Create singleton instance
image_utils = ImageUtils()

# ---------------------------------------------------------------------------------------------------- #