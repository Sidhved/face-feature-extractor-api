# ---------------------------------------------Configuration Settings-------------------------------------------- #

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Facial Feature Extraction API"
    
    # Dlib settings
    DLIB_SHAPE_PREDICTOR_PATH: str = os.getenv("DLIB_SHAPE_PREDICTOR_PATH", "shape_predictor_68_face_landmarks.dat")
    
    # Image settings
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png"]
    TARGET_IMAGE_SIZE: tuple = (512, 512)  # Resize input images to this size

    RESULTS_STORAGE_PATH: str = os.getenv("RESULTS_STORAGE_PATH", "results")
    SAVE_RESULTS: bool = os.getenv("SAVE_RESULTS", "True").lower() == "true"
    SAVE_IMAGES: bool = os.getenv("SAVE_IMAGES", "False").lower() == "true"
    
    # API settings
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100"))  # requests per minute
    
    # Region of Interest settings
    # Facial landmark indices for T-zone (forehead, nose, chin)
    # Based on dlib's 68-point facial landmark model
    # Forehead points (extrapolated from eyebrows and top of face)
    # FOREHEAD_LANDMARKS: list = [
    #     # Creates a forehead region by extrapolating above the eyebrows
    #     [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    # ]

    # T-zone region - shaped like a T with forehead, nose and center chin
    T_ZONE_LANDMARKS: list = [
        # Forehead boundary (across top) - using eyebrows as base
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        # Nose central line
        [27, 28, 29, 30],
        # Nose bottom
        [30, 31, 33, 35],
        # Center chin area
        [8]
    ]

    # Left cheek region
    LEFT_CHEEK_LANDMARKS: list = [
        # Left face boundary
        [0, 1, 2, 3, 4, 5, 6, 7],
        # Left eye 
        [36, 37, 38, 39, 40, 41],
        # Left side mouth
        [48, 49, 50]
    ]

    # Right cheek region
    RIGHT_CHEEK_LANDMARKS: list = [
        # Right face boundary
        [16, 15, 14, 13, 12, 11, 10, 9],
        # Right eye
        [42, 43, 44, 45, 46, 47],
        # Right side mouth
        [52, 53, 54]
    ]

settings = Settings()

# ---------------------------------------------------------------------------------------------------- #