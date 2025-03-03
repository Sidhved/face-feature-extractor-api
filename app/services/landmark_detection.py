# ---------------------------------------------Landmark Detection Service-------------------------------------------- #

import cv2
import dlib
import numpy as np
import time
import traceback
import os

from app.core.config import settings
from app.core.logging import logger

class LandmarkDetector:
    """Facial landmark detection using dlib."""
    
    def __init__(self):
        """Initialize the dlib face detector and shape predictor."""
        self.face_detector = None
        self.shape_predictor = None
        self.initialized = False
        
    def initialize(self):
        """Initialize dlib models."""
        try:
            logger.info(f"Initializing dlib face detector and shape predictor")
            
            # Initialize face detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Check if shape predictor file exists
            predictor_path = settings.DLIB_SHAPE_PREDICTOR_PATH
            if not os.path.exists(predictor_path):
                logger.error(f"Shape predictor file not found: {predictor_path}")
                logger.info("You need to download the shape predictor file from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                return False
            
            # Initialize shape predictor
            self.shape_predictor = dlib.shape_predictor(predictor_path)
            
            self.initialized = True
            logger.info(f"Dlib models initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize dlib: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def detect_landmarks(self, image):
        """
        Detect facial landmarks in the input image.
        
        Args:
            image (numpy.ndarray): BGR image
            
        Returns:
            dict: Dictionary containing landmarks, face detection status, and processing time
        """
        result = {
            "success": False,
            "landmarks": None,
            "processing_time": 0,
            "message": ""
        }
        
        start_time = time.time()
        
        # Initialize models if not already initialized
        if not self.initialized:
            if not self.initialize():
                result["message"] = "Failed to initialize landmark detection models"
                return result
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector(gray, 1)
            
            if len(faces) == 0:
                result["message"] = "No face detected in the image"
                return result
            
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Predict facial landmarks
            shape = self.shape_predictor(gray, largest_face)
            
            # Convert landmarks to list of (x, y) coordinates
            landmarks = []
            for i in range(shape.num_parts):
                pt = shape.part(i)
                landmarks.append([pt.x, pt.y])
            
            result["success"] = True
            result["landmarks"] = landmarks
            result["message"] = "Face landmarks detected successfully"
            
        except Exception as e:
            logger.error(f"Error in landmark detection: {str(e)}")
            logger.error(traceback.format_exc())
            result["message"] = f"Error detecting landmarks: {str(e)}"
            
        finally:
            result["processing_time"] = time.time() - start_time
            
        return result

    def draw_landmarks(self, image, landmarks):
        """
        Draw landmarks on the input image.
        
        Args:
            image (numpy.ndarray): BGR image
            landmarks (list): List of (x, y) coordinates
            
        Returns:
            numpy.ndarray: Image with landmarks drawn
        """
        if landmarks is None:
            return image
        
        vis_img = image.copy()
        
        # Draw each landmark as a circle
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (int(x) + 2, int(y) - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
        # Draw connections between landmarks for better visualization
        # Jawline
        for i in range(0, 16):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Eyebrows
        for i in range(17, 21):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        for i in range(22, 26):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Nose
        for i in range(27, 30):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Nose bottom
        for i in range(30, 35):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Left eye
        for i in range(36, 41):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        # Close left eye
        pt1 = (int(landmarks[41][0]), int(landmarks[41][1]))
        pt2 = (int(landmarks[36][0]), int(landmarks[36][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Right eye
        for i in range(42, 47):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        # Close right eye
        pt1 = (int(landmarks[47][0]), int(landmarks[47][1]))
        pt2 = (int(landmarks[42][0]), int(landmarks[42][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Mouth outer
        for i in range(48, 59):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        # Close mouth outer
        pt1 = (int(landmarks[59][0]), int(landmarks[59][1]))
        pt2 = (int(landmarks[48][0]), int(landmarks[48][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Mouth inner
        for i in range(60, 67):
            pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
            pt2 = (int(landmarks[i+1][0]), int(landmarks[i+1][1]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        # Close mouth inner
        pt1 = (int(landmarks[67][0]), int(landmarks[67][1]))
        pt2 = (int(landmarks[60][0]), int(landmarks[60][1]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
            
        return vis_img

# Create singleton instance
landmark_detector = LandmarkDetector()

# ---------------------------------------------------------------------------------------------------- #