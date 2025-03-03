"""Unit tests for landmark detection service."""

import unittest
import cv2
import numpy as np
import os
import sys
from pathlib import Path

from tests.test_config import TEST_IMAGES_DIR, TEST_IMAGE_URLS

# Import services to test
from app.services.landmark_detection import landmark_detector
from app.services.image_utils import image_utils


class TestLandmarkDetection(unittest.TestCase):
    """Test landmark detection functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - download test images if needed."""
        # Download test images if they don't exist
        import requests
        
        for i, url in enumerate(TEST_IMAGE_URLS[:2]):  # Use first 2 for unit tests
            image_path = os.path.join(TEST_IMAGES_DIR, f"test_image_{i}.jpg")
            if not os.path.exists(image_path):
                response = requests.get(url)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded test image to {image_path}")
    
    def setUp(self):
        """Set up test case."""
        # Initialize detector if not already initialized
        if not landmark_detector.initialized:
            landmark_detector.initialize()
        
        # Load the first test image
        self.test_image_path = os.path.join(TEST_IMAGES_DIR, "test_image_0.jpg")
        self.image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(self.image, "Failed to load test image")
    
    def test_initialization(self):
        """Test that landmark detector initializes correctly."""
        self.assertTrue(landmark_detector.initialized, "Landmark detector did not initialize")
        self.assertIsNotNone(landmark_detector.face_detector, "Face detector not initialized")
        self.assertIsNotNone(landmark_detector.shape_predictor, "Shape predictor not initialized")
    
    def test_face_detection(self):
        """Test that face detection works."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = landmark_detector.face_detector(gray)
        self.assertGreater(len(faces), 0, "No faces detected in test image")
    
    def test_landmark_detection(self):
        """Test that landmark detection works."""
        result = landmark_detector.detect_landmarks(self.image)
        self.assertTrue(result["success"], f"Landmark detection failed: {result['message']}")
        self.assertIsNotNone(result["landmarks"], "No landmarks returned")
        self.assertEqual(len(result["landmarks"]), 68, "Wrong number of landmarks detected")
    
    def test_drawing_landmarks(self):
        """Test that drawing landmarks works."""
        # Detect landmarks
        result = landmark_detector.detect_landmarks(self.image)
        self.assertTrue(result["success"], "Landmark detection failed")
        
        # Draw landmarks
        vis_img = landmark_detector.draw_landmarks(self.image, result["landmarks"])
        self.assertIsNotNone(vis_img, "Failed to generate visualization")
        self.assertEqual(vis_img.shape, self.image.shape, "Visualization has wrong shape")
        
        # Save visualization for manual inspection
        output_path = os.path.join(TEST_IMAGES_DIR, "test_landmarks_vis.jpg")
        cv2.imwrite(output_path, vis_img)


if __name__ == '__main__':
    unittest.main()