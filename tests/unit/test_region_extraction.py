"""Unit tests for region extraction service."""

import unittest
import cv2
import numpy as np
import os
import sys
from pathlib import Path

from tests.test_config import TEST_IMAGES_DIR

# Import services to test
from app.services.landmark_detection import landmark_detector
from app.services.region_extraction import region_extractor


class TestRegionExtraction(unittest.TestCase):
    """Test region extraction functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Initialize detector if not already initialized
        if not landmark_detector.initialized:
            landmark_detector.initialize()
        
        # Load test image
        self.test_image_path = os.path.join(TEST_IMAGES_DIR, "test_image_0.jpg")
        self.image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(self.image, "Failed to load test image")
        
        # Detect landmarks
        result = landmark_detector.detect_landmarks(self.image)
        self.assertTrue(result["success"], "Landmark detection failed")
        self.landmarks = result["landmarks"]
    
    def test_region_extraction(self):
        """Test that region extraction works."""
        result = region_extractor.extract_regions(self.image, self.landmarks)
        self.assertTrue(result["success"], f"Region extraction failed: {result['message']}")
        
        # Check that all regions exist
        self.assertIn("t_zone", result["regions"], "T-zone region missing")
        self.assertIn("left_cheek", result["regions"], "Left cheek region missing")
        self.assertIn("right_cheek", result["regions"], "Right cheek region missing")
        
        # Check that all regions have points and bounding boxes
        for region_name in ["t_zone", "left_cheek", "right_cheek"]:
            region = result["regions"][region_name]
            self.assertTrue(len(region["points"]) > 0, f"{region_name} has no points")
            self.assertIsNotNone(region["bounding_box"], f"{region_name} has no bounding box")
        
        # Check that all region masks exist
        self.assertIn("t_zone", result["region_masks"], "T-zone mask missing")
        self.assertIn("left_cheek", result["region_masks"], "Left cheek mask missing")
        self.assertIn("right_cheek", result["region_masks"], "Right cheek mask missing")
        
        # Save region visualization for manual inspection
        vis_img = region_extractor.overlay_regions(self.image, result["region_masks"])
        output_path = os.path.join(TEST_IMAGES_DIR, "test_regions_vis.jpg")
        cv2.imwrite(output_path, vis_img)
    
    def test_non_overlapping_regions(self):
        """Test that regions don't overlap excessively."""
        result = region_extractor.extract_regions(self.image, self.landmarks)
        self.assertTrue(result["success"], "Region extraction failed")
        
        # Check T-zone and left cheek don't overlap
        t_zone_mask = result["region_masks"]["t_zone"]
        left_cheek_mask = result["region_masks"]["left_cheek"]
        overlap = cv2.bitwise_and(t_zone_mask, left_cheek_mask)
        overlap_pixels = cv2.countNonZero(overlap)
        self.assertLess(overlap_pixels, 100, "T-zone and left cheek overlap too much")
        
        # Check T-zone and right cheek don't overlap
        right_cheek_mask = result["region_masks"]["right_cheek"]
        overlap = cv2.bitwise_and(t_zone_mask, right_cheek_mask)
        overlap_pixels = cv2.countNonZero(overlap)
        self.assertLess(overlap_pixels, 100, "T-zone and right cheek overlap too much")
        
        # Check left cheek and right cheek don't overlap
        overlap = cv2.bitwise_and(left_cheek_mask, right_cheek_mask)
        overlap_pixels = cv2.countNonZero(overlap)
        self.assertLess(overlap_pixels, 100, "Left and right cheeks overlap too much")


if __name__ == '__main__':
    unittest.main()