"""Test configuration for facial feature extraction API."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test dataset paths
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'test_images')
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')

# Create directories if they don't exist
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Test image URLs - diverse set of faces
TEST_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8ZHJ5JTIwc2tpbiUyMGZhY2V8ZW58MHx8MHx8fDA%3D",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fGRyeSUyMHNraW4lMjBmYWNlfGVufDB8fDB8fHww",
    "https://images.unsplash.com/photo-1554151228-14d9def656e4?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fGRyeSUyMHNraW4lMjBmYWNlfGVufDB8fDB8fHww"
]

# Edge case image URLs (glasses, facial hair, makeup, etc.)
EDGE_CASE_IMAGE_URLS = [
    # Add URLs for edge case images
    # Example: "https://example.com/face_with_glasses.jpg"
]

# Test parameters
TEST_TIMEOUT = 30  # seconds