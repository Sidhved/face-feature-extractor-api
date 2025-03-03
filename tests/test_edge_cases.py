"""Test facial feature extraction in edge cases."""

import os
import cv2
import requests
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path

from tests.test_config import TEST_IMAGES_DIR, TEST_RESULTS_DIR, EDGE_CASE_IMAGE_URLS
from app.services.landmark_detection import landmark_detector
from app.services.region_extraction import region_extractor
from app.services.image_utils import image_utils

# Define edge case categories
EDGE_CASES = [
    {"url": "https://images.unsplash.com/photo-1509783236416-c9ad59bae472?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8ZmFjZSUyMHdpdGglMjBnbGFzc2VzfGVufDB8fDB8fHww", "category": "glasses"},
    {"url": "https://images.unsplash.com/photo-1444069069008-83a57aac43ac?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZmFjZSUyMHdpdGglMjBnbGFzc2VzfGVufDB8fDB8fHww", "category": "facial_hair"},
    {"url": "https://images.unsplash.com/photo-1493321384838-70c5a85ba487?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8ZmFjZSUyMHdpdGglMjBtYWtldXB8ZW58MHx8MHx8fDA%3D", "category": "makeup"},
    {"url": "https://images.unsplash.com/photo-1642782151736-7ab8649a332b?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8ZmFjZSUyMGluJTIwbG93JTIwbGlnaHR8ZW58MHx8MHx8fDA%3D", "category": "low_light"},
    {"url": "https://images.unsplash.com/photo-1508184964240-ee96bb9677a7?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTl8fGZhY2UlMjBpbiUyMGV4dHJlbWUlMjBwb3NlfGVufDB8fDB8fHww", "category": "extreme_pose"}
    # Add more edge cases
]

def run_edge_case_tests():
    """Run tests for edge cases and generate a report."""
    # Initialize detector
    if not landmark_detector.initialized:
        landmark_detector.initialize()
    
    results = []
    
    for idx, case in enumerate(EDGE_CASES):
        # Skip if URL is just a placeholder
        if case["url"].startswith("https://example.com"):
            continue
            
        # Download image if needed
        image_path = os.path.join(TEST_IMAGES_DIR, f"edge_{idx}.jpg")
        if not os.path.exists(image_path):
            try:
                response = requests.get(case["url"])
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download image {case['url']}: {str(e)}")
                continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        
        # Process image
        start_time = time.time()
        landmark_result = landmark_detector.detect_landmarks(image)
        
        if not landmark_result["success"]:
            results.append({
                **case,
                "success": False,
                "processing_time": time.time() - start_time,
                "message": landmark_result["message"]
            })
            continue
        
        # Extract regions
        region_result = region_extractor.extract_regions(image, landmark_result["landmarks"])
        processing_time = time.time() - start_time
        
        # Generate visualization
        vis_img = None
        if region_result["success"]:
            vis_img = region_extractor.overlay_regions(
                landmark_detector.draw_landmarks(image, landmark_result["landmarks"]),
                region_result["region_masks"]
            )
            vis_path = os.path.join(TEST_RESULTS_DIR, f"edge_{idx}_vis.jpg")
            cv2.imwrite(vis_path, vis_img)
        
        # Save result
        result = {
            **case,
            "success": True,
            "processing_time": processing_time,
            "landmark_count": len(landmark_result["landmarks"])
        }
        results.append(result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(os.path.join(TEST_RESULTS_DIR, "edge_case_results.csv"), index=False)
        
        # Generate summary report
        with open(os.path.join(TEST_RESULTS_DIR, "edge_case_summary.txt"), 'w') as f:
            f.write("Edge Case Testing Summary\n")
            f.write("========================\n\n")
            
            f.write(f"Total edge cases tested: {len(df)}\n")
            f.write(f"Success rate: {df['success'].mean() * 100:.1f}%\n")
            f.write(f"Average processing time: {df['processing_time'].mean():.3f} seconds\n\n")
            
            f.write("Performance by category:\n")
            group_stats = df.groupby('category').agg({
                'success': 'mean',
                'processing_time': 'mean'
            })
            for category, row in group_stats.iterrows():
                f.write(f"  {category}: {row['success']*100:.1f}% success, {row['processing_time']:.3f}s avg time\n")
    
    print(f"Edge case results saved to {TEST_RESULTS_DIR}")
    return df

if __name__ == "__main__":
    run_edge_case_tests()