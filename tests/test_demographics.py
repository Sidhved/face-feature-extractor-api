"""Test facial feature extraction across different demographics."""

import os
import cv2
import requests
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt

from tests.test_config import TEST_IMAGES_DIR, TEST_RESULTS_DIR
from app.services.landmark_detection import landmark_detector
from app.services.region_extraction import region_extractor
from app.services.image_utils import image_utils

# Define demographic categories to test
# For a real test, you would need a properly labeled dataset
DEMOGRAPHIC_CATEGORIES = {
    "ethnicity": ["caucasian", "asian", "black", "hispanic", "middle_eastern", "south_asian"],
    "gender": ["male", "female", "non_binary"],
    "age_group": ["child", "young_adult", "adult", "senior"]
}

# For now, we'll simulate with a small set of images
DEMO_IMAGES = [
    {"url": "https://images.unsplash.com/photo-1521146764736-56c929d59c83?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8ZHJ5JTIwc2tpbiUyMGZhY2UlMjBjYXVjZXNpYW4lMjBmZW1hbGUlMjB5b3VuZ3xlbnwwfHwwfHx8MA%3D%3D", 
     "ethnicity": "caucasian", "gender": "female", "age_group": "young_adult"},
    {"url": "https://plus.unsplash.com/premium_photo-1723874469356-1b483b86f32f?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8ZmFjZSUyMGJsYWNrJTIwYWR1bHQlMjBtYWxlfGVufDB8fDB8fHww", 
     "ethnicity": "black", "gender": "male", "age_group": "adult"}
    # Add more demographically diverse images
]

def run_demographic_tests():
    """Run tests across different demographics and generate a report."""
    # Initialize detector
    if not landmark_detector.initialized:
        landmark_detector.initialize()
    
    results = []
    
    for idx, image_info in enumerate(DEMO_IMAGES):
        # Download image if needed
        image_path = os.path.join(TEST_IMAGES_DIR, f"demo_{idx}.jpg")
        if not os.path.exists(image_path):
            response = requests.get(image_info["url"])
            with open(image_path, 'wb') as f:
                f.write(response.content)
        
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
                **image_info,
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
            vis_path = os.path.join(TEST_RESULTS_DIR, f"demo_{idx}_vis.jpg")
            cv2.imwrite(vis_path, vis_img)
        
        # Calculate region metrics
        region_metrics = {}
        if region_result["success"]:
            for region_name, mask in region_result["region_masks"].items():
                region_metrics[f"{region_name}_area"] = cv2.countNonZero(mask)
        
        # Save result
        result = {
            **image_info,
            "success": True,
            "processing_time": processing_time,
            "landmark_count": len(landmark_result["landmarks"]),
            **region_metrics
        }
        results.append(result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(TEST_RESULTS_DIR, "demographic_results.csv"), index=False)
    
    # Generate summary report
    if len(df) > 0:
        with open(os.path.join(TEST_RESULTS_DIR, "demographic_summary.txt"), 'w') as f:
            f.write("Demographic Testing Summary\n")
            f.write("===========================\n\n")
            
            f.write(f"Total images processed: {len(df)}\n")
            f.write(f"Success rate: {df['success'].mean() * 100:.1f}%\n")
            f.write(f"Average processing time: {df['processing_time'].mean():.3f} seconds\n\n")
            
            f.write("Performance by demographic:\n")
            for category in ["ethnicity", "gender", "age_group"]:
                f.write(f"\n{category.capitalize()} breakdown:\n")
                group_stats = df.groupby(category).agg({
                    'success': 'mean',
                    'processing_time': 'mean'
                })
                for group, row in group_stats.iterrows():
                    f.write(f"  {group}: {row['success']*100:.1f}% success, {row['processing_time']:.3f}s avg time\n")
            
            f.write("\nRegion metrics:\n")
            region_cols = [col for col in df.columns if col.endswith('_area')]
            if region_cols:
                for col in region_cols:
                    region = col.replace('_area', '')
                    f.write(f"  {region}: {df[col].mean():.1f} avg pixel area\n")

    print(f"Results saved to {TEST_RESULTS_DIR}")
    return df

if __name__ == "__main__":
    run_demographic_tests()