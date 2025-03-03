"""Benchmark facial feature extraction performance."""

import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import requests
from pathlib import Path

from tests.test_config import TEST_IMAGES_DIR, TEST_RESULTS_DIR, TEST_IMAGE_URLS
from app.services.landmark_detection import landmark_detector
from app.services.region_extraction import region_extractor
from app.services.image_utils import image_utils

def benchmark_single_image(image_path, num_runs=5):
    """Benchmark performance on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return None
    
    metrics = {
        "landmark_detection_times": [],
        "region_extraction_times": [],
        "total_times": [],
        "cpu_percent": [],
        "memory_usage_mb": []
    }
    
    # Initialize detector if needed
    if not landmark_detector.initialized:
        landmark_detector.initialize()
    
    for i in range(num_runs):
        # Track CPU and memory
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run landmark detection
        start_time = time.time()
        landmark_result = landmark_detector.detect_landmarks(image)
        landmark_time = time.time() - start_time
        
        if not landmark_result["success"]:
            print(f"Landmark detection failed: {landmark_result['message']}")
            continue
        
        # Run region extraction
        start_time = time.time()
        region_result = region_extractor.extract_regions(image, landmark_result["landmarks"])
        region_time = time.time() - start_time
        
        # Record metrics
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        metrics["landmark_detection_times"].append(landmark_time)
        metrics["region_extraction_times"].append(region_time)
        metrics["total_times"].append(landmark_time + region_time)
        metrics["cpu_percent"].append(psutil.cpu_percent(interval=None))
        metrics["memory_usage_mb"].append(end_memory - start_memory)
    
    # Calculate summary statistics
    summary = {}
    for key, values in metrics.items():
        if not values:
            continue
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)
        summary[f"{key}_min"] = np.min(values)
        summary[f"{key}_max"] = np.max(values)
    
    return summary

def run_benchmark(num_images=5, num_runs=5):
    """Run benchmarks on multiple images."""
    # Download test images if needed
    for i, url in enumerate(TEST_IMAGE_URLS[:num_images]):
        image_path = os.path.join(TEST_IMAGES_DIR, f"benchmark_{i}.jpg")
        if not os.path.exists(image_path):
            try:
                response = requests.get(url)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded benchmark image to {image_path}")
            except Exception as e:
                print(f"Failed to download {url}: {str(e)}")
                continue
    
    results = []
    for i in range(num_images):
        image_path = os.path.join(TEST_IMAGES_DIR, f"benchmark_{i}.jpg")
        if not os.path.exists(image_path):
            continue
        
        print(f"Benchmarking image {i+1}/{num_images}...")
        metrics = benchmark_single_image(image_path, num_runs)
        if metrics:
            metrics["image_index"] = i
            metrics["image_path"] = image_path
            img = cv2.imread(image_path)
            metrics["image_width"] = img.shape[1]
            metrics["image_height"] = img.shape[0]
            results.append(metrics)
    
    # Create results dataframe
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(TEST_RESULTS_DIR, "benchmark_results.csv"), index=False)
        
        # Generate summary report
        with open(os.path.join(TEST_RESULTS_DIR, "benchmark_summary.txt"), 'w') as f:
            f.write("Performance Benchmark Summary\n")
            f.write("============================\n\n")
            
            f.write(f"Number of images tested: {len(df)}\n")
            f.write(f"Runs per image: {num_runs}\n\n")
            
            f.write("Average processing times:\n")
            f.write(f"  Landmark detection: {df['landmark_detection_times_mean'].mean():.3f}s (±{df['landmark_detection_times_std'].mean():.3f}s)\n")
            f.write(f"  Region extraction: {df['region_extraction_times_mean'].mean():.3f}s (±{df['region_extraction_times_std'].mean():.3f}s)\n")
            f.write(f"  Total processing: {df['total_times_mean'].mean():.3f}s (±{df['total_times_std'].mean():.3f}s)\n\n")
            
            f.write("Resource usage:\n")
            f.write(f"  CPU usage: {df['cpu_percent_mean'].mean():.1f}% (±{df['cpu_percent_std'].mean():.1f}%)\n")
            f.write(f"  Memory increase: {df['memory_usage_mb_mean'].mean():.2f}MB (±{df['memory_usage_mb_std'].mean():.2f}MB)\n")
            
        # Create visualizations
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(df)), df["total_times_mean"], yerr=df["total_times_std"], alpha=0.7)
        plt.xlabel("Image index")
        plt.ylabel("Processing time (seconds)")
        plt.title("Total Processing Time per Image")
        plt.tight_layout()
        plt.savefig(os.path.join(TEST_RESULTS_DIR, "benchmark_times.png"))
        
        print(f"Benchmark results saved to {TEST_RESULTS_DIR}")
        return df
    
    return None

if __name__ == "__main__":
    run_benchmark(num_images=3, num_runs=5)