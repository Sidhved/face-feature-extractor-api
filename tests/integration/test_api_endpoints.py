"""Integration tests for API endpoints using pytest."""

import pytest
import json
import os
import requests
from typing import Generator
import subprocess
import time
import signal
from pathlib import Path

import uvicorn
import multiprocessing
from tests.test_config import TEST_IMAGES_DIR, TEST_RESULTS_DIR, TEST_TIMEOUT

# Create necessary directories if they don't exist
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


class TestAPIWithExternalServer:
    """Test API endpoints by running an actual server."""
    
    @classmethod
    def setup_class(cls):
        """Start the FastAPI server in a separate process."""
        # Start server as a separate process
        cls.server_process = subprocess.Popen(
            ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # This allows us to kill the process group later
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Base URL for all requests
        cls.base_url = "http://127.0.0.1:8000"
        
        # Test connectivity
        retries = 5
        while retries > 0:
            try:
                response = requests.get(f"{cls.base_url}/health")
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            
            time.sleep(1)
            retries -= 1
        
        if retries <= 0:
            raise RuntimeError("Could not connect to test server")
    
    @classmethod
    def teardown_class(cls):
        """Stop the FastAPI server."""
        # Kill the server process group, handling potential errors
        if hasattr(cls, 'server_process') and cls.server_process:
            try:
                os.killpg(os.getpgid(cls.server_process.pid), signal.SIGTERM)
                cls.server_process.wait()
            except (ProcessLookupError, OSError):
                # Process might already be terminated
                pass
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200, "Health check failed"
        data = response.json()
        assert data["status"] == "healthy", "Unexpected health status"
        assert data["dlib_initialized"], "Dlib not initialized"
    
    def test_landmark_detection_url(self):
        """Test landmark detection from URL."""
        test_url = "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8ZHJ5JTIwc2tpbiUyMGZhY2V8ZW58MHx8MHx8fDA%3D"
        response = requests.post(
            f"{self.base_url}/api/v1/landmarks/detect/url",
            json={"url": test_url}
        )
        assert response.status_code == 200, "URL detection failed"
        data = response.json()
        assert data["status"] == "success", f"Detection error: {data.get('message')}"
        assert data["face_detected"], "No face detected"
        assert data["landmarks"] is not None, "No landmarks returned"
        assert data["regions_of_interest"] is not None, "No regions returned"
        assert data["masked_image"] is not None, "No visualization returned"
        
        # Save results for inspection
        with open(os.path.join(TEST_RESULTS_DIR, "url_detection_result.json"), 'w') as f:
            # Save a copy without the large base64 image
            save_data = data.copy()
            save_data["masked_image"] = "[base64_image_data]"
            json.dump(save_data, f, indent=2)
    
    def test_landmark_detection_upload(self):
        """Test landmark detection from file upload."""
        test_image_path = os.path.join(TEST_IMAGES_DIR, "test_image_0.jpg")
        with open(test_image_path, 'rb') as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = requests.post(
                f"{self.base_url}/api/v1/landmarks/detect/upload",
                files=files
            )
        assert response.status_code == 200, "File upload detection failed"
        data = response.json()
        assert data["status"] == "success", f"Detection error: {data.get('message')}"
        assert data["face_detected"], "No face detected"
        assert data["landmarks"] is not None, "No landmarks returned"
        assert data["regions_of_interest"] is not None, "No regions returned"
        
        # Save results for inspection
        with open(os.path.join(TEST_RESULTS_DIR, "upload_detection_result.json"), 'w') as f:
            # Save a copy without the large base64 image
            save_data = data.copy()
            save_data["masked_image"] = "[base64_image_data]"
            json.dump(save_data, f, indent=2)
    
    def test_result_saving(self):
        """Test saving results."""
        test_url = "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OHx8ZHJ5JTIwc2tpbiUyMGZhY2V8ZW58MHx8MHx8fDA%3D"
        response = requests.post(
            f"{self.base_url}/api/v1/landmarks/detect/url?save_result=true",
            json={"url": test_url}
        )
        assert response.status_code == 200, "URL detection with saving failed"
        data = response.json()
        assert data["status"] == "success", f"Detection error: {data.get('message')}"
        assert "result_file" in data, "Result file path not in response"
        
        # Check that the file exists
        result_file = data["result_file"]
        assert os.path.exists(result_file), f"Result file does not exist: {result_file}"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])