"""Run all tests for the facial feature extraction API."""

import os
import sys
import unittest
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_config import TEST_IMAGES_DIR, TEST_RESULTS_DIR
from tests.benchmark import run_benchmark
from tests.test_demographics import run_demographic_tests
from tests.test_edge_cases import run_edge_case_tests


def run_unit_tests():
    """Run all unit tests."""
    print("\n=== Running Unit Tests ===\n")
    unit_tests = unittest.TestLoader().discover('tests/unit', pattern='test_*.py')
    unittest.TextTestRunner(verbosity=2).run(unit_tests)


def run_integration_tests():
    """Run all integration tests."""
    print("\n=== Running Integration Tests ===\n")
    integration_tests = unittest.TestLoader().discover('tests/integration', pattern='test_*.py')
    unittest.TextTestRunner(verbosity=2).run(integration_tests)


def run_all_tests():
    """Run all tests and benchmarks."""
    start_time = time.time()
    
    # Run unit tests
    run_unit_tests()
    
    # Run integration tests
    run_integration_tests()
    
    # Run benchmarks
    print("\n=== Running Performance Benchmarks ===\n")
    run_benchmark(num_images=2, num_runs=3)
    
    # Run demographic tests
    print("\n=== Running Demographic Tests ===\n")
    run_demographic_tests()
    
    # Run edge case tests
    print("\n=== Running Edge Case Tests ===\n")
    run_edge_case_tests()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n=== Test Suite Completed in {total_time:.2f} seconds ===")
    print(f"Test results available in: {TEST_RESULTS_DIR}")


if __name__ == "__main__":
    # Create test directories if they don't exist
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    run_all_tests()