#!/usr/bin/env python3
"""
Test runner script for Tag Sentinel.

This script provides convenient ways to run different test suites.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"
    
    project_root = Path(__file__).parent
    
    success = True
    
    if test_type in ("all", "unit"):
        # Run unit tests
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
        if not run_command(cmd, "Unit Tests"):
            success = False
    
    if test_type in ("all", "integration"):
        # Run integration tests
        cmd = ["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short"]
        if not run_command(cmd, "Integration Tests"):
            success = False
    
    if test_type == "coverage":
        # Run with coverage
        try:
            subprocess.run(["python", "-m", "pip", "install", "pytest-cov"], check=True)
        except subprocess.CalledProcessError:
            pass
        
        cmd = [
            "python", "-m", "pytest", 
            "--cov=app", 
            "--cov-report=html", 
            "--cov-report=term-missing",
            "tests/"
        ]
        if not run_command(cmd, "Coverage Tests"):
            success = False
    
    if test_type == "quick":
        # Run quick tests only (exclude slow ones)
        cmd = ["python", "-m", "pytest", "-m", "not slow", "tests/", "-v"]
        if not run_command(cmd, "Quick Tests"):
            success = False
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())