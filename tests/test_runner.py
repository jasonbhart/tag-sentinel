"""Test runner for DataLayer test suite."""

import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unit_tests():
    """Run unit tests with coverage reporting."""
    print("🧪 Running DataLayer Unit Tests...")
    args = [
        "tests/audit/datalayer/",
        "-v",
        "--tb=short",
        "-m", "unit or not integration",
        "--durations=10"
    ]
    return pytest.main(args)


def run_integration_tests():
    """Run integration tests."""
    print("🔗 Running DataLayer Integration Tests...")
    args = [
        "tests/integration/",
        "-v", 
        "--tb=short",
        "-m", "integration",
        "--durations=20"
    ]
    return pytest.main(args)


def run_performance_tests():
    """Run performance/load tests."""
    print("⚡ Running DataLayer Performance Tests...")
    args = [
        "tests/integration/",
        "-v",
        "--tb=short", 
        "-m", "slow or performance",
        "--durations=30"
    ]
    return pytest.main(args)


def run_all_tests():
    """Run complete test suite."""
    print("🚀 Running Complete DataLayer Test Suite...")
    args = [
        "tests/audit/datalayer/",
        "tests/integration/test_datalayer_integration.py",
        "tests/integration/test_datalayer_browser_scenarios.py", 
        "tests/integration/test_datalayer_validation_scenarios.py",
        "-v",
        "--tb=short",
        "--durations=20"
    ]
    return pytest.main(args)


def main():
    """Main test runner with options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DataLayer Test Suite Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.unit:
        return run_unit_tests()
    elif args.integration:
        return run_integration_tests()
    elif args.performance:
        return run_performance_tests()
    elif args.all:
        return run_all_tests()
    else:
        print("Choose test type:")
        print("  --unit         Run unit tests")
        print("  --integration  Run integration tests")
        print("  --performance  Run performance tests")
        print("  --all         Run all tests")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)