"""
Test Runner Script

Runs the test suite with various options and generates coverage reports.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --fast             # Skip slow tests

Author: Your Name
Date: 2026-01-15
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(
    test_type: str = "all",
    coverage: bool = False,
    verbose: bool = False,
    fast: bool = False,
    parallel: bool = False,
    html_report: bool = False
):
    """
    Run the test suite.
    
    Args:
        test_type: Type of tests to run ('all', 'unit', 'integration', 'api')
        coverage: Whether to generate coverage report
        verbose: Whether to use verbose output
        fast: Whether to skip slow tests
        parallel: Whether to run tests in parallel
        html_report: Whether to generate HTML coverage report
    """
    print("="*60)
    print("FRAUD DETECTION SYSTEM - TEST SUITE")
    print("="*60)
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit")
        print("\n📋 Running: Unit Tests")
    elif test_type == "integration":
        cmd.append("tests/integration")
        print("\n📋 Running: Integration Tests")
    elif test_type == "api":
        cmd.extend(["-m", "api"])
        print("\n📋 Running: API Tests")
    else:
        cmd.append("tests")
        print("\n📋 Running: All Tests")
    
    # Add markers
    if fast:
        cmd.extend(["-m", "not slow"])
        print("⚡ Mode: Fast (skipping slow tests)")
    
    # Add verbosity
    if verbose:
        cmd.append("-vv")
        print("📢 Mode: Verbose")
    else:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov=api",
            "--cov-report=term-missing"
        ])
        
        if html_report:
            cmd.append("--cov-report=html")
            print("📊 Coverage: Enabled (HTML report will be generated)")
        else:
            print("📊 Coverage: Enabled")
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
        print("⚡ Parallel: Enabled")
    
    # Add other options
    cmd.extend([
        "--tb=short",
        "--color=yes"
    ])
    
    print("\n" + "="*60)
    print("RUNNING TESTS...")
    print("="*60)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        print("\n" + "="*60)
        if result.returncode == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*60)
        
        if coverage and html_report:
            print("\n📊 Coverage report generated: htmlcov/index.html")
            print("   Open in browser to view detailed coverage")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run fraud detection system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --api              # Run only API tests
  python run_tests.py --coverage         # Run with coverage
  python run_tests.py --coverage --html  # Run with HTML coverage report
  python run_tests.py --fast             # Skip slow tests
  python run_tests.py --parallel         # Run tests in parallel
  python run_tests.py -v                 # Verbose output
        """
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run only API tests"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (requires --coverage)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    # Determine test type
    test_type = "all"
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.api:
        test_type = "api"
    
    # Run tests
    exit_code = run_tests(
        test_type=test_type,
        coverage=args.coverage,
        verbose=args.verbose,
        fast=args.fast,
        parallel=args.parallel,
        html_report=args.html
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

