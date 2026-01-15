#!/bin/bash
# Test Runner for Linux/Mac
#
# Usage:
#   ./run_tests.sh              - Run all tests
#   ./run_tests.sh unit         - Run unit tests
#   ./run_tests.sh integration  - Run integration tests
#   ./run_tests.sh coverage     - Run with coverage

echo "============================================================"
echo "FRAUD DETECTION SYSTEM - TEST SUITE"
echo "============================================================"

if [ -z "$1" ]; then
    echo "Running all tests..."
    python run_tests.py
elif [ "$1" = "unit" ]; then
    echo "Running unit tests..."
    python run_tests.py --unit
elif [ "$1" = "integration" ]; then
    echo "Running integration tests..."
    python run_tests.py --integration
elif [ "$1" = "api" ]; then
    echo "Running API tests..."
    python run_tests.py --api
elif [ "$1" = "coverage" ]; then
    echo "Running tests with coverage..."
    python run_tests.py --coverage --html
elif [ "$1" = "fast" ]; then
    echo "Running fast tests..."
    python run_tests.py --fast
else
    echo "Unknown option: $1"
    echo ""
    echo "Usage:"
    echo "  ./run_tests.sh              - Run all tests"
    echo "  ./run_tests.sh unit         - Run unit tests"
    echo "  ./run_tests.sh integration  - Run integration tests"
    echo "  ./run_tests.sh api          - Run API tests"
    echo "  ./run_tests.sh coverage     - Run with coverage"
    echo "  ./run_tests.sh fast         - Skip slow tests"
fi

