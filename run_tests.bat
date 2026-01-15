@echo off
REM Test Runner for Windows
REM
REM Usage:
REM   run_tests.bat              - Run all tests
REM   run_tests.bat unit         - Run unit tests
REM   run_tests.bat integration  - Run integration tests
REM   run_tests.bat coverage     - Run with coverage

echo ============================================================
echo FRAUD DETECTION SYSTEM - TEST SUITE
echo ============================================================

if "%1"=="" (
    echo Running all tests...
    python run_tests.py
) else if "%1"=="unit" (
    echo Running unit tests...
    python run_tests.py --unit
) else if "%1"=="integration" (
    echo Running integration tests...
    python run_tests.py --integration
) else if "%1"=="api" (
    echo Running API tests...
    python run_tests.py --api
) else if "%1"=="coverage" (
    echo Running tests with coverage...
    python run_tests.py --coverage --html
) else if "%1"=="fast" (
    echo Running fast tests...
    python run_tests.py --fast
) else (
    echo Unknown option: %1
    echo.
    echo Usage:
    echo   run_tests.bat              - Run all tests
    echo   run_tests.bat unit         - Run unit tests
    echo   run_tests.bat integration  - Run integration tests
    echo   run_tests.bat api          - Run API tests
    echo   run_tests.bat coverage     - Run with coverage
    echo   run_tests.bat fast         - Skip slow tests
)

