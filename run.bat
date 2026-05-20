@echo off
REM Physi-Cast Quick Start - Windows Batch Script
REM This script helps you get started with Physi-Cast

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo                    PHYSI-CAST QUICK START (Windows)
echo ============================================================================
echo.

if "%1"=="" (
    echo Usage: run.bat [command]
    echo.
    echo Commands:
    echo   check       - Check dependencies
    echo   train       - Train PINN model
    echo   api         - Start FastAPI backend
    echo   dashboard   - Start Streamlit dashboard
    echo   test        - Run component tests
    echo.
    echo Examples:
    echo   run.bat check
    echo   run.bat train
    echo   run.bat api
    echo.
    exit /b 0
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Execute command
if "%1"=="check" (
    echo Checking dependencies...
    python test_components.py --physics
    goto end
)

if "%1"=="train" (
    echo Training PINN model...
    python main.py --n-train-samples 5000 --adam-epochs 100
    goto end
)

if "%1"=="api" (
    echo Starting FastAPI backend on http://localhost:8000
    python -m uvicorn api.server:app --reload --port 8000
    goto end
)

if "%1"=="dashboard" (
    echo Starting Streamlit dashboard on http://localhost:8501
    streamlit run dashboard/app.py
    goto end
)

if "%1"=="test" (
    echo Running component tests...
    python test_components.py --all
    goto end
)

echo Unknown command: %1
exit /b 1

:end
echo.
echo Finished.
pause
