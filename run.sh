#!/bin/bash

# Physi-Cast Quick Start - Linux/Mac Bash Script
# This script helps you get started with Physi-Cast

set -e

echo ""
echo "============================================================================"
echo "                    PHYSI-CAST QUICK START (Linux/Mac)"
echo "============================================================================"
echo ""

# Function to print usage
usage() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check       - Check dependencies"
    echo "  train       - Train PINN model"
    echo "  api         - Start FastAPI backend"
    echo "  dashboard   - Start Streamlit dashboard"
    echo "  test        - Run component tests"
    echo ""
    echo "Examples:"
    echo "  ./run.sh check"
    echo "  ./run.sh train"
    echo "  ./run.sh api"
    echo ""
}

# Check if command provided
if [ $# -eq 0 ]; then
    usage
    exit 0
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Execute command
case "$1" in
    check)
        echo "Checking dependencies..."
        python test_components.py --physics
        ;;
    
    train)
        echo "Training PINN model..."
        python main.py --n-train-samples 5000 --adam-epochs 100
        ;;
    
    api)
        echo "Starting FastAPI backend on http://localhost:8000"
        python -m uvicorn api.server:app --reload --port 8000
        ;;
    
    dashboard)
        echo "Starting Streamlit dashboard on http://localhost:8501"
        streamlit run dashboard/app.py
        ;;
    
    test)
        echo "Running component tests..."
        python test_components.py --all
        ;;
    
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac

echo ""
echo "Finished."
