"""
Quick Start Script for Physi-Cast
Runs the complete system with API and Dashboard
"""

import subprocess
import time
import os
import sys
from pathlib import Path


def run_training():
    """Run model training"""
    print("\n" + "="*80)
    print("TRAINING PINN MODEL")
    print("="*80)
    
    cmd = [
        sys.executable, 'main.py',
        '--n-train-samples', '5000',
        '--n-collocation', '5000',
        '--adam-epochs', '100',
        '--learning-rate', '0.001'
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode == 0


def run_api():
    """Run FastAPI backend"""
    print("\n" + "="*80)
    print("STARTING FASTAPI BACKEND")
    print("="*80)
    
    cmd = [
        sys.executable, '-m', 'uvicorn',
        'api.server:app',
        '--reload',
        '--port', '8000',
        '--host', '0.0.0.0'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("API will be available at http://localhost:8000")
    print("API Docs: http://localhost:8000/docs\n")
    
    subprocess.run(cmd, cwd=Path(__file__).parent)


def run_dashboard():
    """Run Streamlit dashboard"""
    print("\n" + "="*80)
    print("STARTING STREAMLIT DASHBOARD")
    print("="*80)
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'dashboard/app.py',
        '--logger.level=info'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("Dashboard will be available at http://localhost:8501\n")
    
    subprocess.run(cmd, cwd=Path(__file__).parent)


def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'fastapi', 'streamlit', 'requests']
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Physi-Cast Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quickstart.py --train              # Only train model
  python quickstart.py --api                # Only run API
  python quickstart.py --dashboard          # Only run dashboard
  python quickstart.py --full               # Run all (requires multiple terminals)
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--api', action='store_true', help='Run API backend')
    parser.add_argument('--dashboard', action='store_true', help='Run Streamlit dashboard')
    parser.add_argument('--full', action='store_true', help='Run everything sequentially')
    parser.add_argument('--check', action='store_true', help='Check dependencies only')
    
    args = parser.parse_args()
    
    # Check dependencies
    print("[CHECK] Verifying dependencies...")
    if not check_dependencies():
        print("[ERROR] Missing dependencies. Please install requirements.")
        sys.exit(1)
    print("[CHECK] ✓ All dependencies found\n")
    
    if args.check:
        print("[CHECK] Dependency check complete")
        sys.exit(0)
    
    # Determine what to run
    if args.full:
        print("\n" + "="*80)
        print("PHYSI-CAST FULL SYSTEM START")
        print("="*80)
        
        # Train
        if not run_training():
            print("[ERROR] Training failed")
            sys.exit(1)
        
        # Start API
        print("\n[INFO] Starting API backend...")
        print("[INFO] In production, run this in a separate terminal:")
        print("[INFO] python quickstart.py --api")
        
    elif args.train:
        run_training()
    
    elif args.api:
        run_api()
    
    elif args.dashboard:
        run_dashboard()
    
    else:
        # Default: print instructions
        print("\n" + "="*80)
        print("PHYSI-CAST QUICK START GUIDE")
        print("="*80)
        
        print("\nTo run the complete system, follow these steps:\n")
        
        print("STEP 1: Train the PINN Model")
        print("-" * 80)
        print("python quickstart.py --train\n")
        
        print("STEP 2: Start the FastAPI Backend (in terminal 1)")
        print("-" * 80)
        print("python quickstart.py --api\n")
        
        print("STEP 3: Start the Streamlit Dashboard (in terminal 2)")
        print("-" * 80)
        print("python quickstart.py --dashboard\n")
        
        print("STEP 4: Access the System")
        print("-" * 80)
        print("- Dashboard: http://localhost:8501")
        print("- API: http://localhost:8000")
        print("- API Docs: http://localhost:8000/docs\n")
        
        print("="*80)
        print("For more options, run: python quickstart.py --help")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
