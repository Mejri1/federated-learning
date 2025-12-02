"""
Federated Learning Pipeline Wrapper - Handles TensorFlow Import Issues

This script safely runs the federated learning pipeline while avoiding
TensorFlow import errors that occur with Flower on Windows.
"""

import os
import sys
import subprocess
from pathlib import Path

# Set environment variables BEFORE any imports that might trigger TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

# Monkey-patch tensorflow imports to prevent loading
import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None
sys.modules['tensorflow.python.pywrap_tensorflow_internal'] = None

ROOT_DIR = Path(__file__).resolve().parent

def main():
    """Run federated learning pipeline with error handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete Federated Learning Pipeline for Facial Recognition"
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local training epochs per round")
    parser.add_argument("--lr", type=float, default=1e-4, help="Local learning rate")
    
    args = parser.parse_args()
    
    # Import after environment setup
    try:
        from federated.pipeline import main as pipeline_main
        pipeline_main_func = pipeline_main
    except ImportError as e:
        print(f"Error importing pipeline: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        sys.argv = ["pipeline.py", "--rounds", str(args.rounds), "--epochs", str(args.epochs), "--lr", str(args.lr)]
        pipeline_main_func()
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
