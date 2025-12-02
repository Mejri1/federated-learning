#!/usr/bin/env python
"""
Quick Start Guide for Federated Learning Pipeline
This script demonstrates how to run the complete pipeline with different configurations.

Run examples:
    python federated/quickstart.py --mode basic
    python federated/quickstart.py --mode advanced
    python federated/quickstart.py --mode evaluate
"""

import sys
from pathlib import Path
import logging

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print("  FEDERATED LEARNING FOR FACIAL RECOGNITION - QUICK START GUIDE")
    print("="*80 + "\n")


def print_option(num, title, description, command):
    """Print a formatted option."""
    print(f"\n{num}. {title}")
    print(f"   Description: {description}")
    print(f"   Command: {command}\n")


def main():
    """Show quick start options."""
    print_banner()
    
    logger.info("Welcome to the Federated Learning Pipeline!")
    logger.info("This guide will help you get started with training a federated model.\n")
    
    print("AVAILABLE OPTIONS:\n")
    
    print_option(
        1,
        "Run Full Pipeline (Recommended for First Time)",
        "Complete end-to-end federated learning with evaluation and saving",
        "python federated/pipeline.py --rounds 5 --epochs 2 --lr 1e-4"
    )
    
    print_option(
        2,
        "Run Federated Learning Only",
        "Just run the federated learning rounds without full pipeline",
        "python federated/run_flwr.py --rounds 10 --epochs 3 --lr 1e-4"
    )
    
    print_option(
        3,
        "Evaluate Existing Model",
        "Evaluate a trained model on centralized and client-specific test sets",
        "python federated/evaluate.py --model models/federated_model_*.pt"
    )
    
    print_option(
        4,
        "Quick Test Run",
        "Test the pipeline with minimal rounds (good for debugging)",
        "python federated/pipeline.py --rounds 2 --epochs 1 --lr 1e-4"
    )
    
    print_option(
        5,
        "Production Run",
        "Run with higher rounds and epochs for better convergence",
        "python federated/pipeline.py --rounds 20 --epochs 5 --lr 1e-4"
    )
    
    print("\n" + "-"*80)
    print("RECOMMENDED WORKFLOW:")
    print("-"*80)
    print("""
1. Start with Quick Test Run to verify setup
   python federated/pipeline.py --rounds 2 --epochs 1 --lr 1e-4

2. Run Full Pipeline for baseline results
   python federated/pipeline.py --rounds 5 --epochs 2 --lr 1e-4

3. Evaluate the trained model
   python federated/evaluate.py --model models/federated_model_*.pt

4. Iterate with different hyperparameters as needed
   python federated/pipeline.py --rounds 10 --epochs 3 --lr 5e-5
""")
    
    print("-"*80)
    print("CONFIGURATION PARAMETERS:")
    print("-"*80)
    print("""
--rounds N          Number of federated learning rounds (default: 5)
                    - Lower (2-5): Quick testing
                    - Medium (5-10): Good balance
                    - Higher (20+): Better convergence

--epochs N          Local training epochs per round (default: 2)
                    - Lower (1): Faster, less local training
                    - Medium (2-3): Good balance
                    - Higher (5+): More local training

--lr FLOAT          Learning rate (default: 1e-4)
                    - Lower (1e-5): More stable, slower
                    - Default (1e-4): Recommended
                    - Higher (1e-3): Faster, may be unstable
""")
    
    print("-"*80)
    print("OUTPUT FILES:")
    print("-"*80)
    print("""
After running the pipeline, check the 'models/' directory for:

1. federated_model_<timestamp>.pt
   - The trained model checkpoint
   - Can be loaded for inference or further training

2. federated_history_<timestamp>.json
   - Training metrics for each round
   - Accuracy and loss progression
   - Contains all training statistics

3. evaluation_results_<timestamp>.json
   - Evaluation on centralized test set
   - Per-client evaluation results
   - Summary statistics

4. pipeline_summary_<timestamp>.json
   - Complete pipeline report
   - Configuration and artifacts
   - Final metrics and progress
""")
    
    print("-"*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("""
✓ Choose one of the recommended workflows above
✓ Run the command in your terminal
✓ Monitor the logs for progress
✓ Check the models/ directory for results
✓ Review JSON files for detailed metrics

For more information, see README_PIPELINE.md
""")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
