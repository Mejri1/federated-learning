"""
Results Analysis and Visualization Utility
Analyzes saved federated learning results and provides insights.

Usage:
    python federated/analyze_results.py --history models/federated_history_*.json
    python federated/analyze_results.py --summary models/pipeline_summary_*.json
    python federated/analyze_results.py --comparison models/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import logging
from typing import Dict, List
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def load_json(filepath: Path) -> Dict:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_subheader(text: str):
    """Print formatted subheader."""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80)


def analyze_history(history_file: Path):
    """Analyze training history file."""
    print_header(f"TRAINING HISTORY ANALYSIS: {history_file.name}")
    
    history = load_json(history_file)
    
    # Print metadata
    if "metadata" in history:
        metadata = history["metadata"]
        print("\nConfiguration:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # Analyze centralized metrics
    print_subheader("Centralized Metrics")
    
    metrics_centralized = history.get("metrics_centralized", {})
    
    if "accuracy" in metrics_centralized:
        acc_data = metrics_centralized["accuracy"]
        print(f"\nAccuracy Progress ({len(acc_data)} rounds):")
        print(f"  Round | Accuracy")
        print(f"  ------|----------")
        
        acc_values = []
        for entry in acc_data:
            round_num = entry.get("round", entry[0] if isinstance(entry, list) else "?")
            value = entry.get("value", entry[1] if isinstance(entry, list) else 0)
            print(f"  {round_num:5} | {value:.4f}")
            acc_values.append(value)
        
        if acc_values:
            print(f"\n  Initial Accuracy: {acc_values[0]:.4f}")
            print(f"  Final Accuracy:   {acc_values[-1]:.4f}")
            print(f"  Improvement:      {(acc_values[-1] - acc_values[0])*100:+.2f}%")
            print(f"  Best Accuracy:    {max(acc_values):.4f}")
            print(f"  Worst Accuracy:   {min(acc_values):.4f}")
    
    if "loss" in metrics_centralized:
        loss_data = metrics_centralized["loss"]
        print(f"\nLoss Progress ({len(loss_data)} rounds):")
        print(f"  Round | Loss")
        print(f"  ------|----------")
        
        loss_values = []
        for entry in loss_data:
            round_num = entry.get("round", entry[0] if isinstance(entry, list) else "?")
            value = entry.get("value", entry[1] if isinstance(entry, list) else 0)
            print(f"  {round_num:5} | {value:.4f}")
            loss_values.append(value)
        
        if loss_values:
            print(f"\n  Initial Loss: {loss_values[0]:.4f}")
            print(f"  Final Loss:   {loss_values[-1]:.4f}")
            print(f"  Reduction:    {(loss_values[0] - loss_values[-1])*100:.2f}%")
            print(f"  Best Loss:    {min(loss_values):.4f}")
            print(f"  Worst Loss:   {max(loss_values):.4f}")
    
    # Final metrics
    if "final_accuracy" in history or "final_loss" in history:
        print_subheader("Final Metrics")
        if "final_accuracy" in history:
            print(f"Final Accuracy: {history['final_accuracy']:.4f}")
        if "final_loss" in history:
            print(f"Final Loss: {history['final_loss']:.4f}")


def analyze_evaluation(eval_file: Path):
    """Analyze evaluation results file."""
    print_header(f"EVALUATION RESULTS ANALYSIS: {eval_file.name}")
    
    results = load_json(eval_file)
    
    # Centralized results
    print_subheader("Centralized Test Set")
    centralized = results.get("centralized_evaluation", {})
    print(f"Loss:     {centralized.get('loss', 'N/A'):.4f}")
    print(f"Accuracy: {centralized.get('accuracy', 'N/A'):.4f}")
    print(f"Samples:  {centralized.get('details', {}).get('total_samples', 'N/A')}")
    
    # Per-client results
    print_subheader("Per-Client Evaluation")
    client_eval = results.get("client_evaluation", {})
    
    print(f"\n{'Client':<20} {'Accuracy':>12} {'Loss':>12} {'Samples':>10}")
    print("-" * 56)
    
    for client_id, metrics in client_eval.items():
        acc = metrics.get("accuracy", 0)
        loss = metrics.get("loss", 0)
        samples = metrics.get("test_samples", 0)
        print(f"{client_id:<20} {acc:>12.4f} {loss:>12.4f} {samples:>10}")
    
    # Summary
    print_subheader("Summary Statistics")
    summary = results.get("summary", {})
    
    print(f"Centralized Accuracy:       {summary.get('centralized_accuracy', 'N/A'):.4f}")
    print(f"Average Client Accuracy:    {summary.get('average_client_accuracy', 'N/A'):.4f}")
    print(f"Centralized Loss:           {summary.get('centralized_loss', 'N/A'):.4f}")
    print(f"Average Client Loss:        {summary.get('average_client_loss', 'N/A'):.4f}")


def analyze_summary(summary_file: Path):
    """Analyze pipeline summary file."""
    print_header(f"PIPELINE SUMMARY ANALYSIS: {summary_file.name}")
    
    summary = load_json(summary_file)
    
    # Configuration
    print_subheader("Pipeline Configuration")
    config = summary.get("configuration", {})
    for key, value in config.items():
        if key not in ["clients"]:
            print(f"  {key}: {value}")
    
    if "clients" in config:
        print(f"  Clients: {', '.join(config['clients'])}")
    
    # Artifacts
    print_subheader("Generated Artifacts")
    artifacts = summary.get("artifacts", {})
    for artifact_type, path in artifacts.items():
        print(f"  {artifact_type}: {path}")
    
    # Final metrics
    print_subheader("Final Metrics")
    final_metrics = summary.get("final_metrics", {})
    
    if "aggregated" in final_metrics:
        agg = final_metrics["aggregated"]
        print(f"\n  Centralized Accuracy:    {agg.get('centralized_accuracy', 'N/A'):.4f}")
        print(f"  Average Client Accuracy: {agg.get('average_client_accuracy', 'N/A'):.4f}")
        print(f"  Centralized Loss:        {agg.get('centralized_loss', 'N/A'):.4f}")
        print(f"  Average Client Loss:     {agg.get('average_client_loss', 'N/A'):.4f}")
    
    # Training progress
    if "accuracy_progress" in summary:
        acc_progress = summary["accuracy_progress"]
        print_subheader(f"Accuracy Progress ({len(acc_progress)} rounds)")
        print(f"{'Round':<8} {'Accuracy':>12}")
        print("-" * 22)
        for entry in acc_progress:
            round_num = entry.get("round", "?")
            value = entry.get("value", 0)
            print(f"{round_num:<8} {value:>12.4f}")
        
        if acc_progress:
            values = [e.get("value", 0) for e in acc_progress]
            print(f"\n  Initial: {values[0]:.4f}")
            print(f"  Final:   {values[-1]:.4f}")
            print(f"  Improvement: {(values[-1] - values[0])*100:+.2f}%")


def compare_runs(models_dir: Path):
    """Compare multiple runs."""
    print_header("COMPARING MULTIPLE RUNS")
    
    # Find all summary files
    summary_files = sorted(models_dir.glob("pipeline_summary_*.json"))
    
    if not summary_files:
        logger.warning("No pipeline summary files found!")
        return
    
    logger.info(f"Found {len(summary_files)} pipeline runs")
    
    runs = []
    for summary_file in summary_files:
        summary = load_json(summary_file)
        timestamp = summary.get("pipeline_timestamp", "unknown")
        config = summary.get("configuration", {})
        metrics = summary.get("final_metrics", {}).get("aggregated", {})
        
        runs.append({
            "timestamp": timestamp,
            "file": summary_file.name,
            "rounds": config.get("rounds", "?"),
            "epochs": config.get("epochs_per_round", "?"),
            "lr": config.get("learning_rate", "?"),
            "accuracy": metrics.get("centralized_accuracy", 0),
            "loss": metrics.get("centralized_loss", 0),
        })
    
    # Print comparison table
    print_subheader("Run Comparison")
    print(f"{'Timestamp':<20} {'Rounds':>8} {'Epochs':>8} {'LR':>10} {'Accuracy':>12} {'Loss':>12}")
    print("-" * 80)
    
    for run in runs:
        print(f"{run['timestamp']:<20} {run['rounds']:>8} {run['epochs']:>8} "
              f"{run['lr']:>10} {run['accuracy']:>12.4f} {run['loss']:>12.4f}")
    
    # Find best run
    print_subheader("Best Runs")
    best_acc = max(runs, key=lambda x: x["accuracy"])
    best_loss = min(runs, key=lambda x: x["loss"])
    
    print(f"\nBest Accuracy: {best_acc['timestamp']} ({best_acc['accuracy']:.4f})")
    print(f"Best Loss:     {best_loss['timestamp']} ({best_loss['loss']:.4f})")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze federated learning results"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--history",
        type=str,
        help="Path to training history JSON file"
    )
    group.add_argument(
        "--evaluation",
        "--eval",
        type=str,
        help="Path to evaluation results JSON file"
    )
    group.add_argument(
        "--summary",
        type=str,
        help="Path to pipeline summary JSON file"
    )
    group.add_argument(
        "--comparison",
        type=str,
        help="Directory to compare multiple runs from"
    )
    
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    
    if args.history:
        analyze_history(Path(args.history))
    elif args.evaluation:
        analyze_evaluation(Path(args.evaluation))
    elif args.summary:
        analyze_summary(Path(args.summary))
    elif args.comparison:
        compare_runs(Path(args.comparison))
    else:
        # Default: compare all runs if no argument specified
        if MODELS_DIR.exists():
            compare_runs(MODELS_DIR)
        else:
            logger.error("No models directory found!")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
