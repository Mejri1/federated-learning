"""
Comprehensive evaluation script for federated learning models.
Evaluates the trained model on centralized test set and per-client basis.

Usage:
    python federated/evaluate.py --model models/federated_model_*.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
from typing import Dict, List, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
os.environ.setdefault("PYTHONPATH", str(ROOT_DIR))

from federated.data_utils import DataConfig, build_dataloaders  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

CLIENT_IDS = ["client_abir", "client_jihene", "client_omarbr", "client_omarmej"]
MODELS_DIR = ROOT_DIR / "models"


def create_model(num_classes: int) -> nn.Module:
    """Create MobileNetV3 Small model for facial recognition."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model.to(DEVICE)


def load_model(model: nn.Module, filepath: Path) -> nn.Module:
    """Load model from disk."""
    model.load_state_dict(torch.load(filepath, map_location=DEVICE))
    logger.info(f"Model loaded from {filepath}")
    return model


def evaluate_model(model: nn.Module, loader: DataLoader, dataset_name: str = "Dataset") -> Tuple[float, float, Dict]:
    """Evaluate the model on a dataset."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    loss_total = 0.0
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_total += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Per-class accuracy
            for pred, label in zip(preds, y):
                class_idx = label.item()
                if class_idx not in per_class_total:
                    per_class_total[class_idx] = 0
                    per_class_correct[class_idx] = 0
                
                per_class_total[class_idx] += 1
                if pred.item() == class_idx:
                    per_class_correct[class_idx] += 1
    
    avg_loss = loss_total / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    
    # Calculate per-class accuracy
    per_class_acc = {
        str(class_idx): per_class_correct.get(class_idx, 0) / per_class_total.get(class_idx, 1)
        for class_idx in per_class_total
    }
    
    logger.info(f"{dataset_name} Evaluation:")
    logger.info(f"  - Total samples: {total}")
    logger.info(f"  - Loss: {avg_loss:.4f}")
    logger.info(f"  - Overall Accuracy: {avg_acc:.4f}")
    logger.info(f"  - Per-class Accuracy: {per_class_acc}")
    
    return avg_loss, avg_acc, {"per_class_accuracy": per_class_acc, "total_samples": total}


def evaluate_on_clients(model: nn.Module) -> Dict:
    """Evaluate model on each client's test set."""
    logger.info("Evaluating on individual client test sets...")
    client_results = {}
    
    for client_id in CLIENT_IDS:
        logger.info(f"\nEvaluating on {client_id}...")
        data_dir = ROOT_DIR / "data" / "clients" / client_id
        cfg = DataConfig(data_dir=data_dir, batch_size=32, val_split=0.15, test_split=0.15)
        _, _, test_loader = build_dataloaders(cfg)
        
        loss, acc, details = evaluate_model(model, test_loader, f"{client_id} Test Set")
        client_results[client_id] = {
            "loss": float(loss),
            "accuracy": float(acc),
            "details": details
        }
    
    return client_results


def evaluate_on_centralized_set(model: nn.Module) -> Dict:
    """Evaluate model on centralized test set."""
    logger.info("Evaluating on centralized test set...")
    cfg = DataConfig(
        data_dir=ROOT_DIR / "data" / "normalized",
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
    )
    _, _, test_loader = build_dataloaders(cfg)
    
    loss, acc, details = evaluate_model(model, test_loader, "Centralized Test Set")
    
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "details": details
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate federated learning model on various test sets."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model file (.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for evaluation results (default: results_<timestamp>.json)"
    )
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info("="*70)
    logger.info("Starting Model Evaluation")
    logger.info("="*70)
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Device: {DEVICE}")
    logger.info("="*70)
    
    # Load model
    num_classes = 4  # Number of persons in dataset
    model = create_model(num_classes)
    model = load_model(model, model_path)
    
    # Evaluate on centralized set
    logger.info("\n" + "="*70)
    logger.info("CENTRALIZED EVALUATION")
    logger.info("="*70)
    centralized_results = evaluate_on_centralized_set(model)
    
    # Evaluate on client sets
    logger.info("\n" + "="*70)
    logger.info("CLIENT-WISE EVALUATION")
    logger.info("="*70)
    client_results = evaluate_on_clients(model)
    
    # Compile results
    evaluation_results = {
        "model_path": str(model_path),
        "device": str(DEVICE),
        "centralized_evaluation": centralized_results,
        "client_evaluation": client_results,
        "summary": {
            "centralized_accuracy": centralized_results["accuracy"],
            "centralized_loss": centralized_results["loss"],
            "average_client_accuracy": sum(
                client_results[client_id]["accuracy"] for client_id in CLIENT_IDS
            ) / len(CLIENT_IDS),
            "average_client_loss": sum(
                client_results[client_id]["loss"] for client_id in CLIENT_IDS
            ) / len(CLIENT_IDS),
        }
    }
    
    # Save results
    output_file = args.output or MODELS_DIR / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Evaluation results saved to {output_file}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Centralized Test Accuracy: {evaluation_results['summary']['centralized_accuracy']:.4f}")
    logger.info(f"Centralized Test Loss: {evaluation_results['summary']['centralized_loss']:.4f}")
    logger.info(f"Average Client Accuracy: {evaluation_results['summary']['average_client_accuracy']:.4f}")
    logger.info(f"Average Client Loss: {evaluation_results['summary']['average_client_loss']:.4f}")
    logger.info("="*70)
    
    return evaluation_results


if __name__ == "__main__":
    main()
