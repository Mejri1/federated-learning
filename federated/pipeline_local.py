"""
Federated Learning Pipeline - Local Implementation (No Ray)

This version runs federated learning locally without Ray/Flower simulation
to avoid Windows/TensorFlow compatibility issues.

Usage:
    python federated/pipeline_local.py --rounds 10 --epochs 5 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
from datetime import datetime
import logging
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
os.environ.setdefault("PYTHONPATH", str(ROOT_DIR))

from federated.data_utils import DataConfig, build_dataloaders
from federated.run_flwr import (
    create_model,
    save_model,
    load_model,
    get_parameters,
    set_parameters,
    train,
    evaluate_model,
)

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
MODELS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str):
    """Print a formatted section header."""
    logger.info("\n" + "="*80)
    logger.info(f" {title}".ljust(80, "="))
    logger.info("="*80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    logger.info("\n" + "-"*80)
    logger.info(f" {title}")
    logger.info("-"*80)


def federated_averaging(client_models: List[nn.Module], client_sizes: List[int]) -> List:
    """
    Implement FedAvg aggregation strategy locally.
    
    Args:
        client_models: List of trained client models
        client_sizes: Number of samples each client trained on
        
    Returns:
        Aggregated model parameters
    """
    # Get parameters from all clients
    client_params = [get_parameters(model) for model in client_models]
    
    # Calculate total samples
    total_samples = sum(client_sizes)
    weights = [size / total_samples for size in client_sizes]
    
    # Perform weighted averaging
    aggregated_params = []
    for param_idx in range(len(client_params[0])):
        # Initialize with zeros (handle numpy arrays)
        avg_param = np.zeros_like(client_params[0][param_idx])
        
        for client_idx, (params, weight) in enumerate(zip(client_params, weights)):
            avg_param = avg_param + params[param_idx] * weight
        
        aggregated_params.append(avg_param)
    
    return aggregated_params


def run_federated_learning_local(rounds: int, epochs: int, lr: float):
    """Run federated learning locally without Ray."""
    print_section("FEDERATED LEARNING - LOCAL MODE")
    
    logger.info(f"Configuration:")
    logger.info(f"  - Total Rounds: {rounds}")
    logger.info(f"  - Local Epochs per Round: {epochs}")
    logger.info(f"  - Learning Rate: {lr}")
    logger.info(f"  - Number of Clients: {len(CLIENT_IDS)}")
    logger.info(f"  - Aggregation Strategy: FedAvg (Local)")
    
    # Initialize global model
    global_model = create_model(num_classes=4)
    global_model.to(DEVICE)
    
    # Store metrics
    history = {
        "rounds": [],
        "metrics_centralized": {"accuracy": [], "loss": []},
        "client_metrics": {client_id: [] for client_id in CLIENT_IDS}
    }
    
    # Load centralized test set
    print_subsection("Loading Centralized Test Set")
    cfg_test = DataConfig(
        data_dir=ROOT_DIR / "data" / "normalized",
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
    )
    _, _, test_loader = build_dataloaders(cfg_test)
    logger.info(f"✓ Centralized test set loaded: {len(test_loader.dataset)} samples")
    
    # Federated learning rounds
    print_subsection("Starting Federated Learning Rounds")
    
    for round_num in range(rounds):
        logger.info(f"\n🔄 Round {round_num + 1}/{rounds}")
        
        # Step 1: Local training on each client
        client_models = []
        client_sizes = []
        round_client_metrics = {}
        
        for client_id in CLIENT_IDS:
            logger.info(f"  • Training {client_id.upper()}...")
            
            # Load client data
            data_dir = ROOT_DIR / "data" / "clients" / client_id
            cfg = DataConfig(data_dir=data_dir, batch_size=16, val_split=0.15, test_split=0.15)
            train_loader, val_loader, test_loader_client = build_dataloaders(cfg)
            
            # Create client model as copy of global model
            client_model = create_model(num_classes=4)
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(DEVICE)
            
            # Train locally
            train(client_model, train_loader, epochs, lr)
            
            # Evaluate on client's test set
            loss, acc = evaluate_model(client_model, test_loader_client)
            round_client_metrics[client_id] = {"accuracy": acc, "loss": loss}
            
            client_models.append(client_model)
            client_sizes.append(len(train_loader.dataset))
            logger.info(f"    Client Acc: {acc:.4f}")
        
        # Step 2: Aggregate models (FedAvg)
        logger.info(f"  • Aggregating models...")
        aggregated_params = federated_averaging(client_models, client_sizes)
        set_parameters(global_model, aggregated_params)
        
        # Step 3: Evaluate global model
        loss, acc = evaluate_model(global_model, test_loader)
        history["rounds"].append(round_num + 1)
        history["metrics_centralized"]["accuracy"].append({"round": round_num + 1, "value": acc})
        history["metrics_centralized"]["loss"].append({"round": round_num + 1, "value": loss})
        
        logger.info(f"  • Global Model - Acc: {acc:.4f}, Loss: {loss:.4f}")
        
        # Store client metrics
        for client_id, metrics in round_client_metrics.items():
            history["client_metrics"][client_id].append({
                "round": round_num + 1,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"]
            })
    
    print_subsection("Federated Learning Completed")
    return history, global_model


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Federated Learning Pipeline")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    try:
        print_section("FEDERATED LEARNING COMPLETE PIPELINE")
        logger.info(f"Timestamp: {TIMESTAMP}")
        logger.info(f"Device: {DEVICE}")
        
        # Step 1: Run federated learning
        history, final_model = run_federated_learning_local(args.rounds, args.epochs, args.lr)
        
        # Step 2: Save results
        print_section("SAVING RESULTS")
        
        model_save_path = MODELS_DIR / f"federated_model_{TIMESTAMP}.pt"
        save_model(final_model, model_save_path)
        logger.info(f"✓ Model saved: {model_save_path.name}")
        
        history_save_path = MODELS_DIR / f"federated_history_{TIMESTAMP}.json"
        with open(history_save_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"✓ History saved: {history_save_path.name}")
        
        # Summary
        print_section("PIPELINE SUMMARY")
        logger.info(f"✓ Training completed successfully!")
        logger.info(f"  - Rounds: {args.rounds}")
        logger.info(f"  - Local Epochs: {args.epochs}")
        logger.info(f"  - Learning Rate: {args.lr}")
        logger.info(f"  - Final Accuracy: {history['metrics_centralized']['accuracy'][-1]['value']:.4f}")
        logger.info(f"  - Final Loss: {history['metrics_centralized']['loss'][-1]['value']:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
