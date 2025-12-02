"""
FLWR simulation script for the four clients (abir, jihene, omarbr, omarmej).
Complete federated learning pipeline with FedAvg, model evaluation, and saving.

Usage:
    python federated/run_flwr.py --rounds 5 --epochs 1 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
from typing import Tuple
import logging
from datetime import datetime

# Prevent TensorFlow loading by disabling its imports in Flower
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
import numpy as np

# Import flwr AFTER setting environment variables
import flwr as fl

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
MODELS_DIR.mkdir(exist_ok=True)

# Timestamp for model versioning
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def create_model(num_classes: int) -> nn.Module:
    """Create MobileNetV3 Small model for facial recognition."""
    logger.info(f"Creating MobileNetV3 Small model with {num_classes} classes")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model.to(DEVICE)


def save_model(model: nn.Module, filename: str) -> Path:
    """Save model to disk."""
    filepath = MODELS_DIR / filename
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(model: nn.Module, filepath: Path) -> nn.Module:
    """Load model from disk."""
    model.load_state_dict(torch.load(filepath, map_location=DEVICE))
    logger.info(f"Model loaded from {filepath}")
    return model


def get_parameters(model: nn.Module):
    """Extract model parameters as numpy arrays."""
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters) -> None:
    """Set model parameters from numpy arrays."""
    state_dict = model.state_dict()
    params_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
    model.load_state_dict(params_dict, strict=True)


def train(model: nn.Module, loader, epochs: int, lr: float):
    """Train the model locally on client data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(loader)
        total_loss += avg_epoch_loss
        logger.debug(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
    
    return total_loss / epochs


def evaluate_model(model: nn.Module, loader) -> Tuple[float, float]:
    """Evaluate the model on test data."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_total += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    avg_loss = loss_total / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    
    return avg_loss, avg_acc


class FaceClient(fl.client.NumPyClient):
    """Federated client for facial recognition training."""
    
    def __init__(self, client_name: str, epochs: int, lr: float):
        self.client_name = client_name
        logger.info(f"Initializing {client_name}")
        
        # Load client data
        data_dir = ROOT_DIR / "data" / "clients" / client_name
        cfg = DataConfig(data_dir=data_dir, batch_size=16, val_split=0.15, test_split=0.15)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(cfg)
        
        # Get number of classes
        num_classes = len(self.train_loader.dataset.dataset.classes)  # type: ignore[attr-defined]
        logger.info(f"{client_name} has {len(self.train_loader.dataset)} training samples, {num_classes} classes")
        
        self.model = create_model(num_classes)
        self.epochs = epochs
        self.lr = lr

    def get_parameters(self, config):
        """Return current model parameters."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Train the model locally."""
        logger.info(f"{self.client_name} starting training...")
        set_parameters(self.model, parameters)
        train_loss = train(self.model, self.train_loader, self.epochs, self.lr)
        logger.info(f"{self.client_name} training complete - Loss: {train_loss:.4f}")
        return get_parameters(self.model), len(self.train_loader.dataset), {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        """Evaluate the model locally."""
        logger.info(f"{self.client_name} starting evaluation...")
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.test_loader)
        logger.info(f"{self.client_name} evaluation complete - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc), "loss": float(loss)}


def client_fn_builder(epochs: int, lr: float):
    """Build client function for FLWR simulation."""
    def client_fn(cid: str):
        idx = int(cid)
        client_name = CLIENT_IDS[idx]
        return FaceClient(client_name, epochs, lr)

    return client_fn


def aggregate_metrics(metrics, metric_name: str = "accuracy"):
    """Aggregate metrics across clients using weighted averaging."""
    if not metrics:
        return {}
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    weighted_sum = sum(num_examples * m.get(metric_name, 0) for num_examples, m in metrics)
    return {metric_name: float(weighted_sum / total)}


def aggregate_accuracy(metrics):
    """Aggregate accuracy metrics."""
    return aggregate_metrics(metrics, "accuracy")


def aggregate_loss(metrics):
    """Aggregate loss metrics."""
    return aggregate_metrics(metrics, "loss")


def get_evaluate_fn():
    """Create server-side evaluation function using centralized test set."""
    logger.info("Loading centralized test set for server evaluation...")
    cfg = DataConfig(
        data_dir=ROOT_DIR / "data" / "normalized",
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
    )
    _, _, test_loader = build_dataloaders(cfg)
    num_classes = len(test_loader.dataset.dataset.classes)  # type: ignore[attr-defined]
    model = create_model(num_classes)
    
    logger.info(f"Centralized test set size: {len(test_loader.dataset)}")
    
    # Module-level storage for final parameters
    final_params = {"weights": None}

    def evaluate(server_round: int, parameters, config):
        set_parameters(model, parameters)
        loss, acc = evaluate_model(model, test_loader)
        logger.info(f"Server Round {server_round} - Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
        # Store parameters from each round
        final_params["weights"] = parameters
        return loss, {"accuracy": float(acc), "loss": float(loss)}

    evaluate.final_params = final_params
    return evaluate


def serialize_history(history: fl.server.history.History) -> dict:
    """Serialize Flower history to JSON-compatible format."""
    def serialize(metrics_dict):
        return {
            name: [{"round": rnd, "value": float(val)} for rnd, val in series]
            for name, series in metrics_dict.items()
        }

    return {
        "metrics_centralized": serialize(history.metrics_centralized),
        "metrics_distributed": serialize(history.metrics_distributed),
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FLWR simulation for federated facial recognition with FedAvg aggregation."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated learning rounds (default: 5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of local training epochs per round (default: 1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Local learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    return parser.parse_args()


def main():
    """Main federated learning pipeline."""
    args = parse_args()
    
    logger.info("="*70)
    logger.info("Starting Federated Learning Pipeline for Facial Recognition")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  - Rounds: {args.rounds}")
    logger.info(f"  - Local Epochs: {args.epochs}")
    logger.info(f"  - Learning Rate: {args.lr}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Device: {DEVICE}")
    logger.info(f"  - Timestamp: {TIMESTAMP}")
    logger.info("="*70)
    
    # Create output directories
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Build client function
    logger.info(f"Building client function for {len(CLIENT_IDS)} clients...")
    client_fn = client_fn_builder(args.epochs, args.lr)
    
    # Create FedAvg strategy
    logger.info("Initializing FedAvg strategy...")
    evaluate_fn = get_evaluate_fn()
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=len(CLIENT_IDS),
        min_available_clients=len(CLIENT_IDS),
        evaluate_metrics_aggregation_fn=aggregate_accuracy,
        evaluate_fn=evaluate_fn,
    )
    
    # Setup Ray runtime environment
    runtime_env = {
        "working_dir": str(ROOT_DIR),
        "env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", str(ROOT_DIR))},
    }
    
    # Run federated learning simulation
    logger.info("Starting federated learning simulation...")
    logger.info("-"*70)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(CLIENT_IDS),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        ray_init_args={"runtime_env": runtime_env},
    )
    
    logger.info("-"*70)
    logger.info("Federated learning completed!")
    
    # Extract and save results
    logger.info("Processing and saving results...")
    history_payload = serialize_history(history)
    
    # Extract final metrics
    acc_series = history.metrics_centralized.get("accuracy", [])
    loss_series = history.metrics_centralized.get("loss", [])
    
    if acc_series:
        final_acc = acc_series[-1][1]
        history_payload["final_accuracy"] = final_acc
        logger.info(f"Final Accuracy: {final_acc:.4f}")
    
    if loss_series:
        final_loss = loss_series[-1][1]
        history_payload["final_loss"] = final_loss
        logger.info(f"Final Loss: {final_loss:.4f}")
    
    # Add metadata
    history_payload["metadata"] = {
        "rounds": args.rounds,
        "epochs_per_round": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "num_clients": len(CLIENT_IDS),
        "clients": CLIENT_IDS,
        "timestamp": TIMESTAMP,
        "device": str(DEVICE),
        "aggregation_strategy": "FedAvg",
    }
    
    # Save history
    history_file = MODELS_DIR / f"federated_history_{TIMESTAMP}.json"
    with history_file.open("w", encoding="utf-8") as fp:
        json.dump(history_payload, fp, indent=2)
    logger.info(f"Training history saved to {history_file}")
    
    # Create final model with parameters from last round
    logger.info("Creating and saving final federated model...")
    num_classes = 4  # Number of persons in dataset (abir, jihene, omarbr, omarmej)
    final_model = create_model(num_classes)
    
    # Get final parameters from the evaluate function
    if evaluate_fn.final_params["weights"] is not None:
        set_parameters(final_model, evaluate_fn.final_params["weights"])
        logger.info("Final model parameters loaded from training.")
    else:
        logger.warning("No parameters captured from training, using initial model.")
    
    final_model_path = MODELS_DIR / f"federated_model_{TIMESTAMP}.pt"
    save_model(final_model, f"federated_model_{TIMESTAMP}.pt")
    
    # Generate summary report
    logger.info("="*70)
    logger.info("FEDERATED LEARNING SUMMARY")
    logger.info("="*70)
    logger.info(f"Total Rounds: {args.rounds}")
    logger.info(f"Number of Clients: {len(CLIENT_IDS)}")
    logger.info(f"Aggregation Strategy: FedAvg")
    if acc_series:
        logger.info(f"Initial Accuracy: {acc_series[0][1]:.4f}")
        logger.info(f"Final Accuracy: {final_acc:.4f}")
        logger.info(f"Accuracy Improvement: {(final_acc - acc_series[0][1])*100:.2f}%")
    if loss_series:
        logger.info(f"Initial Loss: {loss_series[0][1]:.4f}")
        logger.info(f"Final Loss: {final_loss:.4f}")
        logger.info(f"Loss Reduction: {(loss_series[0][1] - final_loss)*100:.2f}%")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"History saved to: {history_file}")
    logger.info("="*70)
    
    return history, final_model


if __name__ == "__main__":
    main()

