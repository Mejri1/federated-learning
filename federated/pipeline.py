"""
Complete Federated Learning Pipeline for Facial Recognition

This script implements a complete end-to-end federated learning pipeline:
1. Data Loading and Preparation
2. Model Initialization
3. Federated Learning with FedAvg
4. Model Evaluation
5. Results Saving and Reporting

Usage:
    python federated/pipeline.py --rounds 5 --epochs 2 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
from datetime import datetime
import logging
from typing import Tuple, Dict

# Set environment variables before importing TensorFlow/Ray
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
from torchvision import models

import flwr as fl

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
os.environ.setdefault("PYTHONPATH", str(ROOT_DIR))

from federated.data_utils import DataConfig, build_dataloaders  # noqa: E402
from federated.run_flwr import (  # noqa: E402
    create_model,
    save_model,
    load_model,
    get_parameters,
    set_parameters,
    train,
    evaluate_model,
    FaceClient,
    client_fn_builder,
    aggregate_accuracy,
    get_evaluate_fn,
    serialize_history,
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


def load_and_analyze_data():
    """Step 1: Load and analyze the federated data."""
    print_section("STEP 1: DATA LOADING AND ANALYSIS")
    
    logger.info(f"Clients: {CLIENT_IDS}")
    logger.info(f"Data directory: {ROOT_DIR / 'data' / 'clients'}")
    
    client_data_info = {}
    total_samples = 0
    
    for client_id in CLIENT_IDS:
        data_dir = ROOT_DIR / "data" / "clients" / client_id
        cfg = DataConfig(data_dir=data_dir, batch_size=16, val_split=0.15, test_split=0.15)
        
        try:
            train_loader, val_loader, test_loader = build_dataloaders(cfg)
            num_classes = len(train_loader.dataset.dataset.classes)  # type: ignore[attr-defined]
            
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            test_size = len(test_loader.dataset)
            total_client_samples = train_size + val_size + test_size
            
            client_data_info[client_id] = {
                "train_samples": train_size,
                "val_samples": val_size,
                "test_samples": test_size,
                "total_samples": total_client_samples,
                "num_classes": num_classes,
            }
            
            total_samples += total_client_samples
            
            logger.info(f"\n{client_id}:")
            logger.info(f"  - Train samples: {train_size}")
            logger.info(f"  - Validation samples: {val_size}")
            logger.info(f"  - Test samples: {test_size}")
            logger.info(f"  - Total: {total_client_samples}")
            logger.info(f"  - Classes: {num_classes}")
            
        except Exception as e:
            logger.error(f"Error loading data for {client_id}: {e}")
    
    logger.info(f"\nTotal samples across all clients: {total_samples}")
    return client_data_info


def initialize_model():
    """Step 2: Initialize the federated model."""
    print_section("STEP 2: MODEL INITIALIZATION")
    
    num_classes = 4  # Based on dataset: abir, jihene, omarbr, omarmej
    model = create_model(num_classes)
    
    logger.info(f"Model: MobileNetV3 Small")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Device: {DEVICE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def run_federated_learning(rounds: int, epochs: int, lr: float):
    """Step 3: Run federated learning with FedAvg."""
    print_section("STEP 3: FEDERATED LEARNING WITH FEDAVG")
    
    logger.info(f"Configuration:")
    logger.info(f"  - Total Rounds: {rounds}")
    logger.info(f"  - Local Epochs per Round: {epochs}")
    logger.info(f"  - Learning Rate: {lr}")
    logger.info(f"  - Number of Clients: {len(CLIENT_IDS)}")
    logger.info(f"  - Aggregation Strategy: FedAvg")
    
    # Build client function
    client_fn = client_fn_builder(epochs, lr)
    
    # Create FedAvg strategy and evaluate function
    print_subsection("Starting Federated Learning Simulation (No Ray)")
    evaluate_fn = get_evaluate_fn()
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=len(CLIENT_IDS),
        min_available_clients=len(CLIENT_IDS),
        evaluate_metrics_aggregation_fn=aggregate_accuracy,
        evaluate_fn=evaluate_fn,
    )
    
    # Run simulation WITHOUT Ray runtime_env (avoids Windows TensorFlow DLL issues)
    # This uses in-process simulation which is more stable
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(CLIENT_IDS),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )
        logger.info("Federated learning simulation completed successfully")
    except Exception as e:
        logger.warning(f"Simulation with default settings failed: {e}")
        logger.info("Retrying with Ray disabled...")
        # Fallback: disable Ray entirely
        import multiprocessing
        os.environ["RAY_memory"] = str(int(multiprocessing.cpu_count() * 100_000_000))
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(CLIENT_IDS),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            ray_init_args={"ignore_reinit_error": True, "log_to_driver": False},
        )
        logger.info("Federated learning simulation completed with Ray fallback")
    
    # Extract final parameters and create model
    final_model = create_model(num_classes=4)
    if evaluate_fn.final_params["weights"] is not None:
        set_parameters(final_model, evaluate_fn.final_params["weights"])
        logger.info("Final model parameters loaded from training.")
    else:
        logger.warning("No parameters captured from training, using initial model.")
    
    print_subsection("Federated Learning Completed")
    
    return history, final_model


def evaluate_federated_model(model: nn.Module):
    """Step 4: Evaluate the federated model."""
    print_section("STEP 4: MODEL EVALUATION")
    
    # Evaluate on centralized test set
    print_subsection("Centralized Test Set Evaluation")
    cfg = DataConfig(
        data_dir=ROOT_DIR / "data" / "normalized",
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
    )
    _, _, test_loader = build_dataloaders(cfg)
    loss, acc = evaluate_model(model, test_loader)
    
    centralized_results = {
        "loss": float(loss),
        "accuracy": float(acc),
        "test_samples": len(test_loader.dataset),
    }
    
    logger.info(f"Centralized Test Results:")
    logger.info(f"  - Loss: {loss:.4f}")
    logger.info(f"  - Accuracy: {acc:.4f}")
    logger.info(f"  - Test Samples: {len(test_loader.dataset)}")
    
    # Evaluate on each client's test set
    print_subsection("Per-Client Test Set Evaluation")
    client_results = {}
    
    for client_id in CLIENT_IDS:
        data_dir = ROOT_DIR / "data" / "clients" / client_id
        cfg = DataConfig(data_dir=data_dir, batch_size=32, val_split=0.15, test_split=0.15)
        _, _, test_loader = build_dataloaders(cfg)
        
        loss, acc = evaluate_model(model, test_loader)
        client_results[client_id] = {
            "loss": float(loss),
            "accuracy": float(acc),
            "test_samples": len(test_loader.dataset),
        }
        
        logger.info(f"{client_id}:")
        logger.info(f"  - Loss: {loss:.4f}")
        logger.info(f"  - Accuracy: {acc:.4f}")
        logger.info(f"  - Test Samples: {len(test_loader.dataset)}")
    
    return centralized_results, client_results


def save_results(history, final_model, rounds: int, epochs: int, lr: float,
                 centralized_results: Dict, client_results: Dict):
    """Step 5: Save model, history, and results."""
    print_section("STEP 5: SAVING RESULTS AND ARTIFACTS")
    
    # Save final model
    print_subsection("Saving Federated Model")
    model_filename = f"federated_model_{TIMESTAMP}.pt"
    model_path = save_model(final_model, model_filename)
    logger.info(f"Model checkpoint saved: {model_path}")
    
    # Save training history
    print_subsection("Saving Training History")
    history_payload = serialize_history(history)
    
    # Extract final metrics
    acc_series = history.metrics_centralized.get("accuracy", [])
    loss_series = history.metrics_centralized.get("loss", [])
    
    if acc_series:
        history_payload["final_accuracy"] = acc_series[-1][1]
        history_payload["initial_accuracy"] = acc_series[0][1]
    
    if loss_series:
        history_payload["final_loss"] = loss_series[-1][1]
        history_payload["initial_loss"] = loss_series[0][1]
    
    # Add metadata
    history_payload["metadata"] = {
        "rounds": rounds,
        "epochs_per_round": epochs,
        "learning_rate": lr,
        "num_clients": len(CLIENT_IDS),
        "clients": CLIENT_IDS,
        "timestamp": TIMESTAMP,
        "device": str(DEVICE),
        "aggregation_strategy": "FedAvg",
    }
    
    history_file = MODELS_DIR / f"federated_history_{TIMESTAMP}.json"
    with history_file.open("w", encoding="utf-8") as fp:
        json.dump(history_payload, fp, indent=2)
    logger.info(f"Training history saved: {history_file}")
    
    # Save evaluation results
    print_subsection("Saving Evaluation Results")
    evaluation_payload = {
        "timestamp": TIMESTAMP,
        "model_path": str(model_path),
        "centralized_evaluation": centralized_results,
        "client_evaluation": client_results,
        "summary": {
            "centralized_accuracy": centralized_results["accuracy"],
            "centralized_loss": centralized_results["loss"],
            "average_client_accuracy": sum(
                r["accuracy"] for r in client_results.values()
            ) / len(client_results),
            "average_client_loss": sum(
                r["loss"] for r in client_results.values()
            ) / len(client_results),
        }
    }
    
    eval_file = MODELS_DIR / f"evaluation_results_{TIMESTAMP}.json"
    with eval_file.open("w", encoding="utf-8") as fp:
        json.dump(evaluation_payload, fp, indent=2)
    logger.info(f"Evaluation results saved: {eval_file}")
    
    # Create pipeline summary
    print_subsection("Creating Pipeline Summary Report")
    
    summary_payload = {
        "pipeline_timestamp": TIMESTAMP,
        "configuration": {
            "rounds": rounds,
            "epochs_per_round": epochs,
            "learning_rate": lr,
            "batch_size": 16,
            "num_clients": len(CLIENT_IDS),
            "clients": CLIENT_IDS,
            "device": str(DEVICE),
            "aggregation_strategy": "FedAvg",
        },
        "artifacts": {
            "model_checkpoint": str(model_path),
            "training_history": str(history_file),
            "evaluation_results": str(eval_file),
        },
        "final_metrics": {
            "centralized": centralized_results,
            "per_client": client_results,
            "aggregated": evaluation_payload["summary"],
        },
        "training_progress": history_payload.get("metrics_centralized", {}),
    }
    
    if acc_series:
        summary_payload["accuracy_progress"] = [
            {"round": rnd, "value": float(val)} for rnd, val in acc_series
        ]
    
    if loss_series:
        summary_payload["loss_progress"] = [
            {"round": rnd, "value": float(val)} for rnd, val in loss_series
        ]
    
    summary_file = MODELS_DIR / f"pipeline_summary_{TIMESTAMP}.json"
    with summary_file.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)
    logger.info(f"Pipeline summary saved: {summary_file}")
    
    return model_path, history_file, eval_file, summary_file


def print_final_report(rounds: int, epochs: int, lr: float,
                      history, centralized_results: Dict, client_results: Dict):
    """Print comprehensive final report."""
    print_section("FINAL PIPELINE REPORT")
    
    logger.info("Configuration Summary:")
    logger.info(f"  - Federated Rounds: {rounds}")
    logger.info(f"  - Local Epochs per Round: {epochs}")
    logger.info(f"  - Learning Rate: {lr}")
    logger.info(f"  - Aggregation Strategy: FedAvg")
    logger.info(f"  - Number of Clients: {len(CLIENT_IDS)}")
    
    # Training metrics
    acc_series = history.metrics_centralized.get("accuracy", [])
    loss_series = history.metrics_centralized.get("loss", [])
    
    if acc_series and loss_series:
        logger.info("\nTraining Metrics:")
        logger.info(f"  - Initial Accuracy: {acc_series[0][1]:.4f}")
        logger.info(f"  - Final Accuracy: {acc_series[-1][1]:.4f}")
        logger.info(f"  - Accuracy Improvement: {(acc_series[-1][1] - acc_series[0][1])*100:+.2f}%")
        logger.info(f"  - Initial Loss: {loss_series[0][1]:.4f}")
        logger.info(f"  - Final Loss: {loss_series[-1][1]:.4f}")
        logger.info(f"  - Loss Reduction: {(loss_series[0][1] - loss_series[-1][1])*100:.2f}%")
    
    # Evaluation results
    logger.info("\nEvaluation Results:")
    logger.info(f"  - Centralized Test Accuracy: {centralized_results['accuracy']:.4f}")
    logger.info(f"  - Centralized Test Loss: {centralized_results['loss']:.4f}")
    
    avg_client_acc = sum(r["accuracy"] for r in client_results.values()) / len(client_results)
    logger.info(f"  - Average Client Accuracy: {avg_client_acc:.4f}")
    
    logger.info("\nPer-Client Results:")
    for client_id, results in client_results.items():
        logger.info(f"  {client_id}:")
        logger.info(f"    - Accuracy: {results['accuracy']:.4f}")
        logger.info(f"    - Loss: {results['loss']:.4f}")
    
    logger.info("\nAll artifacts saved to: " + str(MODELS_DIR))
    logger.info("="*80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete Federated Learning Pipeline for Facial Recognition"
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
        default=2,
        help="Local training epochs per round (default: 2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Local learning rate (default: 1e-4)"
    )
    return parser.parse_args()


def main():
    """Execute complete federated learning pipeline."""
    args = parse_args()
    
    print_section("FEDERATED LEARNING COMPLETE PIPELINE")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Device: {DEVICE}")
    
    try:
        # Step 1: Load and analyze data
        client_data_info = load_and_analyze_data()
        
        # Step 2: Initialize model
        model = initialize_model()
        
        # Step 3: Run federated learning
        history, final_model = run_federated_learning(args.rounds, args.epochs, args.lr)
        
        # Step 4: Evaluate model
        centralized_results, client_results = evaluate_federated_model(final_model)
        
        # Step 5: Save results
        model_path, history_file, eval_file, summary_file = save_results(
            history, final_model, args.rounds, args.epochs, args.lr,
            centralized_results, client_results
        )
        
        # Print final report
        print_final_report(args.rounds, args.epochs, args.lr,
                          history, centralized_results, client_results)
        
        logger.info(f"\n✓ Pipeline completed successfully!")
        logger.info(f"✓ Model saved to: {model_path}")
        logger.info(f"✓ History saved to: {history_file}")
        logger.info(f"✓ Results saved to: {eval_file}")
        logger.info(f"✓ Summary saved to: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
