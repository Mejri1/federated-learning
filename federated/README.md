# Federated Learning - Implementation Details

This directory contains the complete federated learning implementation for facial recognition using Flower (FLWR) and FedAvg aggregation.

## File Structure

```
federated/
├── __init__.py                 # Package initialization
├── pipeline.py                 # Complete end-to-end pipeline ⭐ START HERE
├── run_flwr.py                # Core FLWR implementation
├── evaluate.py                # Model evaluation script
├── data_utils.py              # Data loading utilities
├── quickstart.py              # Quick start guide
├── analyze_results.py         # Results analysis utility
└── __pycache__/               # Python cache directory
```

## Quick Reference

### Start Here: Complete Pipeline
```bash
python federated/pipeline.py --rounds 5 --epochs 2 --lr 1e-4
```

### Alternative: Just Training
```bash
python federated/run_flwr.py --rounds 5 --epochs 2 --lr 1e-4
```

### Evaluate Model
```bash
python federated/evaluate.py --model models/federated_model_*.pt
```

### Analyze Results
```bash
python federated/analyze_results.py --summary models/pipeline_summary_*.json
python federated/analyze_results.py --comparison models/
```

### Quick Start Guide
```bash
python federated/quickstart.py
```

## Detailed Component Documentation

### 1. pipeline.py - Main Orchestration ⭐

**Purpose:** Complete end-to-end federated learning workflow

**Key Functions:**
- `load_and_analyze_data()`: Validates data structure and statistics
- `initialize_model()`: Creates MobileNetV3 model
- `run_federated_learning()`: Executes federated learning rounds
- `evaluate_federated_model()`: Comprehensive evaluation
- `save_results()`: Saves all artifacts

**Output Files:**
1. `federated_model_<timestamp>.pt` - Model checkpoint
2. `federated_history_<timestamp>.json` - Training metrics
3. `evaluation_results_<timestamp>.json` - Evaluation metrics
4. `pipeline_summary_<timestamp>.json` - Complete report

**Usage:**
```bash
python federated/pipeline.py [--rounds N] [--epochs N] [--lr FLOAT]
```

### 2. run_flwr.py - Federated Learning Core

**Purpose:** Implements Flower federated learning simulation

**Key Classes:**
- `FaceClient(fl.client.NumPyClient)`: Federated client implementation
  - `fit()`: Local training on client data
  - `evaluate()`: Local evaluation
  - `get_parameters()`: Get model parameters

**Key Functions:**
- `create_model()`: Initialize pretrained MobileNetV3
- `train()`: Local training loop
- `evaluate_model()`: Evaluation with loss and accuracy
- `get_parameters() / set_parameters()`: Parameter serialization
- `client_fn_builder()`: Factory for creating clients
- `aggregate_accuracy()`: FedAvg metric aggregation
- `get_evaluate_fn()`: Server-side evaluation

**Strategy:**
- FedAvg (Federated Averaging)
- Weighted averaging by dataset size
- Server-side evaluation on centralized test set

**Usage:**
```bash
python federated/run_flwr.py [--rounds N] [--epochs N] [--lr FLOAT]
```

### 3. evaluate.py - Model Evaluation

**Purpose:** Standalone evaluation of trained models

**Evaluation Types:**
1. **Centralized Evaluation**: Test on aggregated centralized data
2. **Client-wise Evaluation**: Test on each client's test set separately
3. **Per-class Metrics**: Accuracy breakdown by person

**Key Functions:**
- `evaluate_on_centralized_set()`: Evaluate on normalized data
- `evaluate_on_clients()`: Evaluate on individual clients
- `evaluate_model()`: Core evaluation logic with per-class metrics

**Output:** Comprehensive evaluation report in JSON format

**Usage:**
```bash
python federated/evaluate.py --model models/federated_model_*.pt [--output results.json]
```

### 4. data_utils.py - Data Loading Utilities

**Purpose:** Shared data loading and preprocessing utilities

**Key Classes:**
- `DataConfig`: Configuration dataclass for data loading
- `RGBImageFolder`: Custom ImageFolder ensuring RGB conversion

**Key Functions:**
- `build_transforms()`: Training augmentation pipeline
- `build_eval_transforms()`: Evaluation preprocessing
- `build_dataloaders()`: Create train/val/test DataLoaders
- `_calculate_split_sizes()`: Compute split sizes for train/val/test

**Data Augmentation:**
- Resize to 224×224 (MobileNetV3 input size)
- Random horizontal flip (training only)
- Tensor conversion
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Usage:**
```python
from federated.data_utils import DataConfig, build_dataloaders

config = DataConfig(data_dir="path/to/data", batch_size=16)
train_loader, val_loader, test_loader = build_dataloaders(config)
```

### 5. quickstart.py - Getting Started Guide

**Purpose:** Interactive guide for running different scenarios

**Features:**
- Option descriptions and commands
- Recommended workflow
- Parameter explanations
- Output file descriptions

**Usage:**
```bash
python federated/quickstart.py
```

### 6. analyze_results.py - Results Analysis

**Purpose:** Analyze and visualize training results

**Analysis Types:**
- `--history`: Analyze training history file
- `--evaluation`: Analyze evaluation results
- `--summary`: Analyze pipeline summary
- `--comparison`: Compare multiple runs

**Output:**
- Formatted tables with metrics
- Progress summaries
- Best run identification
- Improvement calculations

**Usage:**
```bash
python federated/analyze_results.py --summary models/pipeline_summary_*.json
python federated/analyze_results.py --comparison models/
```

## Data Format

### Input Data Structure
```
data/clients/
├── client_abir/
│   ├── abir/           # Own face images
│   ├── jihene/         # Other client's images
│   ├── omarbr/
│   └── omarmej/
├── client_jihene/
│   ├── abir/
│   ├── jihene/         # Own face images
│   ├── omarbr/
│   └── omarmej/
└── ... (similar for other clients)

data/normalized/        # Centralized test set
├── abir/
├── jihene/
├── omarbr/
└── omarmej/
```

### Image Format
- Format: JPG/PNG
- Size: Variable (will be resized to 224×224)
- Color Space: RGB (automatic conversion if needed)

## Federated Learning Process

### Flowchart
```
┌─────────────────────────────────────────────────────┐
│ Initialize Global Model (Server)                    │
└────────────────────┬────────────────────────────────┘
                     │
              For each round:
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼────────────────────────▼──┐
    │ Send model to all clients      │
    └───┬────────────────────────┬───┘
        │                        │
        │  ┌──────────────────┐  │  ┌──────────────────┐
        │  │ Client 1 (abir)  │  │  │ Client 2 (jihene)│
        │  │ Train locally    │  │  │ Train locally    │
        │  │ epochs times     │  │  │ epochs times     │
        │  └──────────────────┘  │  └──────────────────┘
        │
        │  ┌──────────────────┐  │  ┌──────────────────┐
        │  │ Client 3 (omarbr)│  │  │ Client 4 (omarmej│
        │  │ Train locally    │  │  │ Train locally    │
        │  │ epochs times     │  │  │ epochs times     │
        │  └──────────────────┘  │  └──────────────────┘
        │                        │
    ┌───┴────────────────────────┴───┐
    │ Receive updated models         │
    │ FedAvg: Weighted average       │
    │ parameters by dataset size     │
    └───┬────────────────────────────┘
        │
    ┌───▼────────────────────────┐
    │ Server-side Evaluation     │
    │ on centralized test set    │
    └───┬────────────────────────┘
        │
        └───────────────────────┬─────────────────────┐
                                │                     │
                        More rounds?          Return final model
                                │                     │
                                └─────────────────────┘
```

### FedAvg Aggregation
```
w_new = Σ(n_i / n_total) * w_i

Where:
- w_i: Model parameters from client i
- n_i: Number of training samples at client i
- n_total: Total training samples across all clients
```

## Model Architecture

### MobileNetV3 Small
- **Base Model**: ImageNet-pretrained MobileNetV3 Small
- **Input**: 224×224 RGB images
- **Output Layer**: Replaced with 4-class classifier
- **Parameters**: ~2.5M total (efficient for edge devices)
- **Inference Speed**: ~10-30ms per image (GPU)

```
Input (3, 224, 224)
  │
  ├─ MobileNetV3 Feature Extractor
  │  └─ 2.3M parameters
  │
  └─ Classification Head
     ├─ Adaptive Average Pool
     ├─ Dropout (0.2)
     └─ Linear (1024 → 4) ← Modified for 4 classes
```

## Hyperparameters

### Default Configuration
```python
# Training
batch_size = 16
learning_rate = 1e-4
optimizer = Adam

# Data Splitting
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Federated Learning
rounds = 5
local_epochs = 2
min_clients = 4 (all clients must participate)

# Model
image_size = (224, 224)
num_classes = 4
model = MobileNetV3_Small (pretrained)
```

## Performance Metrics

### Measured in Results
1. **Accuracy**: Correct predictions / Total predictions
2. **Loss**: Cross-entropy loss
3. **Per-class Accuracy**: Per-person recognition accuracy
4. **Per-round Metrics**: Progress tracking over rounds

### Typical Results
- Initial Accuracy: ~40-50% (random initialization)
- Final Accuracy: ~70-80% (after federated learning)
- Convergence: Usually 3-5 rounds for noticeable improvement

## Output Artifacts

### Model Checkpoint (federated_model_*.pt)
PyTorch model state dictionary. Load with:
```python
model = create_model(num_classes=4)
model.load_state_dict(torch.load("federated_model_*.pt"))
```

### Training History (federated_history_*.json)
Metrics from each federation round:
```json
{
  "metrics_centralized": {
    "accuracy": [[1, 0.45], [2, 0.52], ...],
    "loss": [[1, 1.23], [2, 1.15], ...]
  },
  "final_accuracy": 0.75,
  "metadata": {...}
}
```

### Evaluation Results (evaluation_results_*.json)
Evaluation on different test sets:
```json
{
  "centralized_evaluation": {"loss": 0.65, "accuracy": 0.75},
  "client_evaluation": {
    "client_abir": {"loss": 0.62, "accuracy": 0.78},
    ...
  }
}
```

### Pipeline Summary (pipeline_summary_*.json)
Complete pipeline execution report with all metrics and artifacts.

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size or use CPU
```python
# In DataConfig
DataConfig(batch_size=8, ...)  # Reduce from 16
```

### Issue: "No images found"
**Solution:** Check data directory structure
```bash
# Verify structure
ls data/clients/client_abir/abir/  # Should contain images
ls data/normalized/abir/           # Should contain images
```

### Issue: Model accuracy not improving
**Solution:** Adjust learning rate or increase epochs
```bash
python federated/pipeline.py --lr 5e-5 --epochs 3
```

### Issue: ImportError for flwr
**Solution:** Install dependencies
```bash
pip install flwr[simulation] torch torchvision
```

## Advanced Usage

### Custom Model Architecture
Edit `create_model()` in `run_flwr.py` to use different architectures:
```python
def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, num_classes)
    return model.to(DEVICE)
```

### Custom Aggregation Strategy
Modify `FedAvg` parameters in `run_flwr.py`:
```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Use 50% of clients per round
    fraction_evaluate=1.0,  # Evaluate all clients
    min_fit_clients=2,  # Minimum 2 clients per round
)
```

### Custom Data Splits
Modify `DataConfig` in `pipeline.py`:
```python
config = DataConfig(
    data_dir=data_dir,
    batch_size=32,
    val_split=0.1,
    test_split=0.2,
)
```

## References

- [Flower (FLWR) Documentation](https://flower.ai/)
- [FedAvg: Communication-Efficient Learning of Deep Networks](https://arxiv.org/abs/1602.05629)
- [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/abs/1905.02175)
- [Federated Learning with Non-IID Data](https://arxiv.org/abs/1909.06335)

## Citation

If you use this implementation, please cite:
```bibtex
@misc{fed_facial_recognition,
  title={Federated Learning for Facial Recognition},
  year={2024},
  note={Implementation using Flower (FLWR) and FedAvg}
}
```

---

For questions or issues, check the logs or refer to individual script documentation.
