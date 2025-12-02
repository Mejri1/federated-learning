# Federated Learning for Facial Recognition

A comprehensive implementation and comparison of three machine learning approaches for facial recognition: **Local Training**, **Centralized Learning**, and **Federated Learning**.

## 📋 Overview

This project demonstrates privacy-preserving machine learning techniques by comparing three distinct training paradigms:

1. **Local Training**: Independent models trained on each participant's isolated data
2. **Centralized Learning**: Single model trained on aggregated data from all participants
3. **Federated Learning**: Distributed training with parameter aggregation (FedAvg) without centralizing raw data

## 🎯 Key Features

- ✅ **Privacy-First Architecture**: No centralized data collection in federated approach
- ✅ **Comprehensive Comparison**: Performance, privacy, and communication efficiency analysis
- ✅ **Production-Ready**: Working implementation on Windows without distributed frameworks
- ✅ **Reproducible**: Clear pipeline with documented results
- ✅ **Educational**: Well-structured code for learning federated learning concepts

## 📁 Project Structure

```
federated-learning-facial-recognition/
├── federated/
│   ├── pipeline_local.py          # Main federated learning implementation
│   ├── data_utils.py              # Data preprocessing utilities
│   ├── evaluate.py                # Model evaluation functions
│   └── analyze_results.py         # Results analysis
├── notebooks/
│   ├── comparison_three_approaches.ipynb  # Main comparison analysis
│   ├── centralized_baseline.ipynb         # Centralized approach details
│   └── federated_analysis.ipynb           # Federated training analysis
├── scripts/
│   ├── prepare/
│   │   ├── distribute_clients.py  # Data distribution among participants
│   │   └── rename_images.py       # Image preparation utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 4GB+ RAM
- 2GB+ disk space (excluding raw data)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/federated-learning-facial-recognition.git
cd federated-learning-facial-recognition
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Running the Analysis

### Option 1: Quick Comparison (Recommended)
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/comparison_three_approaches.ipynb
```

This notebook contains:
- All three models loaded and ready for inference
- Side-by-side performance comparison
- Privacy and communication efficiency analysis
- Sample inference demonstrations

### Option 2: Federated Learning Training
Run the local federated learning pipeline:
```bash
python -m federated.pipeline_local
```

This will:
- Train 4 federated models over 10 communication rounds
- Aggregate parameters using FedAvg algorithm
- Save trained model and training history
- Output: `models/federated_model_*.pt` and `models/federated_history_*.json`

## 📈 Results Summary

### Performance Comparison

| Approach | Accuracy | Privacy | Communication | Scalability |
|----------|----------|---------|----------------|-------------|
| **Local** | Low (Limited data) | ✓ High | None | Low |
| **Centralized** | ✓ High | ✗ Low | High overhead | Medium |
| **Federated** | ✓ High | ✓ High | Efficient | ✓ High |

### Key Findings

- **Federated Learning** achieves comparable accuracy to centralized approach while preserving participant privacy
- **Local models** underperform due to limited training data per participant
- **Federated approach** requires fewer communication rounds than expected due to FedAvg effectiveness
- **Convergence**: Federated model reached 100% accuracy by round 8

## 🔐 Privacy & Security Considerations

### ✅ What's Private
- ❌ Raw data **never** leaves participant's local machine (Federated approach)
- ✅ Only model parameter updates are communicated
- ✅ No participant data collection or aggregation

### ⚠️ Important Notes
- This implementation demonstrates **proof-of-concept**
- Production deployment requires:
  - Secure parameter server infrastructure
  - Differential privacy mechanisms
  - Communication encryption
  - Access control mechanisms

## 🔧 Technical Details

### Federated Averaging (FedAvg)
```
1. Initialize global model
2. For each communication round:
   a. Distribute model to all participants
   b. Each participant trains locally
   c. Collect updated parameters
   d. Compute weighted average of parameters
   e. Update global model
3. Repeat until convergence
```

### Model Architecture
- **Base Model**: MobileNetV3 Small (1.52M parameters)
- **Input**: 224×224 RGB images
- **Output**: 4-class classification
- **Framework**: PyTorch

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs per Round**: 5
- **Communication Rounds**: 10

## 📚 Dependencies

See `requirements.txt` for full list:
- PyTorch (model training)
- TensorFlow/Keras (centralized baseline)
- scikit-learn (metrics)
- pandas/numpy (data processing)
- Jupyter (notebooks)

## 📖 Documentation

For detailed technical documentation, see `PROJECT_DOCUMENTATION.md`

## 🎓 Educational Purpose

This project is designed for:
- Students learning federated learning concepts
- Researchers evaluating privacy-preserving ML approaches
- Practitioners implementing distributed training pipelines

## ⚖️ License

This project is provided for educational purposes. Please cite appropriately if used in research.

## 📧 Contact

For questions or improvements, please open an issue on GitHub.

## 🙏 Acknowledgments

- MobileNetV3 architecture from TorchVision
- FedAvg algorithm based on McMahan et al. (2017)
- Inspired by federated learning research papers

---

**Note**: This README does not include specific participant information or sensitive data locations. Please configure paths according to your environment before running.
