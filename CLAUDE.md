# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DBond is a Python-based deep learning research project for **mirror-image peptide data storage optimization**. The project implements neural network models to predict peptide bond cleavage in mirror-image peptides (composed of D-amino acids) for tandem mass spectrometry (MS/MS) analysis.

**Core Purpose**: Optimize mirror-image peptide sequence design for data storage by predicting which peptide bonds are likely to break during mass spectrometry, making sequences easier to sequence/read.

## Common Development Commands

### Environment Setup
```bash
# Install PyTorch (select appropriate CUDA version)
pip install torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt

# Alternative: Use Docker environment
docker build -t dbond:latest .
docker run --gpus all -v $(pwd):/workspace -it dbond:latest
```

### Training Models

**Single-label model (DBond-s) - Recommended**:
```bash
python train.dbond_s.py --config dbond_s_config/default.yaml
```

**Multi-label model (DBond-m)**:
```bash
python train.dbond_m.py --config dbond_m_config/default.yaml
```

**Graph Neural Network model** (new extension):
```bash
python graph_transform/scripts/train_graph_model.py --config graph_transform/config/default.yaml
```

### Model Evaluation
```bash
# Single-label evaluation
python evaluate.dbond_s.py --config dbond_s_config/default.yaml

# Multi-label evaluation
python evaluate.dbond_m.py --config dbond_m_config/default.yaml

# Graph model evaluation
python graph_transform/scripts/evaluate_graph_model.py --config graph_transform/config/default.yaml --model_path best_model/graph_transform.pt
```

### Data Processing
```bash
# Convert MGF to CSV for single-label model
python PBCLA/mgf2csv.dbond_s.py --input your_data.mgf --output output_s.csv

# Convert MGF to CSV for multi-label model
python PBCLA/mgf2csv.dbond_m.py --input your_data.mgf --output output_m.csv
```

### Running Tests
```bash
# Test graph model data processing
python graph_transform/scripts/test_multilabel_data.py

# No formal unit test suite currently exists - tests are script-based
```

## Project Architecture

### Core Model Architecture

**Two Main Strategies**:
1. **DBond-s (Single-label)**: Predicts each peptide bond individually (82.42% accuracy - recommended)
2. **DBond-m (Multi-label)**: Predicts all bonds in a peptide sequence simultaneously

**Feature Fusion (4-dimensional input)**:
- **Sequence Features**: D-amino acid sequence (24-character alphabet including padding)
- **State Features**: Charge, m/z, intensity
- **Bond Features**: Bond position in sequence
- **Environment Features**: NCE (normalized collision energy), scan number

**Network Architecture**:
- **Multi-head Self-Attention**: Processes sequence features
- **Numerical Embedding**: Handles state, bond, and environment features
- **MLP**: Final prediction layer

### Key Components

**Core Model Files**:
- [`dbond_s.py`](dbond_s.py): Single-label classification model
- [`dbond_m.py`](dbond_m.py): Multi-label classification model
- [`data_utils_dbond_s.py`](data_utils_dbond_s.py): Data processing for single-label
- [`data_utils_dbond_m.py`](data_utils_dbond_m.py): Data processing for multi-label

**Graph Neural Network Extension** (newly added):
- [`graph_transform/models/graph_transformer.py`](graph_transform/models/graph_transformer.py): Main GNN architecture
- [`graph_transform/data/graph_builder.py`](graph_transform/data/graph_builder.py): Sequence-to-graph conversion
- [`graph_transform/training/trainer.py`](graph_transform/training/trainer.py): Training utilities

### Data Flow

1. **Input**: CSV files with peptide sequences and features
2. **Processing**: Feature embedding and batch preparation
3. **Model**: Transformer-based neural network processing
4. **Output**: Binary predictions for peptide bond cleavage
5. **Application**: Sequence screening for data storage optimization

### Configuration System

**YAML-based configuration** with separate configs for each model type:
- [`dbond_s_config/default.yaml`](dbond_s_config/default.yaml): Single-label model settings
- [`dbond_m_config/default.yaml`](dbond_m_config/default.yaml): Multi-label model settings
- [`graph_transform/config/default.yaml`](graph_transform/config/default.yaml): Graph model settings

**Key Parameters**:
- `hidden_dim: 256`: Model hidden dimension
- `num_heads: 4-8`: Attention heads
- `max_len: 36-100`: Maximum sequence length
- `batch_size: 4096` (training), `32` (inference)

## Important Implementation Notes

### Performance Characteristics
- **DBond-s**: 82.42% accuracy (superior, recommended)
- **DBond-m**: Lower accuracy due to sparse label space
- **Training**: Large batch sizes (4096) for efficiency
- **Memory**: Requires 16GB+ RAM for training

### Data Format Requirements
- Training data in CSV format with specific column names
- Single-label: One row per bond prediction
- Multi-label: One row per peptide with semicolon-separated labels
- Features: sequence, charge, pep_mass, nce, scan_num, rt, intensity, bond_position

### Model Selection Guidance
- **Use DBond-s** for production/real applications (higher accuracy)
- **Use DBond-m** for research/experimental purposes
- **Use Graph Transform** for advanced multi-label classification with GNN

### GPU/CPU Optimization
- CUDA support highly recommended for training
- Large batch processing for efficiency
- Mixed precision training available in graph extension
- Memory-efficient data loading with multiple workers

### Development Workflow
1. Prepare data in CSV format (use PBCLA tools for MGF conversion)
2. Configure model parameters in YAML files
3. Train model using appropriate training script
4. Evaluate with test dataset
5. Use trained model for prediction on new sequences

## Dependencies and Environment

**Core Requirements**:
- Python 3.7+
- PyTorch 1.8+
- CUDA 11.8 (for GPU acceleration)
- scikit-learn, pandas, pyteomics
- tensorboard for monitoring

**Graph Extension Additional**:
- torch-geometric
- networkx
- plotly (for visualization)

**Docker Environment**:
- Based on pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
- Pre-installed dependencies
- GPU support through Docker --gpus flag