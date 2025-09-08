# Federated Learning Classification Framework

A comprehensive federated learning framework for classification tasks, supporting multiple algorithms, datasets.

## Overview

This framework implements various federated learning algorithms for classification tasks, with support for different data partitioning strategies, learning rate schedules, and experimental setups.

## Key Features

- **Multiple FL Algorithms**: FedAvg, FedProx, FedNova, and Stochastic FL
- **Diverse Datasets**: MNIST, CIFAR-10, BloodMNIST, OrganSMNIST
- **Data Partitioning**: IID, near-pathological non-IID, single rare class scenarios

## Project Structure

```
classification/
├── main.py                          # Main experiment runner
├── client.py                        # Federated client implementation
├── server.py                        # Federated server implementation
├── strategies.py                    # FL algorithm implementations
├── models.py                        # Neural network architectures
├── dataset.py                       # Dataset loading and partitioning
├── plot_results.py                  # General plotting utilities
├── plot_resutts_single_rare.py      # Single experiment plotting
├── pyproject.toml                  # Project dependencies
├── data/                           # Dataset storage
├── results/                        # Experiment results
│   ├── same_step_size/            # Standard FL experiments
│   ├── single_rare/               # Rare class experiments
│   └── plots/                     # Generated visualizations
```

## Installation

### Prerequisites
- Python >= 3.12
- PyTorch >= 2.8.0
- CUDA (optional, for GPU acceleration)

### Setup
1. Clone or navigate to the project directory
2. Install dependencies using uv (recommended) or pip:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install torch>=2.8.0 torchvision>=0.23.0 medmnist>=3.0.2 seaborn>=0.13.2
```

## Usage

### Basic Experiment

Run a basic federated learning experiment:

```bash
python main.py --dataset cifar10 --algorithm fedavg --experiment-type same_step_size --num-rounds 100
```
```bash
python main.py --dataset cifar10 --algorithm stochastic --experiment-type same_step_size --num-rounds 100 --tapering
```

### Advanced Configuration

```bash
python main.py \
  --dataset bloodmnist \
  --algorithm stochastic \
  --experiment-type single_rare \
  --num-rounds 200 \
  --num-local-epochs 5 \
  --learning-rate 0.01 \
  --batch-size 32 \
  --tapering \
  --single-dominant-lr 0.1 \
  --single-non-dominant-lr 0.01 \
  --device cuda
```

### Command Line Arguments

#### Core Parameters
- `--dataset`: Dataset choice (`mnist`, `cifar10`, `bloodmnist`, `organsmnist`)
- `--algorithm`: FL algorithm (`fedavg`, `fedprox`, `fednova`, `stochastic`)
- `--experiment-type`: Experiment type (`same_step_size`, `single_rare`)
- `--num-rounds`: Number of federated learning rounds (default: 100)
- `--num-local-epochs`: Local training epochs per round (default: 5)
- `--learning-rate`: Base learning rate (default: 0.01)
- `--batch-size`: Training batch size (default: 32)

#### Advanced Parameters
- `--tapering`: Enable learning rate tapering
- `--mu`: Proximal term coefficient for FedProx/FedNova (default: 0.01)
- `--alpha`: Dirichlet distribution parameter for non-IID data (default: 0.1)
- `--partition-type`: Data partitioning (`iid`, `near_pathological`, `single_rare`)
- `--weighted-aggregate`: Use sample-size weighted aggregation (default: true)

#### Single Rare Class Parameters
- `--single-dominant-lr`: Learning rate for rare class client (default: 0.1)
- `--single-non-dominant-lr`: Learning rate for other clients (default: 0.01)
- `--dominant-decay-power`: Decay power for rare client (default: 0.76)
- `--non-dominant-decay-power`: Decay power for other clients (default: 1.0)

#### Stochastic Algorithm Parameters
- `--stochastic-c`: Stochastic tapering coefficient (default: 1.0)
- `--stochastic-delta`: Stochastic tapering delta parameter (default: 0.76)

## Supported Algorithms

### 1. FedAvg (Federated Averaging)
Standard federated averaging algorithm with weighted parameter aggregation.

### 2. FedProx (Federated Proximal)
Adds a proximal term to handle system heterogeneity, controlled by the `--mu` parameter.

### 3. FedNova (Federated Nova)
Addresses objective inconsistency in federated optimization through normalized averaging.

### 4. Stochastic FL
Proposed stochastic federated learning with tapering and heterogeneous learning rate scheduling.

## Datasets and Models

### Supported Datasets
- **MNIST**: 28×28 grayscale handwritten digits (10 classes)
- **CIFAR-10**: 32×32 color images (10 classes)  
- **BloodMNIST**: Medical blood cell images (8 classes)
- **OrganSMNIST**: Medical organ images (11 classes)

### Model Architectures
- **MNIST**: LeNet-5 with ReLU activations
- **CIFAR-10/Medical**: ResNet-9 with Group Normalization and Adaptive Pooling

### Data Partitioning Strategies
- **IID**: Uniform random distribution across clients
- **Near-pathological**: Highly non-IID
- **Single rare**: One client gets rare class distribution, others get remaining classes

## Experiment Types

### 1. Same Step Size (`same_step_size`)
Standard federated learning setup where all clients use the same learning rate schedule. Suitable for comparing different algorithms under identical conditions.

### 2. Single Rare (`single_rare`)
Specialized setup where one client has access to a rare class while others have the remaining classes. Useful for studying federated learning in imbalanced scenarios.

## Visualization and Analysis

### Plotting Scripts

#### 1. General Results (`plot_results.py`)
```bash
python plot_results.py --experiment-type same_step_size --dataset cifar10 --algorithm fedavg
```

#### 2. Single Experiment Plotting (`plot_results_single_rare.py`)
```bash
python plot_single.py --experiment-type single_rare --dataset bloodmnist --rare-class 6
```

## Configuration and Reproducibility

### Experiment Configuration
Each experiment automatically saves:
- Complete parameter configuration (`experiment_config.json`)
- Training history and metrics (`results_*.pkl`)
- Model checkpoints at regular intervals
- Client statistics and data distributions

### Reproducibility
- Fixed random seeds for deterministic results
- Complete parameter logging

