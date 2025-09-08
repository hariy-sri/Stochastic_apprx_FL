# Federated Learning Regression Experiments

This folder contains scripts to run and visualize federated learning (FL) experiments for linear regression tasks on synthetic multi-client datasets.

## Overview

This framework implements various federated learning algorithms for linear regression tasks, with support for different data heterogeneity scenarios, learning rate schedules, and experimental setups. The experiments focus on understanding the behavior of federated learning in regression settings with varying levels of data heterogeneity and client participation patterns.

## Key Features

- **Multiple FL Algorithms**: Stochastic Approximation, FedAvg, FedProx, and FedNova
- **Synthetic Datasets**: Configurable linear regression datasets with controlled heterogeneity
- **Data Heterogeneity Studies**: Combined heterogeneity, feature variance, parameter variance, and target variance scenarios
- **Ablation Studies**: Single-dominant, no-dominant, and dual-dominant client scenarios
- **Learning Rate Tapering**: Configurable learning rate schedules with tapering support

## Project Structure

```
regression/
├── baselines_comparison_study.py     # Main baseline comparison experiments
├── utils.py                         # Mathematical utilities (SGD, gradients, losses)
├── ablation_study_single_dominant.py # Single dominant client ablation studies
├── ablation_study_no_dominant.py    # No dominant client ablation studies  
├── ablation_study_dual_dominant.py  # Dual dominant client ablation studies
├── plot_baselines.py                # Baseline results plotting
├── plot_single_dominant.py          # Single dominant plotting utilities
├── plot_no_dominant.py              # No dominant plotting utilities
├── plot_dual_dominant.py            # Dual dominant plotting utilities
├── pyproject.toml                   # Project dependencies
```

## Installation

### Prerequisites
- Python >= 3.12
- NumPy >= 1.24.0
- Pandas >= 2.3.1
- Seaborn >= 0.13.2
- Joblib >= 1.5.1

### Setup
1. Clone or navigate to the project directory
2. Install dependencies using uv (recommended) or pip:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install numpy>=1.24.0 pandas>=2.3.1 seaborn>=0.13.2 joblib>=1.5.1
```

## Usage

### Basic Baseline Comparison

Run a comprehensive comparison of federated learning algorithms:

```bash
python baselines_comparison_study.py \
  --json_file ../data_generation/federated_datasets_combined_hetero/normal_theta_5_xvar_5.json \
  --save_dir baselines_results \
  --rounds 2000
```

### Advanced Configuration

```bash
python baselines_comparison_study.py \
  --json_file ../data_generation/federated_datasets_combined_hetero/normal_theta_10_xvar_10.json \
  --save_dir advanced_results \
  --rounds 5000 \
  --no_tapering
```

### Command Line Arguments

#### Core Parameters
- `--json_file`: Path to dataset JSON file (required)
- `--save_dir`: Output directory for results (default: "results")
- `--rounds`: Number of federated learning rounds (default: 1000)
- `--no_tapering`: Disable learning rate tapering (default: tapering enabled)

#### Algorithm Configuration
- **Stochastic SGD**: Base learning rate 1.0 with tapering (exponent 0.76)
- **FedAvg**: Base learning rate 0.01 without tapering
- **FedProx**: Base learning rate 0.01 with proximal term (μ=0.1)
- **FedNova**: Base learning rate 0.01 with normalized averaging

## Supported Algorithms

### 1. Stochastic Approximation
Proposed stochastic federated learning with adaptive learning rate tapering and heterogeneous client participation.

### 2. FedAvg 
Standard federated averaging algorithm with weighted parameter aggregation across clients.

### 3. FedProx 
Adds a proximal term to handle system heterogeneity and non-IID data, controlled by the μ parameter.

### 4. FedNova 
Addresses objective inconsistency in federated optimization through normalized averaging and local step compensation.

## Dataset Format

### Expected JSON Structure
The framework expects JSON files with the following structure:

```json
{
  "clients_data": {
    "client_0": {
      "theta": [1.0, 2.0, 3.0],
      "X": [[1.0, 2.0, 3.0], ...],
      "Y": [10.0, 15.0, ...],
      "length": 100
    },
    "client_1": { ... }
  },
  "metadata": {
    "reference_theta": [1.0, 2.0, 3.0]
  }
}
```

### Dataset Generation
Datasets are generated using the `../data_generation/` module with configurable:
- **Parameter heterogeneity**: Varying true parameters across clients
- **Feature variance**: Different input feature distributions
- **Target variance**: Varying noise levels in target variables
- **Combined heterogeneity**: Multiple sources of heterogeneity

## Experiment Types

### 1. Baseline Comparison (`baselines_comparison_study.py`)
Comprehensive comparison of all algorithms across multiple local step sizes (2, 5, 10). Generates:
- Per-algorithm performance trajectories
- Cross-algorithm comparison plots
- Parameter error analysis (when reference parameters available)
- Gradient norm and convergence analysis

### 2. Ablation Studies

#### Single Dominant Client (`ablation_study_single_dominant.py`)
Studies scenarios where one client has significantly different data characteristics:
- **Heterogeneity Study**: Progressive increase in data heterogeneity
- **Tapering Study**: Effect of learning rate tapering exponents

#### No Dominant Client (`ablation_study_no_dominant.py`)
Balanced scenarios with no single dominant client:
- **Heterogeneity Study**: Uniform heterogeneity across clients
- **Tapering Study**: Tapering effects in balanced settings

#### Dual Dominant Client (`ablation_study_dual_dominant.py`)
Two-client scenarios with competing dominant patterns:
- **Heterogeneity Study**: Dual-source heterogeneity effects
- **Tapering Study**: Tapering in multi-dominant scenarios

## Visualization and Analysis

### Plotting Scripts

#### 1. Baseline Results (`plot_baselines.py`)
```bash
python plot_baselines.py \
  --pickle_file baselines_results/complete_results.pkl \
  --save_dir plotting_output
```

#### 2. Ablation Study Plotting
```bash
# Single dominant
python plot_single_dominant.py \
  heterogeneity_study_single_dominant/combined_hetero \
  --output-dir plots_single_dominant

# No dominant  
python plot_no_dominant.py \
  heterogeneity_study_no_dominant/combined_hetero \
  --output-dir plots_no_dominant

# Dual dominant
python plot_dual_dominant.py \
  heterogeneity_study_dual_dominant/combined_hetero \
  --output-dir plots_dual_dominant
```

### Output Metrics

Each experiment generates comprehensive metrics:
- **Global parameter trajectories**: Evolution of global model parameters
- **Parameter error trajectories**: Distance from true parameters (when available)
- **Loss trajectories**: Global and per-client loss evolution
- **Gradient trajectories**: Gradient norms and directions
- **Convergence analysis**: Final performance and convergence rates

## Configuration and Reproducibility

### Experiment Configuration
Each experiment automatically saves:
- Complete parameter configuration and results (`*.pkl` files)
- Human-readable summaries (`results_summary.txt`)
- Per-round trajectories for all metrics
- Client-specific statistics and data distributions

### Reproducibility
- Fixed random seeds (42) for deterministic results
- Complete parameter logging and configuration tracking
- Consistent batch sampling across experiments

### Key Parameters
- **Number of clients**: 10 (configurable)
- **Local steps**: [2, 5, 10] for baseline comparison
- **Batch size**: 50
- **Initial parameter distribution**: Normal(0, 20²)
- **Learning rate tapering**: Exponent 0.76 (configurable)
- **FedProx μ**: 0.1

