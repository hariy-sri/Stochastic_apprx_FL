# Data Generation for Federated Learning Research

## Overview

This repository provides four specialized data generation systems for federated learning research, each designed to study different types of heterogeneity in distributed machine learning scenarios. All processes generate synthetic regression datasets for multi-client federated learning environments with comprehensive theoretical benchmarks for performance evaluation.

The generated datasets follow a linear regression model: **Y = X @ θ + ε**, where each client has potentially different model parameters (θ), data distributions (X), and/or noise levels (ε), depending on the heterogeneity type being studied.


## Data Generation Types

### 1. Combined Heterogeneity Generation
**File:** `generate_data_with_heterogeneity.py`

Generates datasets with **both parameter and feature heterogeneity** simultaneously to study the combined effects of client differences in model parameters (θ) and data distributions (X).

**Configuration:**
- **Dual Heterogeneity**: Varies both θ (model parameters) and X variance across clients
- **Paired Levels**: (θ_spread=5, X_var=2), (10,4), (15,6), (20,8), (25,10)
- **Output**: 5 datasets with increasing heterogeneity levels

**Data Structure:**
- X: Input features (3D, normal distribution with varying variance)
- θ: Model parameters (3D, normal distribution with varying spread)  
- Y: Target values with controlled SNR (10 dB default)
- reference_theta: Global optimal parameters (theoretical benchmark)

### 2. Feature Heterogeneity Generation  
**File:** `generate_data_with_feature_heterogeneity.py`

Generates datasets with **feature heterogeneity only** while keeping model parameters constant across all clients. This isolates the effect of data distribution differences.

**Configuration:**
- **Constant Parameters**: All clients use θ = [1.0, 1.0, 1.0]
- **Varying X Distribution**: Each client has different X variance levels
- **Progressive Mode**: X variance levels [5, 10, 15, 20, 25]
- **Heterogeneous Mode**: Client 0 gets max variance (25), others random from {5, 10, 15, 20}

**Output Directories:**
- `federated_datasets_feature_variance/` - Progressive datasets
- `feature_heterogeneity/` - Heterogeneous dataset

### 3. Parameter Heterogeneity Generation
**File:** `generate_data_with_parameter_heterogeneity.py`

Generates datasets with **parameter heterogeneity only** while keeping feature distributions constant. This isolates the effect of model parameter differences.

**Configuration:**
- **Constant Features**: All clients use X ~ N(0, 5.0)
- **Varying Parameters**: Each client has different θ spread levels
- **Progressive Mode**: θ spreads [5, 10, 15, 20, 25]
- **Heterogeneous Mode**: Client 0 gets max spread (25), others random from {5, 10, 15, 20}

**Output Directories:**
- `federated_datasets_param_variance/` - Progressive datasets
- `parameter_heterogeneity/` - Heterogeneous dataset

### 4. Target Heterogeneity Generation
**File:** `generate_data_with_target_heterogeneity.py`

Generates datasets with **target/noise heterogeneity only** while keeping model parameters and feature distributions constant. This isolates the effect of different noise levels across clients.

**Configuration:**
- **Constant Parameters**: All clients use θ = [1.0, 1.0, 1.0]
- **Constant Features**: All clients use X ~ N(0, 5.0)
- **Varying SNR**: Each client has different signal-to-noise ratio levels
- **Progressive Mode**: SNR variance ranges [5, 10, 15, 20, 25]
- **Heterogeneous Mode**: Client 0 gets SNR=20dB, others random from {5, 8, 12, 15}

**Output Directories:**
- `federated_datasets_target_variance/` - Progressive datasets
- `target_heterogeneity/` - Heterogeneous dataset

## Dataset Structure

All datasets are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "description": "Dataset type description",
    "distribution": "normal", 
    "heterogeneity_theta": 0.1234,
    "heterogeneity_feature": 0.0567,
    "target_heterogeneity": 2.1234,
    "reference_theta": [1.0234, 0.9876, 1.0123],
    "single_dominant": [1.0456, 0.9654, 1.0345],
    "dual_dominant": [1.0345, 0.9765, 1.0234],
    "triple_dominant": [1.0321, 0.9798, 1.0189],
    "num_clients": 10,
    "datapoints_per_client": 5000,
    "snr_db": 10,
    "total_datapoints": 50000,
    "generation_time": "ISO timestamp",
    "note": "Configuration-specific details"
  },
  "clients_data": {
    "0": {
      "theta": [θ₁, θ₂, θ₃],
      "X": [[X features]], 
      "Y": [target values],
      "length": data_point_count,
      "X_variance": [var₁, var₂, var₃],
      "client_snr_db": 15.5
    },
    "1": { ... },
    ...
  }
}
```

### Key Fields

- **`reference_theta`**: Global optimal parameters computed via weighted least squares across all clients
- **`single_dominant`**: Reference theta using only client 0 (weight = 1)
- **`dual_dominant`**: Reference theta using clients 0-1 (weights = 1, 1/2)
- **`triple_dominant`**: Reference theta using clients 0-2 (weights = 1, 1/2, 1/3)
- **`heterogeneity_theta`**: Measure of parameter diversity across clients
- **`heterogeneity_feature`**: Measure of feature distribution diversity across clients
- **`target_heterogeneity`**: Measure of noise level diversity across clients (standard deviation of SNR)
- **`clients_data`**: Individual client datasets with local parameters and data
- **`theta`**: Local model parameters for each client
- **`X`**: Input features for each client
- **`Y`**: Target values for each client
- **`X_variance`**: Variance of input features for each client
- **`client_snr_db`**: Signal-to-noise ratio for each client (target heterogeneity only)

## Reference Theta Computations

### 1. Global Reference Theta
```python
# Global optimal solution using all client data
Cov = Σ(X_i.T @ X_i)     # Sum over all clients i
Cov_t = Σ(X_i.T @ Y_i)   # Sum over all clients i
reference_theta = Cov^(-1) @ Cov_t
```
Theoretical benchmark representing the centralized training solution.

### 2. Weighted Dominant Reference Thetas
```python
# Weighted aggregation with decreasing client importance
# Weights: client 0 → 1, client 1 → 1/2, client 2 → 1/3, etc.
def compute_weighted_theta(num_clients):
    for k in range(num_clients):
        weight = 1.0 / (k + 1)
        Cov_global += weight * Cov_k
        Cov_theta_global += weight * Cov_k @ theta_k
    return solve(Cov_global, Cov_theta_global)
```

- **Single Dominant**: Only client 0 influences the solution
- **Dual Dominant**: Clients 0-1 with weights [1, 1/2]
- **Triple Dominant**: Clients 0-2 with weights [1, 1/2, 1/3]

## Heterogeneity Metrics

### 1. Theta Heterogeneity
```python
np.mean(np.linalg.norm(thetas - theta_mean, axis=1) ** 2)
```
Measures how much client parameters deviate from the global mean.

### 2. Feature Heterogeneity
Mean cosine dissimilarity between client feature means. Measures how different client data distributions are from each other.

### 3. Target Heterogeneity
```python
np.std([client_snr_db for client in clients])
```
Standard deviation of SNR levels across clients. Measures noise level diversity.

## Technical Implementation

### Reproducibility
- **Deterministic Seeding**: Each client gets unique, reproducible seed (base_seed + client_id * 10000)
- **Dataset Separation**: Large seed increments (100,000) between different datasets
- **Parallel Processing**: Uses `joblib.Parallel` for efficient client data generation

### Default Configuration
- **Clients**: 10 per dataset
- **Data Points**: 3,000-6,000 per client (variable) or fixed 5,000
- **SNR**: 10 dB signal-to-noise ratio (except target heterogeneity)
- **Features**: 3-dimensional input space
- **Parameters**: 3-dimensional parameter space

## Usage

### Basic Execution
```bash
python generate_data_with_heterogeneity.py          # Combined heterogeneity
python generate_data_with_feature_heterogeneity.py  # Feature heterogeneity only  
python generate_data_with_parameter_heterogeneity.py # Parameter heterogeneity only
python generate_data_with_target_heterogeneity.py   # Target/noise heterogeneity only
```

### Custom Configuration
Each script can be imported and used with custom parameters:

```python
from generate_data_with_heterogeneity import generate_federated_dataset

dataset = generate_federated_dataset(
    num_clients=20,
    datapoints_per_client=3000,
    theta_spread=10.0,
    x_variance=5.0,
    snr_db=15,
    seed=123
)

# Access different reference solutions
global_optimum = dataset['reference_theta']
client_0_only = dataset['single_dominant']
top_2_weighted = dataset['dual_dominant']
top_3_weighted = dataset['triple_dominant']
```

## Requirements

```python
numpy
scikit-learn
pandas
joblib
```

## Output

The scripts automatically create appropriate output directories and generate:
- Individual dataset JSON files with comprehensive metadata
- Summary CSV files with heterogeneity metrics
- Console output with generation progress and statistics

Each dataset includes four reference theta variants (global, single_dominant, dual_dominant, triple_dominant) for comprehensive benchmarking of federated learning algorithms against theoretical optima and weighted aggregation schemes.
