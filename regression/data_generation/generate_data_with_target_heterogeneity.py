import numpy as np
import json
import os
from joblib import Parallel, delayed
import pandas as pd
from datetime import datetime

def compute_reference_theta(clients_data):
    """Compute the global optimal theta using weighted least squares."""
    # Compute covariance matrix and cross-term
    Cov = np.zeros((3, 3))
    Cov_t = np.zeros(3)

    for client_data in clients_data.values():
        X = np.array(client_data["X"])
        Y = np.array(client_data["Y"])
        
        Cov += X.T @ X
        Cov_t += X.T @ Y

    global_optimal_theta = np.linalg.solve(Cov, Cov_t)
    return global_optimal_theta

def compute_global_theta_weighted(clients_data, num_weighted_clients):
    """Compute global surrogate theta using weighted aggregation (weights: 1, 1/2, 1/3, ..., 1/N)."""
    d = len(next(iter(clients_data.values()))["theta"])  # Dimension of theta
    Cov_global = np.zeros((d, d))
    Cov_theta_global = np.zeros(d)

    for k_str, data in clients_data.items():
        k = int(k_str)
        if k >= num_weighted_clients:
            continue  # Skip unweighted clients

        weight = 1.0 / (k + 1)  # w^(k) = 1 / (k+1)

        X = np.array(data["X"])
        theta_k = np.array(data["theta"])
        n_k = len(X)

        Cov_k = (X.T @ X) / n_k
        Cov_theta_k = Cov_k @ theta_k

        Cov_global += weight * Cov_k
        Cov_theta_global += weight * Cov_theta_k

    theta_global = np.linalg.solve(Cov_global, Cov_theta_global)
    return theta_global

def compute_dominant_reference_thetas(clients_data):
    """Compute single_dominant, dual_dominant, and triple_dominant reference thetas."""
    single_dominant = compute_global_theta_weighted(clients_data, 1)
    dual_dominant = compute_global_theta_weighted(clients_data, 2)
    triple_dominant = compute_global_theta_weighted(clients_data, 3)
    
    return {
        'single_dominant': single_dominant.tolist(),
        'dual_dominant': dual_dominant.tolist(),
        'triple_dominant': triple_dominant.tolist()
    }

def compute_target_heterogeneity(clients_data):
    """Compute target heterogeneity metric: standard deviation of noise levels across clients."""
    noise_levels = []
    for client_id, client_data in clients_data.items():
        if 'client_snr_db' in client_data:
            noise_levels.append(client_data['client_snr_db'])
    
    if len(noise_levels) > 0:
        heterogeneity_target = np.std(noise_levels)
    else:
        heterogeneity_target = 0.0
    
    return heterogeneity_target

def generate_federated_dataset_with_varying_snr(num_clients=10, datapoints_per_client=5000, 
                                               max_snr_db=25.0, min_snr_db=5.0,
                                               constant_theta=None, constant_x_variance=5.0, seed=None):
    """Generate federated dataset where client 0 has max SNR, others have random SNR from {5,10,15,20}."""
    # Use deterministic seeding with RandomState for reproducibility
    base_seed = seed if seed is not None else 42
    rng = np.random.RandomState(base_seed)
    
    # Set constant theta across all clients
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
    theta_global = np.array(constant_theta)
    
    # Generate SNR levels: client 0 gets max_snr_db, others get random choices from {5, 10, 15, 20}
    client_snr_levels = [max_snr_db]
    
    available_snrs = [5, 10, 15, 20]
    for i in range(1, num_clients):
        snr_db = rng.choice(available_snrs)
        client_snr_levels.append(int(snr_db))
    
    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 10000
        client_rng = np.random.RandomState(client_seed)
        
        snr_db = client_snr_levels[client_id]
        
        client_datapoints = client_rng.randint(3000, 6001) if datapoints_per_client == 'variable' else datapoints_per_client
        
        # Constant X variance across all clients
        X_variance = np.array([constant_x_variance, constant_x_variance, constant_x_variance])
        
        X = client_rng.normal(0, np.sqrt(X_variance), (client_datapoints, 3))
        signal = X @ theta_global
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        epsilon = client_rng.normal(0, np.sqrt(target_noise_power), client_datapoints)
        Y = signal + epsilon
        
        return client_id, {
            "theta": theta_global.tolist(),
            "X": X.tolist(),
            "Y": Y.tolist(),
            "length": len(X),
            "X_variance": X_variance.tolist(),
            "client_snr_db": snr_db
        }

    clients_data = dict(Parallel(n_jobs=-1)(
        delayed(generate_client_data)(cid) for cid in range(num_clients)
    ))
    
    target_heterogeneity = compute_target_heterogeneity(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    
    return {
        'clients_data': clients_data,
        'heterogeneity_target': target_heterogeneity,
        'reference_theta': reference_theta.tolist(),
        'single_dominant': dominant_thetas['single_dominant'],
        'dual_dominant': dominant_thetas['dual_dominant'],
        'triple_dominant': dominant_thetas['triple_dominant'],
        'max_snr_db': max_snr_db,
        'min_snr_db': min_snr_db,
        'client_snr_levels': client_snr_levels,
        'distribution': 'normal',
        'constant_theta': constant_theta,
        'constant_x_variance': constant_x_variance,
        'num_clients': num_clients,
        'total_datapoints': sum(client['length'] for client in clients_data.values())
    }

def create_heterogeneous_target_datasets(output_dir="./target_heterogeneity", num_clients=10, datapoints_per_client=5000, 
                                        max_snr_db=25.0, min_snr_db=5.0, constant_theta=None, 
                                        constant_x_variance=5.0, seed=42):
    """Create heterogeneous normal dataset with varying client SNR levels."""
    # Set default theta if not provided
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
        
    print(f"Generating heterogeneous dataset: Client 0 SNR={max_snr_db}dB, others from [5,10,15,20]dB")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    normal_dataset = generate_federated_dataset_with_varying_snr(
        num_clients=num_clients,
        datapoints_per_client=datapoints_per_client,
        max_snr_db=max_snr_db,
        min_snr_db=min_snr_db,
        constant_theta=constant_theta,
        constant_x_variance=constant_x_variance,
        seed=seed
    )
    
    normal_filename = os.path.join(output_dir, "heterogeneous_normal_snr.json")
    normal_save_data = {
        'metadata': {
            'description': 'Heterogeneous normal dataset with discrete client SNR levels',
            'distribution': 'normal',
            'max_snr_db': max_snr_db,
            'min_snr_db': min_snr_db,
            'client_snr_levels': normal_dataset['client_snr_levels'],
            'available_snrs_for_others': [5, 10, 15, 20],
            'client_0_snr_db': max_snr_db,
            'constant_theta': normal_dataset['constant_theta'],
            'constant_x_variance': constant_x_variance,
            'reference_theta': normal_dataset['reference_theta'],
            'single_dominant': normal_dataset['single_dominant'],
            'dual_dominant': normal_dataset['dual_dominant'],
            'triple_dominant': normal_dataset['triple_dominant'],
            'num_clients': num_clients,
            'datapoints_per_client': datapoints_per_client,
            'heterogeneity_target': normal_dataset['heterogeneity_target'],
            'total_datapoints': normal_dataset['total_datapoints'],
            'generation_time': datetime.now().isoformat(),
            'note': 'Client 0 has SNR=25dB, others have SNR from {5, 10, 15, 20}, theta and X are constant'
        },
        'clients_data': {str(k): v for k, v in normal_dataset['clients_data'].items()}
    }
    
    with open(normal_filename, 'w') as f:
        json.dump(normal_save_data, f, indent=2)
    
    print(f"Saved: {normal_filename} (heterogeneity: {normal_dataset['heterogeneity_target']:.4f})")
    
    return normal_dataset

def generate_federated_dataset(num_clients=10, datapoints_per_client=5000, snr_db=20.0, 
                             constant_theta=None, constant_x_variance=5.0, seed=None):
    """Generate federated dataset where all clients have the same SNR level."""
    # Use deterministic seeding with RandomState for reproducibility
    base_seed = seed if seed is not None else 42
    
    # Set constant theta across all clients
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
    theta_global = np.array(constant_theta)
    
    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 10000
        rng = np.random.RandomState(client_seed)
        
        # Constant SNR across all clients
        client_snr_db = snr_db
        
        client_datapoints = rng.randint(3000, 6001) if datapoints_per_client == 'variable' else datapoints_per_client
        
        # Constant X variance across all clients
        X_variance = np.array([constant_x_variance, constant_x_variance, constant_x_variance])
        
        X = rng.normal(0, np.sqrt(X_variance), (client_datapoints, 3))
        signal = X @ theta_global
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (client_snr_db / 10))
        epsilon = rng.normal(0, np.sqrt(target_noise_power), client_datapoints)
        Y = signal + epsilon
        
        return client_id, {
            "theta": theta_global.tolist(),
            "X": X.tolist(),
            "Y": Y.tolist(),
            "length": len(X),
            "X_variance": X_variance.tolist(),
            "client_snr_db": client_snr_db
        }

    clients_data = dict(Parallel(n_jobs=-1)(
        delayed(generate_client_data)(cid) for cid in range(num_clients)
    ))
    
    target_heterogeneity = compute_target_heterogeneity(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    
    return {
        'clients_data': clients_data,
        'heterogeneity_target': target_heterogeneity,
        'reference_theta': reference_theta.tolist(),
        'single_dominant': dominant_thetas['single_dominant'],
        'dual_dominant': dominant_thetas['dual_dominant'],
        'triple_dominant': dominant_thetas['triple_dominant'],
        'snr_db': snr_db,
        'distribution': 'normal',
        'constant_theta': constant_theta,
        'constant_x_variance': constant_x_variance,
        'num_clients': num_clients,
        'total_datapoints': sum(client['length'] for client in clients_data.values())
    }

def generate_multiple_federated_datasets(output_dir="federated_datasets_target_hetero", num_clients=10, 
                                       datapoints_per_client=5000, constant_theta=None, 
                                       constant_x_variance=5.0, seed=42):
    """Generate multiple federated datasets with different constant SNR levels from {5,10,15,20,25}."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default theta if not provided
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
    
    # Define discrete SNR levels for each dataset
    snr_levels = [5, 10, 15, 20, 25]
    
    results_summary = []
    
    print(f"Generating {len(snr_levels)} datasets with constant SNR per dataset")
    
    for i, snr_db in enumerate(snr_levels):
        
        dataset_seed = seed + i * 5000
        
        dataset = generate_federated_dataset(
            num_clients=num_clients,
            datapoints_per_client=datapoints_per_client,
            snr_db=snr_db,
            constant_theta=constant_theta,
            constant_x_variance=constant_x_variance,
            seed=dataset_seed
        )
        
        filename = f"normal_snr_{snr_db}.json"
        filepath = os.path.join(output_dir, filename)
        
        save_data = {
            'metadata': {
                'description': 'Federated dataset with constant SNR across all clients',
                'distribution': 'normal',
                'snr_db': snr_db,
                'constant_theta': constant_theta,
                'constant_x_variance': constant_x_variance,
                'reference_theta': dataset['reference_theta'],
                'single_dominant': dataset['single_dominant'],
                'dual_dominant': dataset['dual_dominant'],
                'triple_dominant': dataset['triple_dominant'],
                'num_clients': num_clients,
                'datapoints_per_client': datapoints_per_client,
                'x_distribution': f'N(0, {constant_x_variance})',
                'heterogeneity_target': dataset['heterogeneity_target'],
                'total_datapoints': dataset['total_datapoints'],
                'generation_time': datetime.now().isoformat(),
                'note': f'All clients have constant theta {constant_theta}, X ~ N(0, {constant_x_variance}), and SNR = {snr_db} dB'
            },
            'clients_data': {str(k): v for k, v in dataset['clients_data'].items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        results_summary.append({
            'filename': filename,
            'distribution': 'normal',
            'snr_db': snr_db,
            'heterogeneity_target': dataset['heterogeneity_target']
        })
        

    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_dir, 'federated_datasets_summary.csv'), index=False)
    
    print(f"Generated {len(results_summary)} datasets in {output_dir}")
    print(summary_df[['distribution', 'snr_db', 'heterogeneity_target']].to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    print("Generating federated datasets with constant SNR per dataset")
    
    # Set constant theta and X variance as required
    CONSTANT_THETA = [1.0, 1.0, 1.0]
    CONSTANT_X_VARIANCE = 5.0
    
    summary_df = generate_multiple_federated_datasets(
        output_dir="federated_datasets_constant_snr",
        num_clients=10,
        datapoints_per_client=5000,
        constant_theta=CONSTANT_THETA,
        constant_x_variance=CONSTANT_X_VARIANCE,
        seed=42
    )
    
    print("\nGenerating heterogeneous target dataset")
    
    normal_data = create_heterogeneous_target_datasets(
        output_dir="./target_heterogeneity",
        num_clients=10,
        datapoints_per_client=5000,
        max_snr_db=25.0,
        min_snr_db=5.0,
        constant_theta=CONSTANT_THETA,
        constant_x_variance=CONSTANT_X_VARIANCE,
        seed=42
    ) 