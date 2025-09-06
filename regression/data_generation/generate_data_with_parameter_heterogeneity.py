import numpy as np
import json
import os
from joblib import Parallel, delayed
import pandas as pd
from datetime import datetime

def compute_reference_theta(clients_data):
    """Compute the global optimal theta using weighted least squares."""
    total_datapoints = sum(client_data["length"] for client_data in clients_data.values())

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
    """
    Compute global surrogate theta using weighted aggregation from a subset of clients.
    Weights: client 0 → 1, client 1 → 1/2, client 2 → 1/3, ..., client N-1 → 1/N
    """
    d = len(next(iter(clients_data.values()))["theta"])
    Cov_global = np.zeros((d, d))
    Cov_theta_global = np.zeros(d)

    for k_str, data in clients_data.items():
        k = int(k_str)
        if k >= num_weighted_clients:
            continue

        weight = 1.0 / (k + 1)

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

def compute_theta_heterogeneity(clients_data):
    """Compute theta heterogeneity metric: mean squared deviation of client thetas from global mean."""
    thetas = []
    for client_id, client_data in clients_data.items():
        theta = np.array(client_data["theta"])
        thetas.append(theta)

    thetas = np.array(thetas)
    theta_mean = np.mean(thetas, axis=0)
    heterogeneity_theta = np.mean(np.linalg.norm(thetas - theta_mean, axis=1) ** 2)

    return heterogeneity_theta

def generate_federated_dataset_with_varying_spreads(num_clients=10, datapoints_per_client=5000, 
                                                  max_spread=25.0, 
                                                  snr_db=10, constant_x_variance=5.0, seed=None):
    """
    Generate federated dataset where client 0 has maximum spread and others have varying spreads.
    Client 0 gets std=25, others get random choices from {5, 10, 15, 20}.
    """
    base_seed = seed if seed is not None else 42
    rng = np.random.RandomState(base_seed)
    
    client_spreads = [max_spread]
    available_stds = [5, 10, 15, 20]
    
    for i in range(1, num_clients):
        spread = rng.choice(available_stds)
        client_spreads.append(int(spread))
    
    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 10000
        client_rng = np.random.RandomState(client_seed)
        
        spread = client_spreads[client_id]
        theta_client = client_rng.normal(0, spread, size=3)
        
        client_datapoints = client_rng.randint(3000, 6001) if datapoints_per_client == 'variable' else datapoints_per_client
        
        # Constant X variance across all clients
        X_variance = np.array([constant_x_variance, constant_x_variance, constant_x_variance])
        
        X = client_rng.normal(0, np.sqrt(X_variance), (client_datapoints, 3))
        signal = X @ theta_client
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        epsilon = client_rng.normal(0, np.sqrt(target_noise_power), client_datapoints)
        Y = signal + epsilon
        
        return client_id, {
            "theta": theta_client.tolist(),
            "X": X.tolist(),
            "Y": Y.tolist(),
            "length": len(X),
            "X_variance": X_variance.tolist(),
            "client_spread": spread
        }

    clients_data = dict(Parallel(n_jobs=-1)(
        delayed(generate_client_data)(cid) for cid in range(num_clients)
    ))
    
    theta_heterogeneity = compute_theta_heterogeneity(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    
    return {
        'clients_data': clients_data,
        'heterogeneity_theta': theta_heterogeneity,
        'reference_theta': reference_theta.tolist(),
        'single_dominant': dominant_thetas['single_dominant'],
        'dual_dominant': dominant_thetas['dual_dominant'],
        'triple_dominant': dominant_thetas['triple_dominant'],
        'max_spread': max_spread,
        'client_spreads': client_spreads,
        'distribution': 'normal',
        'num_clients': num_clients,
        'total_datapoints': sum(client['length'] for client in clients_data.values()),
        'constant_x_variance': constant_x_variance
    }

def create_heterogeneous_datasets(output_dir="./parameter_heterogeneity", num_clients=10, datapoints_per_client=5000, 
                                max_spread=25.0, snr_db=10, constant_x_variance=5.0, seed=42):
    """Create heterogeneous normal dataset with varying client spreads."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating heterogeneous dataset: Client 0 std={max_spread}, others from [5,10,15,20]")
    
    normal_dataset = generate_federated_dataset_with_varying_spreads(
        num_clients=num_clients,
        datapoints_per_client=datapoints_per_client,
        max_spread=max_spread,
        snr_db=snr_db,
        constant_x_variance=constant_x_variance,
        seed=seed
    )
    
    normal_filename = os.path.join(output_dir, "heterogeneous_normal_std.json")
    normal_save_data = {
        'metadata': {
            'description': 'Heterogeneous normal dataset with discrete client spreads',
            'distribution': 'normal',
            'max_spread': max_spread,
            'client_spreads': normal_dataset['client_spreads'],
            'available_stds_for_others': [5, 10, 15, 20],
            'client_0_std': max_spread,
            'reference_theta': normal_dataset['reference_theta'],
            'single_dominant': normal_dataset['single_dominant'],
            'dual_dominant': normal_dataset['dual_dominant'],
            'triple_dominant': normal_dataset['triple_dominant'],
            'num_clients': num_clients,
            'datapoints_per_client': datapoints_per_client,
            'snr_db': snr_db,
            'constant_x_variance': constant_x_variance,
            'x_distribution': f'N(0, {constant_x_variance})',
            'heterogeneity_theta': normal_dataset['heterogeneity_theta'],
            'total_datapoints': normal_dataset['total_datapoints'],
            'generation_time': datetime.now().isoformat(),
            'note': 'Client 0 has std=25, others have std from {5, 10, 15, 20}'
        },
        'clients_data': {str(k): v for k, v in normal_dataset['clients_data'].items()}
    }
    
    with open(normal_filename, 'w') as f:
        json.dump(normal_save_data, f, indent=2)
    
    print(f"Saved: {normal_filename}")
    print(f"Theta heterogeneity: {normal_dataset['heterogeneity_theta']:.4f}")
    
    return normal_dataset

def generate_federated_dataset(num_clients=10, datapoints_per_client=5000, spread=5.0, 
                             snr_db=10, constant_x_variance=5.0, seed=None):
    """Generate federated dataset with parameter heterogeneity only (constant X variance)."""
    base_seed = seed if seed is not None else 42
    
    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 10000
        rng = np.random.RandomState(client_seed)
        
        theta_client = rng.normal(0, spread, size=3)
        client_datapoints = rng.randint(3000, 6001) if datapoints_per_client == 'variable' else datapoints_per_client
        
        # Constant X variance across all clients (key difference from feature heterogeneity)
        X_variance = np.array([constant_x_variance, constant_x_variance, constant_x_variance])
        
        X = rng.normal(0, np.sqrt(X_variance), (client_datapoints, 3))
        signal = X @ theta_client
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        epsilon = rng.normal(0, np.sqrt(target_noise_power), client_datapoints)
        Y = signal + epsilon
        
        return client_id, {
            "theta": theta_client.tolist(),
            "X": X.tolist(),
            "Y": Y.tolist(),
            "length": len(X),
            "X_variance": X_variance.tolist()
        }

    clients_data = dict(Parallel(n_jobs=-1)(
        delayed(generate_client_data)(cid) for cid in range(num_clients)
    ))
    
    theta_heterogeneity = compute_theta_heterogeneity(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    
    return {
        'clients_data': clients_data,
        'heterogeneity_theta': theta_heterogeneity,
        'reference_theta': reference_theta.tolist(),
        'single_dominant': dominant_thetas['single_dominant'],
        'dual_dominant': dominant_thetas['dual_dominant'],
        'triple_dominant': dominant_thetas['triple_dominant'],
        'spread': spread,
        'distribution': 'normal',
        'num_clients': num_clients,
        'total_datapoints': sum(client['length'] for client in clients_data.values()),
        'constant_x_variance': constant_x_variance
    }

def generate_multiple_federated_datasets(output_dir="federated_datasets_param_hetero", num_clients=10, 
                                       datapoints_per_client=5000, snr_db=10, constant_x_variance=5.0, seed=42):
    """Generate multiple federated datasets with varying parameter heterogeneity levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    normal_spreads = [5, 10, 15, 20, 25]
    results_summary = []
    
    print(f"Generating parameter-heterogeneity datasets with X ~ N(0, {constant_x_variance})")
    
    for i, spread in enumerate(normal_spreads):
        print(f"Dataset {i+1}/5: Normal std={spread}")
        
        dataset_seed = seed + i * 1000
        
        dataset = generate_federated_dataset(
            num_clients=num_clients,
            datapoints_per_client=datapoints_per_client,
            spread=spread,
            snr_db=snr_db,
            constant_x_variance=constant_x_variance,
            seed=dataset_seed
        )
        
        filename = f"normal_std_{spread}.json"
        filepath = os.path.join(output_dir, filename)
        
        save_data = {
            'metadata': {
                'description': 'Parameter heterogeneity only (constant X variance)',
                'distribution': 'normal',
                'spread': spread,
                'std_dev': spread,
                'reference_theta': dataset['reference_theta'],
                'single_dominant': dataset['single_dominant'],
                'dual_dominant': dataset['dual_dominant'],
                'triple_dominant': dataset['triple_dominant'],
                'num_clients': num_clients,
                'datapoints_per_client': datapoints_per_client,
                'snr_db': snr_db,
                'constant_x_variance': constant_x_variance,
                'x_distribution': f'N(0, {constant_x_variance})',
                'heterogeneity_theta': dataset['heterogeneity_theta'],
                'total_datapoints': dataset['total_datapoints'],
                'generation_time': datetime.now().isoformat(),
                'note': f'All clients have theta ~ N(0, {spread})'
            },
            'clients_data': {str(k): v for k, v in dataset['clients_data'].items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        results_summary.append({
            'filename': filename,
            'distribution': 'normal',
            'spread': spread,
            'heterogeneity_theta': dataset['heterogeneity_theta']
        })
        
        print(f"  Theta heterogeneity: {dataset['heterogeneity_theta']:.4f}")
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_dir, 'parameter_heterogeneity_summary.csv'), index=False)
    
    print(f"Generated {len(results_summary)} datasets in: {output_dir}")
    print("\nTheta heterogeneity summary:")
    print(summary_df[['distribution', 'spread', 'heterogeneity_theta']].to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    CONSTANT_X_VARIANCE = 5.0
    
    print("Generating parameter-heterogeneity datasets (normal distribution)")
    summary_df = generate_multiple_federated_datasets(
        output_dir="federated_datasets_param_variance",
        num_clients=10,
        datapoints_per_client=5000,
        snr_db=10,
        constant_x_variance=CONSTANT_X_VARIANCE,
        seed=42
    )
    
    print("\nGenerating heterogeneous dataset with discrete client spreads")
    normal_data = create_heterogeneous_datasets(
        output_dir="./parameter_heterogeneity",
        num_clients=10,
        datapoints_per_client=5000,
        max_spread=25.0,
        snr_db=10,
        constant_x_variance=CONSTANT_X_VARIANCE,
        seed=42
    )