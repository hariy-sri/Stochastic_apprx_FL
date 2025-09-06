import numpy as np
import json
import os
from joblib import Parallel, delayed
try:
    from scipy.spatial.distance import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        """Simple approximation of Wasserstein distance using sorted values."""
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)
        if len(u_sorted) != len(v_sorted):
            min_len = min(len(u_sorted), len(v_sorted))
            u_sorted = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(u_sorted)), u_sorted)
            v_sorted = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(v_sorted)), v_sorted)
        return np.mean(np.abs(u_sorted - v_sorted))

from sklearn.metrics.pairwise import cosine_similarity
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

def compute_feature_heterogeneity(clients_data):
    """Compute feature heterogeneity metric using Wasserstein distance between client feature distributions."""
    client_features = []
    for client_id, client_data in clients_data.items():
        X = np.array(client_data["X"])
        client_features.append(X)
    
    wasserstein_distances = []
    num_features = client_features[0].shape[1]
    
    for i in range(len(client_features)):
        for j in range(i + 1, len(client_features)):
            X_i = client_features[i]
            X_j = client_features[j]
            
            feature_distances = []
            for feature_dim in range(num_features):
                wd = wasserstein_distance(X_i[:, feature_dim], X_j[:, feature_dim])
                feature_distances.append(wd)
            
            avg_distance = np.mean(feature_distances)
            wasserstein_distances.append(avg_distance)
    
    heterogeneity_feature = np.mean(wasserstein_distances)
    return heterogeneity_feature

def generate_federated_dataset_with_varying_x_variance(num_clients=10, datapoints_per_client=5000, 
                                                      max_x_variance=25.0, 
                                                      snr_db=10, constant_theta=None, seed=None):
    """
    Generate federated dataset where client 0 has maximum X variance and others have varying X variances.
    Client 0 gets X_std=25, others get random choices from {5, 10, 15, 20}.
    All clients have the same constant theta.
    """
    base_seed = seed if seed is not None else 42
    rng = np.random.RandomState(base_seed)
    
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
    theta_global = np.array(constant_theta)
    
    client_x_variances = [max_x_variance]
    available_x_stds = [5, 10, 15, 20]
    
    for i in range(1, num_clients):
        x_std = rng.choice(available_x_stds)
        client_x_variances.append(int(x_std))
    
    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 500
        client_rng = np.random.RandomState(client_seed)
        
        x_variance = client_x_variances[client_id]
        client_datapoints = client_rng.randint(3000, 6001) if datapoints_per_client == 'variable' else datapoints_per_client
        
        X_variance = np.array([x_variance, x_variance, x_variance])
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
            "client_x_variance": x_variance
        }

    clients_data = dict(Parallel(n_jobs=-1)(
        delayed(generate_client_data)(cid) for cid in range(num_clients)
    ))
    
    feature_heterogeneity = compute_feature_heterogeneity(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    
    return {
        'clients_data': clients_data,
        'heterogeneity_feature': feature_heterogeneity,
        'reference_theta': reference_theta.tolist(),
        'single_dominant': dominant_thetas['single_dominant'],
        'dual_dominant': dominant_thetas['dual_dominant'],
        'triple_dominant': dominant_thetas['triple_dominant'],
        'max_x_variance': max_x_variance,
        'client_x_variances': client_x_variances,
        'distribution': 'normal',
        'constant_theta': constant_theta,
        'num_clients': num_clients,
        'total_datapoints': sum(client['length'] for client in clients_data.values())
    }

def create_heterogeneous_feature_datasets(output_dir="./feature_heterogeneity", num_clients=10, datapoints_per_client=5000, 
                                         max_x_variance=25.0, snr_db=10, constant_theta=None, seed=42):
    """Create heterogeneous normal dataset with varying client X variances."""
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating heterogeneous dataset: Client 0 X_std={max_x_variance}, others from [5,10,15,20]")
    
    normal_dataset = generate_federated_dataset_with_varying_x_variance(
        num_clients=num_clients,
        datapoints_per_client=datapoints_per_client,
        max_x_variance=max_x_variance,
        snr_db=snr_db,
        constant_theta=constant_theta,
        seed=seed
    )
    
    normal_filename = os.path.join(output_dir, "heterogeneous_normal_std.json")
    normal_save_data = {
        'metadata': {
            'description': 'Heterogeneous normal dataset with discrete client X variances',
            'distribution': 'normal',
            'max_x_variance': max_x_variance,
            'client_x_variances': normal_dataset['client_x_variances'],
            'available_x_stds_for_others': [5, 10, 15, 20],
            'client_0_x_std': max_x_variance,
            'constant_theta': normal_dataset['constant_theta'],
            'reference_theta': normal_dataset['reference_theta'],
            'single_dominant': normal_dataset['single_dominant'],
            'dual_dominant': normal_dataset['dual_dominant'],
            'triple_dominant': normal_dataset['triple_dominant'],
            'num_clients': num_clients,
            'datapoints_per_client': datapoints_per_client,
            'snr_db': snr_db,
            'heterogeneity_feature': normal_dataset['heterogeneity_feature'],
            'total_datapoints': normal_dataset['total_datapoints'],
            'generation_time': datetime.now().isoformat(),
            'note': 'Client 0 has X_std=25, others have X_std from {5, 10, 15, 20}, theta is constant'
        },
        'clients_data': {str(k): v for k, v in normal_dataset['clients_data'].items()}
    }
    
    with open(normal_filename, 'w') as f:
        json.dump(normal_save_data, f, indent=2)
    
    print(f"Saved: {normal_filename}")
    print(f"Feature heterogeneity: {normal_dataset['heterogeneity_feature']:.4f}")
    
    return normal_dataset

def generate_federated_dataset(num_clients=10, datapoints_per_client=5000, x_variance=5.0, 
                             snr_db=10, constant_theta=None, seed=None):
    """Generate federated dataset with feature heterogeneity only. All clients have the same constant theta."""
    base_seed = seed if seed is not None else 42
    
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
    theta_global = np.array(constant_theta)
    
    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 1000
        rng = np.random.RandomState(client_seed)
        
        client_x_variance = rng.uniform(1.0, x_variance)
        client_datapoints = rng.randint(3000, 6001) if datapoints_per_client == 'variable' else datapoints_per_client
        
        X_variance = np.array([client_x_variance, client_x_variance, client_x_variance])
        X = rng.normal(0, np.sqrt(X_variance), (client_datapoints, 3))
        
        signal = X @ theta_global
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        epsilon = rng.normal(0, np.sqrt(target_noise_power), client_datapoints)
        Y = signal + epsilon
        
        return client_id, {
            "theta": theta_global.tolist(),
            "X": X.tolist(),
            "Y": Y.tolist(),
            "length": len(X),
            "X_variance": X_variance.tolist()
        }

    clients_data = dict(Parallel(n_jobs=-1)(
        delayed(generate_client_data)(cid) for cid in range(num_clients)
    ))
    
    feature_heterogeneity = compute_feature_heterogeneity(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    
    return {
        'clients_data': clients_data,
        'heterogeneity_feature': feature_heterogeneity,
        'reference_theta': reference_theta.tolist(),
        'single_dominant': dominant_thetas['single_dominant'],
        'dual_dominant': dominant_thetas['dual_dominant'],
        'triple_dominant': dominant_thetas['triple_dominant'],
        'x_variance': x_variance,
        'distribution': 'normal',
        'constant_theta': constant_theta,
        'num_clients': num_clients,
        'total_datapoints': sum(client['length'] for client in clients_data.values())
    }

def generate_multiple_federated_datasets(output_dir="federated_datasets_feature_hetero", num_clients=10, 
                                       datapoints_per_client=5000, snr_db=10, constant_theta=None, seed=42):
    """Generate multiple federated datasets with varying feature heterogeneity levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    if constant_theta is None:
        constant_theta = [1.0, 1.0, 1.0]
    
    normal_x_variances = [5, 10, 15, 20, 25]
    results_summary = []
    
    print(f"Generating feature-heterogeneity datasets with constant theta: {constant_theta}")
    
    for i, x_var in enumerate(normal_x_variances):
        print(f"Dataset {i+1}/5: Normal X variance {x_var}")
        
        dataset_seed = seed + i * 100000
        
        dataset = generate_federated_dataset(
            num_clients=num_clients,
            datapoints_per_client=datapoints_per_client,
            x_variance=x_var,
            snr_db=snr_db,
            constant_theta=constant_theta,
            seed=dataset_seed
        )
        
        filename = f"normal_std_{x_var}.json"
        filepath = os.path.join(output_dir, filename)
        
        save_data = {
            'metadata': {
                'description': 'Feature heterogeneity only (constant theta)',
                'distribution': 'normal',
                'x_variance': x_var,
                'constant_theta': constant_theta,
                'reference_theta': dataset['reference_theta'],
                'single_dominant': dataset['single_dominant'],
                'dual_dominant': dataset['dual_dominant'],
                'triple_dominant': dataset['triple_dominant'],
                'num_clients': num_clients,
                'datapoints_per_client': datapoints_per_client,
                'snr_db': snr_db,
                'heterogeneity_feature': dataset['heterogeneity_feature'],
                'total_datapoints': dataset['total_datapoints'],
                'generation_time': datetime.now().isoformat(),
                'note': f'All clients have constant theta {constant_theta}, X variance varies up to {x_var}'
            },
            'clients_data': {str(k): v for k, v in dataset['clients_data'].items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        results_summary.append({
            'filename': filename,
            'distribution': 'normal',
            'x_variance': x_var,
            'heterogeneity_feature': dataset['heterogeneity_feature']
        })
        
        print(f"  Feature heterogeneity: {dataset['heterogeneity_feature']:.4f}")
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_dir, 'feature_heterogeneity_summary.csv'), index=False)
    
    print(f"Generated {len(results_summary)} datasets in: {output_dir}")
    print("\nFeature heterogeneity summary:")
    print(summary_df[['distribution', 'x_variance', 'heterogeneity_feature']].to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    CONSTANT_THETA = [1.0, 1.0, 1.0]
    
    print("Generating feature-heterogeneity datasets (normal distribution)")
    summary_df = generate_multiple_federated_datasets(
        output_dir="federated_datasets_feature_variance",
        num_clients=10,
        datapoints_per_client=5000,
        snr_db=10,
        constant_theta=CONSTANT_THETA,
        seed=42
    )
    
    print("\nGenerating heterogeneous feature dataset with discrete X variances")
    normal_data = create_heterogeneous_feature_datasets(
        output_dir="./feature_heterogeneity",
        num_clients=10,
        datapoints_per_client=5000,
        max_x_variance=25.0,
        snr_db=10,
        constant_theta=CONSTANT_THETA,
        seed=42
    )