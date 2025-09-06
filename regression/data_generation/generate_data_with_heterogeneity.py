import numpy as np
import json
import os
from joblib import Parallel, delayed
try:
    from scipy.spatial.distance import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        """Approximate 1D Wasserstein distance via sorted values."""
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)
        if len(u_sorted) != len(v_sorted):
            min_len = min(len(u_sorted), len(v_sorted))
            u_sorted = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(u_sorted)), u_sorted)
            v_sorted = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(v_sorted)), v_sorted)
        return np.mean(np.abs(u_sorted - v_sorted))

import pandas as pd
from datetime import datetime


def compute_heterogeneity_metrics(clients_data):
    """Compute theta MSE from global mean and mean Wasserstein distance across client features."""
    thetas = []
    client_features = []
    for _, client_data in clients_data.items():
        thetas.append(np.array(client_data["theta"]))
        client_features.append(np.array(client_data["X"]))
    thetas = np.array(thetas)
    theta_mean = np.mean(thetas, axis=0)
    heterogeneity_theta = np.mean(np.linalg.norm(thetas - theta_mean, axis=1) ** 2)
    wasserstein_distances = []
    num_features = client_features[0].shape[1]
    for i in range(len(client_features)):
        for j in range(i + 1, len(client_features)):
            X_i, X_j = client_features[i], client_features[j]
            feature_distances = [wasserstein_distance(X_i[:, d], X_j[:, d]) for d in range(num_features)]
            wasserstein_distances.append(np.mean(feature_distances))
    heterogeneity_feature = np.mean(wasserstein_distances)
    return heterogeneity_theta, heterogeneity_feature


def compute_reference_theta(clients_data):
    """Compute global optimal theta using normal equations."""
    Cov = np.zeros((3, 3))
    Cov_t = np.zeros(3)
    for client_data in clients_data.values():
        X = np.array(client_data["X"])
        Y = np.array(client_data["Y"])
        Cov += X.T @ X
        Cov_t += X.T @ Y
    return np.linalg.solve(Cov, Cov_t)


def compute_global_theta_weighted(clients_data, num_weighted_clients):
    """Compute weighted surrogate theta using clients 0..(num_weighted_clients-1) with weights 1/(k+1)."""
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
    return np.linalg.solve(Cov_global, Cov_theta_global)


def compute_dominant_reference_thetas(clients_data):
    """Compute single-, dual-, and triple-dominant reference thetas."""
    single_dominant = compute_global_theta_weighted(clients_data, 1)
    dual_dominant = compute_global_theta_weighted(clients_data, 2)
    triple_dominant = compute_global_theta_weighted(clients_data, 3)
    return {
        "single_dominant": single_dominant.tolist(),
        "dual_dominant": dual_dominant.tolist(),
        "triple_dominant": triple_dominant.tolist(),
    }


def load_and_analyze_existing_data(json_file):
    """Load client data JSON and compute metrics and basic stats."""
    with open(json_file, "r") as f:
        clients_data = json.load(f)
    clients_data_int_keys = {int(k): v for k, v in clients_data.items()}
    hetero_theta, hetero_feature = compute_heterogeneity_metrics(clients_data_int_keys)
    num_clients = len(clients_data)
    total_datapoints = sum(client["length"] for client in clients_data.values())
    thetas = [np.array(client["theta"]) for client in clients_data.values()]
    theta_ranges = [np.max(theta) - np.min(theta) for theta in thetas]
    return {
        "clients_data": clients_data_int_keys,
        "num_clients": num_clients,
        "total_datapoints": total_datapoints,
        "heterogeneity_theta": hetero_theta,
        "heterogeneity_feature": hetero_feature,
        "theta_ranges": theta_ranges,
        "mean_theta_range": np.mean(theta_ranges),
    }


def generate_federated_dataset(
    num_clients=10,
    datapoints_per_client=5000,
    theta_spread=5.0,
    x_variance=2.0,
    snr_db=10,
    seed=None,
):
    """Generate one federated dataset with heterogeneity and metrics."""
    base_seed = seed if seed is not None else 42

    def generate_client_data(client_id):
        client_seed = base_seed + client_id * 5000
        rng = np.random.RandomState(client_seed)
        theta_client = rng.normal(0, theta_spread, size=3)
        client_datapoints = rng.randint(3000, 6001) if datapoints_per_client == "variable" else datapoints_per_client
        X_variance = np.array([x_variance, x_variance, x_variance])
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
            "X_variance": X_variance.tolist(),
        }

    clients_data = dict(Parallel(n_jobs=-1)(delayed(generate_client_data)(cid) for cid in range(num_clients)))
    hetero_theta, hetero_feature = compute_heterogeneity_metrics(clients_data)
    reference_theta = compute_reference_theta(clients_data)
    dominant_thetas = compute_dominant_reference_thetas(clients_data)
    return {
        "clients_data": clients_data,
        "heterogeneity_theta": hetero_theta,
        "heterogeneity_feature": hetero_feature,
        "reference_theta": reference_theta.tolist(),
        "single_dominant": dominant_thetas["single_dominant"],
        "dual_dominant": dominant_thetas["dual_dominant"],
        "triple_dominant": dominant_thetas["triple_dominant"],
        "theta_spread": theta_spread,
        "x_variance": x_variance,
        "distribution": "normal",
        "num_clients": num_clients,
        "total_datapoints": sum(client["length"] for client in clients_data.values()),
    }


def generate_multiple_federated_datasets(
    output_dir="federated_datasets_combined_hetero",
    num_clients=10,
    datapoints_per_client=5000,
    snr_db=10,
    seed=42,
):
    """Generate multiple datasets at paired heterogeneity levels and save a summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    theta_spreads = [5, 10, 15, 20, 25]
    x_variances = [5, 10, 15, 20, 25]
    results_summary = []
    print(f"Generating datasets in '{output_dir}'...")
    for i, (theta_spread, x_variance) in enumerate(zip(theta_spreads, x_variances), start=1):
        dataset_seed = seed + (i - 1) * 100000
        dataset = generate_federated_dataset(
            num_clients=num_clients,
            datapoints_per_client=datapoints_per_client,
            theta_spread=theta_spread,
            x_variance=x_variance,
            snr_db=snr_db,
            seed=dataset_seed,
        )
        filename = f"normal_theta_{theta_spread}_xvar_{x_variance}.json"
        filepath = os.path.join(output_dir, filename)
        save_data = {
            "metadata": {
                "description": "Combined parameter and feature heterogeneity",
                "distribution": "normal",
                "theta_spread": theta_spread,
                "x_variance": x_variance,
                "num_clients": num_clients,
                "datapoints_per_client": datapoints_per_client,
                "snr_db": snr_db,
                "heterogeneity_theta": dataset["heterogeneity_theta"],
                "heterogeneity_feature": dataset["heterogeneity_feature"],
                "reference_theta": dataset["reference_theta"],
                "single_dominant": dataset["single_dominant"],
                "dual_dominant": dataset["dual_dominant"],
                "triple_dominant": dataset["triple_dominant"],
                "total_datapoints": dataset["total_datapoints"],
                "generation_time": datetime.now().isoformat(),
                "note": f"All clients have theta ~ N(0, {theta_spread}), X ~ N(0, {x_variance})",
            },
            "clients_data": {str(k): v for k, v in dataset["clients_data"].items()},
        }
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)
        results_summary.append({
            "filename": filename,
            "distribution": "normal",
            "theta_spread": theta_spread,
            "x_variance": x_variance,
            "heterogeneity_theta": dataset["heterogeneity_theta"],
            "heterogeneity_feature": dataset["heterogeneity_feature"],
        })
        print(f"[{i}/5] Saved '{filename}' (theta_het={dataset['heterogeneity_theta']:.4f}, feature_het={dataset['heterogeneity_feature']:.4f})")
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "heterogeneity_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to '{summary_path}'.")
    return summary_df


def analyze_heterogeneity_progression(summary_df):
    """Print start/end heterogeneity and growth factors."""
    first_row = summary_df.iloc[0]
    last_row = summary_df.iloc[-1]
    theta_growth = last_row["heterogeneity_theta"] / first_row["heterogeneity_theta"]
    feature_growth = last_row["heterogeneity_feature"] / first_row["heterogeneity_feature"]
    print("Heterogeneity progression:")
    print(f"  Theta: {first_row['heterogeneity_theta']:.4f} -> {last_row['heterogeneity_theta']:.4f} ({theta_growth:.2f}x)")
    print(f"  Feature: {first_row['heterogeneity_feature']:.4f} -> {last_row['heterogeneity_feature']:.4f} ({feature_growth:.2f}x)")


if __name__ == "__main__":
    print("Analyzing existing client data...")
    try:
        analysis = load_and_analyze_existing_data("clients_data.json")
        print(f"Clients={analysis['num_clients']}, datapoints={analysis['total_datapoints']}, theta_het={analysis['heterogeneity_theta']:.4f}, feature_het={analysis['heterogeneity_feature']:.4f}, mean_theta_range={analysis['mean_theta_range']:.4f}")
    except FileNotFoundError:
        print("clients_data.json not found. Skipping existing data analysis.")
    print("Generating combined heterogeneity datasets (normal)...")
    summary_df = generate_multiple_federated_datasets(
        output_dir="federated_datasets_combined_hetero",
        num_clients=10,
        datapoints_per_client=5000,
        snr_db=10,
        seed=42,
    )
    analyze_heterogeneity_progression(summary_df)
