import numpy as np
import json
import os
import pickle
import sys
from utils import *


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUM_CLIENTS = 10
ROUNDS = 1000
N_VALUES = [2, 5, 10]
BATCH_SIZE = 50

STOCHASTIC_BASE_LR_CLIENT0 = 1.0
STOCHASTIC_BASE_LR_OTHER = 0.01
STOCHASTIC_LR_OFFSET = 10

FINAL_AVERAGING_ROUNDS = 100

INITIAL_THETA_MEAN = 0
INITIAL_THETA_STD = 20
INITIAL_THETA_SIZE = 3

BATCH_SEED_BASE = 42
BATCH_SEED_ROUND_MULTIPLIER = 1000
BATCH_SEED_CLIENT_MULTIPLIER = 100

TAPERING_EXPONENTS = [3.0/4.0, 3.3/4.0, 3.6/4.0, 3.9/4.0, 4.0/4.0, 4.3/4.0, 4.6/4.0, 5.0/4.0, 6.0/4.0, 8.0/4.0]

HETEROGENEITY_DATASETS = {
    'combined_hetero': {
        'dir': '../data_generation/federated_datasets_combined_hetero',
        'progressive': [
            'normal_theta_5_xvar_5.json',
            'normal_theta_10_xvar_10.json',
            'normal_theta_15_xvar_15.json',
            'normal_theta_20_xvar_20.json',
            'normal_theta_25_xvar_25.json'
        ],
        'single_hetero': []
    },
    'feature_variance': {
        'dir': '../data_generation/federated_datasets_feature_variance',
        'progressive': [
            'normal_std_5.json',
            'normal_std_10.json',
            'normal_std_15.json',
            'normal_std_20.json',
            'normal_std_25.json'
        ],
        'single_hetero': ['../data_generation/feature_heterogeneity/heterogeneous_normal_std.json']
    },
    'param_variance': {
        'dir': '../data_generation/federated_datasets_param_variance',
        'progressive': [
            'normal_std_5.json',
            'normal_std_10.json',
            'normal_std_15.json',
            'normal_std_20.json',
            'normal_std_25.json'
        ],
        'single_hetero': ['../data_generation/parameter_heterogeneity/heterogeneous_normal_std.json']
    },
    'target_variance': {
        'dir': '../data_generation/federated_datasets_constant_snr',
        'progressive': [
            'normal_snr_5.json',
            'normal_snr_10.json',
            'normal_snr_15.json',
            'normal_snr_20.json',
            'normal_snr_25.json'
        ],
        'single_hetero': ['../data_generation/target_heterogeneity/heterogeneous_normal_snr.json']
    }
}

def load_client_data_with_metadata(dataset_path):
    """Load client data from new JSON format with metadata"""
    print(f"Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        raw_data = json.load(f)
    
    if 'clients_data' not in raw_data or 'metadata' not in raw_data:
        raise ValueError(f"Invalid dataset format in {dataset_path}. Expected 'clients_data' and 'metadata' keys.")
    
    clients_data_raw = raw_data['clients_data']
    metadata = raw_data['metadata']
    
    true_theta = np.array(metadata['single_dominant'], dtype=np.float32)
    
    client0_theta = np.array(clients_data_raw['0']['theta'], dtype=np.float32)
    theta_diff = np.linalg.norm(true_theta - client0_theta)
    if theta_diff > 1e-3:
        print(f"WARNING: single_dominant theta differs from client 0 theta by {theta_diff:.6f}")
    else:
        print(f"✓ single_dominant theta matches client 0 theta (diff: {theta_diff:.6f})")
    
    def format_hetero_value(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    
    hetero_theta = metadata.get('heterogeneity_theta', 'N/A')
    hetero_feature = metadata.get('heterogeneity_feature', 'N/A')
    hetero_target = metadata.get('heterogeneity_target', 'N/A')
    
    print(f"Dataset: {metadata.get('description', 'Unknown')} - "
          f"Heterogeneity - θ: {format_hetero_value(hetero_theta)}, "
          f"Feature: {format_hetero_value(hetero_feature)}, "
          f"Target: {format_hetero_value(hetero_target)}")
    
    clients_data = {}
    for client_id, client_data in clients_data_raw.items():
        client_id = int(client_id)
        theta = np.array(client_data['theta'], dtype=np.float32)
        X = np.array(client_data['X'], dtype=np.float32)
        Y = np.array(client_data['Y'], dtype=np.float32)
        length = client_data['length']
        clients_data[client_id] = (theta, X, Y, length)
    
    return clients_data, true_theta, metadata

def create_consistent_batches(X, Y, batch_size, round_num, client_id, epoch_num=0):
    """Create consistent batches using deterministic seeding"""
    seed = BATCH_SEED_BASE + round_num * BATCH_SEED_ROUND_MULTIPLIER + client_id * BATCH_SEED_CLIENT_MULTIPLIER + epoch_num
    rng = np.random.RandomState(seed)
    
    data_size = len(X)
    indices = rng.permutation(data_size)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    num_batches = data_size // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batches.append((X_shuffled[start_idx:end_idx], Y_shuffled[start_idx:end_idx]))
    
    return batches

def process_client_stochastic(client_id, X, Y, theta, base_lr, batch_counter, local_steps, round_num, lr_exponent):
    """Process N local steps for a client using stochastic method with tapered learning rate"""
    all_gradients = []
    current_batch_counter = batch_counter
    
    for epoch in range(local_steps):
        batches = create_consistent_batches(X, Y, BATCH_SIZE, round_num, client_id, epoch)
        
        for batch_idx, (X_batch, Y_batch) in enumerate(batches):
            is_client_zero = client_id == 0
            lr_denominator = (current_batch_counter + STOCHASTIC_LR_OFFSET) ** lr_exponent if is_client_zero else current_batch_counter
            current_lr = base_lr / lr_denominator
            
            theta, gradient = sgd_update(theta, X_batch, Y_batch, current_lr)
            all_gradients.append(gradient)
            current_batch_counter += 1
    
    mean_gradient = np.mean(all_gradients, axis=0) if all_gradients else np.zeros(theta.shape)
    return theta, mean_gradient, current_batch_counter

def run_stochastic_simulation(clients_data, true_theta, lr_exponent, dataset_name=None):
    """Run stochastic simulation for all N values"""
    print(f"Running Stochastic simulation with LR exponent={lr_exponent:.3f}" + 
          (f" on {dataset_name}" if dataset_name else ""))
    
    X0, Y0 = clients_data[0][1], clients_data[0][2]
    data_sizes = np.array([data[3] for data in clients_data.values()])
    data_weights = data_sizes / np.sum(data_sizes)
    
    init_rng = np.random.RandomState(RANDOM_SEED)
    initial_client_thetas = init_rng.normal(INITIAL_THETA_MEAN, INITIAL_THETA_STD, (NUM_CLIENTS, INITIAL_THETA_SIZE))
    
    results = {}
    
    for N in N_VALUES:
        print(f"  Processing N={N} (each client performs {N} local steps)...")
        
        client_thetas = initial_client_thetas.copy()
        client_batch_counters = np.ones(NUM_CLIENTS, dtype=int)
        
        global_theta_trajectory = []
        client0_gradient_trajectory = []
        client0_gradient_norm_trajectory = []
        all_clients_gradient_trajectory = []
        all_clients_gradient_norm_trajectory = []
        parameter_error_trajectory = []
        global_loss_trajectory = []
        
        for r in range(ROUNDS):
            updated_client_thetas = np.zeros_like(client_thetas)
            round_gradients = np.zeros((NUM_CLIENTS, INITIAL_THETA_SIZE))
            round_gradient_norms = np.zeros(NUM_CLIENTS)
            
            for client_id in range(NUM_CLIENTS):
                _, X, Y, _ = clients_data[client_id]
                base_lr = STOCHASTIC_BASE_LR_CLIENT0 if client_id == 0 else STOCHASTIC_BASE_LR_OTHER
                
                theta, gradient, new_batch_counter = process_client_stochastic(
                    client_id, X, Y, client_thetas[client_id].copy(), 
                    base_lr, client_batch_counters[client_id], N, r, lr_exponent
                )
                
                updated_client_thetas[client_id] = theta
                client_batch_counters[client_id] = new_batch_counter
                
                gradient_norm = np.linalg.norm(gradient)
                round_gradients[client_id] = gradient.copy()
                round_gradient_norms[client_id] = gradient_norm
                
                if client_id == 0:
                    client0_gradient_trajectory.append(gradient.copy())
                    client0_gradient_norm_trajectory.append(gradient_norm)
            
            all_clients_gradient_trajectory.append(round_gradients.copy())
            all_clients_gradient_norm_trajectory.append(round_gradient_norms.copy())
            
            global_theta = np.sum(updated_client_thetas * data_weights[:, np.newaxis], axis=0)
            global_theta_trajectory.append(global_theta.copy())
            
            parameter_error = np.linalg.norm(global_theta - true_theta)
            parameter_error_trajectory.append(parameter_error)
            
            global_loss = compute_global_loss(clients_data, global_theta, NUM_CLIENTS)
            global_loss_trajectory.append(global_loss)
            
            client_thetas = np.tile(global_theta, (NUM_CLIENTS, 1))
        
        averaging_rounds = min(FINAL_AVERAGING_ROUNDS, len(global_theta_trajectory))
        final_global_theta = np.mean(global_theta_trajectory[-averaging_rounds:], axis=0)
        final_parameter_error = np.linalg.norm(final_global_theta - true_theta)
        final_client0_gradient = compute_gradient(X0, Y0, final_global_theta)
        final_gradient_norm = np.linalg.norm(final_client0_gradient)
        
        results[N] = {
            'global_theta_trajectory': global_theta_trajectory,
            'client0_gradient_trajectory': client0_gradient_trajectory,
            'client0_gradient_norm_trajectory': client0_gradient_norm_trajectory,
            'all_clients_gradient_trajectory': all_clients_gradient_trajectory,
            'all_clients_gradient_norm_trajectory': all_clients_gradient_norm_trajectory,
            'parameter_error_trajectory': parameter_error_trajectory,
            'global_loss_trajectory': global_loss_trajectory,
            'final_parameter_error': final_parameter_error,
            'final_gradient_norm': final_gradient_norm,
            'final_global_theta': final_global_theta,
            'true_theta': true_theta
        }
        
        print(f"    Final parameter error: {final_parameter_error:.6f}, Final gradient norm: {final_gradient_norm:.6f}")
    
    return results


def run_heterogeneity_study():
    """Run heterogeneity ablation study for all dataset types"""
    print("Starting Heterogeneity Ablation Study...")
    
    base_save_dir = "heterogeneity_study_single_dominant"
    os.makedirs(base_save_dir, exist_ok=True)
    
    lr_exponent = TAPERING_EXPONENTS[0]
    
    for hetero_type, config in HETEROGENEITY_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing {hetero_type.upper()} datasets")
        print(f"{'='*60}")
        
        hetero_save_dir = os.path.join(base_save_dir, hetero_type)
        os.makedirs(hetero_save_dir, exist_ok=True)
        
        if config['progressive']:
            print(f"\nProcessing PROGRESSIVE datasets for {hetero_type}...")
            progressive_results = {}
            
            for dataset_name in config['progressive']:
                dataset_path = os.path.join(config['dir'], dataset_name)
                print(f"Processing: {dataset_name}")
                
                try:
                    clients_data, true_theta, metadata = load_client_data_with_metadata(dataset_path)
                    results = run_stochastic_simulation(clients_data, true_theta, lr_exponent, dataset_name)
                    progressive_results[dataset_name] = results
                except Exception as e:
                    print(f"WARNING: Error processing {dataset_name}: {e}")
                    continue
            
            if progressive_results:
                progressive_save_dir = os.path.join(hetero_save_dir, "progressive")
                with open(f"{progressive_save_dir}_results.pkl", "wb") as f:
                    pickle.dump(progressive_results, f)
                print(f"Progressive results saved to {progressive_save_dir}_results.pkl")
        
        if config['single_hetero']:
            print(f"\nProcessing SINGLE HETEROGENEOUS datasets for {hetero_type}...")
            single_hetero_results = {}
            
            for dataset_path in config['single_hetero']:
                dataset_name = os.path.basename(dataset_path)
                print(f"Processing: {dataset_name}")
                
                try:
                    clients_data, true_theta, metadata = load_client_data_with_metadata(dataset_path)
                    results = run_stochastic_simulation(clients_data, true_theta, lr_exponent, dataset_name)
                    single_hetero_results[dataset_name] = results
                except Exception as e:
                    print(f"WARNING: Error processing {dataset_name}: {e}")
                    continue
            
            if single_hetero_results:
                single_hetero_save_dir = os.path.join(hetero_save_dir, "single_heterogeneous")
                with open(f"{single_hetero_save_dir}_results.pkl", "wb") as f:
                    pickle.dump(single_hetero_results, f)
                print(f"Single heterogeneous results saved to {single_hetero_save_dir}_results.pkl")
    
    print(f"\nHeterogeneity study completed! Results saved in {base_save_dir}/")
    return base_save_dir

def run_tapering_study():
    """Run tapering ablation study"""
    print("Starting Tapering Ablation Study...")
    
    save_dir = "tapering_study_single_dominant"
    os.makedirs(save_dir, exist_ok=True)
    
    default_dataset_path = os.path.join(
        HETEROGENEITY_DATASETS['combined_hetero']['dir'],
        HETEROGENEITY_DATASETS['combined_hetero']['progressive'][0]
    )
    
    clients_data, true_theta, metadata = load_client_data_with_metadata(default_dataset_path)
    print(f"Using dataset: {os.path.basename(default_dataset_path)}")
    
    tapering_results = {}
    
    for lr_exponent in TAPERING_EXPONENTS:
        print(f"\nProcessing LR exponent: {lr_exponent:.3f}")
        results = run_stochastic_simulation(clients_data, true_theta, lr_exponent)
        tapering_results[lr_exponent] = results
    
    with open(f"{save_dir}/tapering_results.pkl", "wb") as f:
        pickle.dump(tapering_results, f)
    print(f"Results saved to {save_dir}/tapering_results.pkl")
    
    return tapering_results


if __name__ == "__main__":
    print("Starting New Ablation Studies for Stochastic Method with Heterogeneity Datasets")
    print("=" * 80)
    
    if sys.stdin.isatty():
        print("Select study to run:")
        print("1. Heterogeneity Study (All types)")
        print("2. Tapering Study")
        print("3. Both Studies")
        
        choice = input("Enter choice (1/2/3): ").strip()
    else:
        choice = "3"
        print("Running in non-interactive mode. Using default choice: 3 (Both Studies)")
        print("Select study to run:")
        print("1. Heterogeneity Study (All types)")
        print("2. Tapering Study")
        print("3. Both Studies")
        print(f"Selected: {choice}")
    
    if choice == "1":
        run_heterogeneity_study()
    elif choice == "2":
        run_tapering_study()
    elif choice == "3":
        run_heterogeneity_study()
        print("\n" + "=" * 80)
        run_tapering_study()
    else:
        print("Invalid choice. Running both studies by default.")
        run_heterogeneity_study()
        print("\n" + "=" * 80)
        run_tapering_study()
    
    print("\nAll ablation studies completed successfully!")