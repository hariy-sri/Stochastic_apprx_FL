import numpy as np
import json
import os
import argparse
import pickle
from utils import sgd_update, compute_gradient, compute_loss, compute_global_loss

# Configuration
RANDOM_SEED = 42
NUM_CLIENTS = 10
ROUNDS = 1000
BATCH_SIZE = 50
LOCAL_STEPS_VALUES = [2, 5, 10]  
LOCAL_STEPS = 1

BASE_LR = 1.0
BASE_LR_NO_TAPERING = 0.01
TAPERING = False
LR_OFFSET = 10
LR_EXPONENT = 0.76

FEDPROX_MU = 0.1

INITIAL_THETA_MEAN = 0
INITIAL_THETA_STD = 20
INITIAL_THETA_SIZE = 3

BATCH_SEED_BASE = 42
BATCH_SEED_ROUND_MULTIPLIER = 1000
BATCH_SEED_CLIENT_MULTIPLIER = 100

DTYPE = np.float32

def load_clients_data(json_file):
    """Load and preprocess client data from JSON file"""
    np.random.seed(RANDOM_SEED)
    
    with open(json_file, 'r') as f:
        raw_data = json.load(f)

    if 'clients_data' in raw_data:
        clients_raw_data = raw_data['clients_data']
        metadata = raw_data.get('metadata', {})
    else:
        clients_raw_data = raw_data
        metadata = {}

    true_theta = None
    if 'reference_theta' in metadata:
        true_theta = np.array(metadata['reference_theta'], dtype=DTYPE)

    clients_data = {}
    for client_id, client_data in clients_raw_data.items():
        client_id = int(client_id)
        theta = np.array(client_data['theta'], dtype=DTYPE)
        X = np.array(client_data['X'], dtype=DTYPE)
        Y = np.array(client_data['Y'], dtype=DTYPE)
        length = client_data['length']
        clients_data[client_id] = (theta, X, Y, length)

    data_sizes = np.array([data[3] for data in clients_data.values()], dtype=np.int32)
    total_data_size = np.sum(data_sizes)
    data_weights = (data_sizes / total_data_size).astype(DTYPE)
    
    return clients_data, data_weights, true_theta

def create_consistent_batches(X, Y, batch_size, round_num, client_id, epoch_num=0):
    """Create consistent batches across methods using deterministic seeding"""
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

def get_learning_rate(batch_counter):
    """Compute learning rate with optional tapering"""
    if TAPERING:
        lr_denominator = (batch_counter + LR_OFFSET) ** LR_EXPONENT
        return BASE_LR / lr_denominator
    else:
        return BASE_LR_NO_TAPERING

def get_learning_rate_stochastic(batch_counter):
    """Compute learning rate for stochastic method - always tapers"""
    lr_denominator = (batch_counter + LR_OFFSET) ** LR_EXPONENT
    return BASE_LR / lr_denominator

def generate_varying_local_steps(N_value, round_num, num_clients):
    """Generate consistent local steps with heterogeneous step sizes"""
    if N_value >= 3:
        min_steps = N_value - 2
        max_steps = N_value
    else:
        min_steps = max_steps = N_value
    
    seed = RANDOM_SEED + round_num * 10000 + N_value * 1000
    rng = np.random.RandomState(seed)
    
    if min_steps == max_steps:
        return np.full(num_clients, min_steps)
    else:
        return rng.randint(min_steps, max_steps + 1, size=num_clients)

def run_stochastic(clients_data, data_weights, rounds, true_theta=None):
    """Run stochastic gradient descent"""
    np.random.seed(RANDOM_SEED)
    client_thetas = np.random.normal(INITIAL_THETA_MEAN, INITIAL_THETA_STD, 
                                   (NUM_CLIENTS, INITIAL_THETA_SIZE)).astype(DTYPE)
    
    global_theta_trajectory = []
    individual_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    summed_gradient_trajectory = []
    global_loss_trajectory = []
    local_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    local_thetas_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    client_loss_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    parameter_error_trajectory = []
    
    batch_counters = np.ones(NUM_CLIENTS, dtype=np.int32)
    
    for round_num in range(rounds):
        round_gradients = {}
        
        for client_id in range(NUM_CLIENTS):
            theta_true, X, Y, _ = clients_data[client_id]
            current_theta = client_thetas[client_id]
            
            all_gradients = []
            for epoch in range(LOCAL_STEPS):
                batches = create_consistent_batches(X, Y, BATCH_SIZE, round_num, client_id)
                for batch_idx, (X_batch, Y_batch) in enumerate(batches):
                    lr = get_learning_rate_stochastic(batch_counters[client_id])
                    current_theta, gradient = sgd_update(current_theta, X_batch, Y_batch, lr)
                    all_gradients.append(gradient)
                    batch_counters[client_id] += 1
            
            mean_gradient = np.mean(all_gradients, axis=0) if all_gradients else np.zeros(current_theta.shape)
            round_gradients[client_id] = mean_gradient
            local_gradient_trajectories[client_id].append(mean_gradient.copy())
            local_thetas_trajectories[client_id].append(current_theta.copy())
            client_thetas[client_id] = current_theta
        
        global_theta = np.sum(client_thetas * data_weights[:, np.newaxis], axis=0)
        global_theta_trajectory.append(global_theta.copy())
        
        if true_theta is not None:
            parameter_error = np.linalg.norm(global_theta - true_theta)
            parameter_error_trajectory.append(parameter_error)
        
        individual_gradients = []
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_gradient = compute_gradient(X_client, Y_client, global_theta)
            individual_gradient_trajectories[client_id].append(client_gradient.copy())
            individual_gradients.append(client_gradient)
        
        summed_gradient = np.sum(individual_gradients, axis=0)
        summed_gradient_trajectory.append(summed_gradient.copy())
        
        global_loss = compute_global_loss(clients_data, global_theta, NUM_CLIENTS)
        global_loss_trajectory.append(global_loss)
        
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_loss = compute_loss(X_client, Y_client, global_theta)
            client_loss_trajectories[client_id].append(client_loss)
        
        client_thetas = np.tile(global_theta, (NUM_CLIENTS, 1))
    
    return {
        'global_theta_trajectory': global_theta_trajectory,
        'individual_gradient_trajectories': individual_gradient_trajectories,
        'summed_gradient_trajectory': summed_gradient_trajectory,
        'global_loss_trajectory': global_loss_trajectory,
        'local_gradient_trajectories': local_gradient_trajectories,
        'local_thetas_trajectories': local_thetas_trajectories,
        'client_loss_trajectories': client_loss_trajectories,
        'parameter_error_trajectory': parameter_error_trajectory,
        'true_theta': true_theta
    }

def run_fedavg(clients_data, data_weights, rounds, true_theta=None):
    """Run FedAvg algorithm"""
    np.random.seed(RANDOM_SEED)
    client_thetas = np.random.normal(INITIAL_THETA_MEAN, INITIAL_THETA_STD, 
                                   (NUM_CLIENTS, INITIAL_THETA_SIZE)).astype(DTYPE)
    
    global_theta_trajectory = []
    individual_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    summed_gradient_trajectory = []
    global_loss_trajectory = []
    local_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    local_thetas_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    client_loss_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    parameter_error_trajectory = []
    
    batch_counters = np.ones(NUM_CLIENTS, dtype=np.int32)
    
    for round_num in range(rounds):
        client_local_steps = generate_varying_local_steps(LOCAL_STEPS, round_num, NUM_CLIENTS)
        updated_client_thetas = np.zeros_like(client_thetas)
        round_gradients = {}
        
        for client_id in range(NUM_CLIENTS):
            theta_true, X, Y, _ = clients_data[client_id]
            current_theta = client_thetas[client_id].copy()
            client_local_step_count = client_local_steps[client_id]
            
            all_gradients = []
            for epoch in range(client_local_step_count):
                batches = create_consistent_batches(X, Y, BATCH_SIZE, round_num, client_id)
                for batch_idx, (X_batch, Y_batch) in enumerate(batches):
                    lr = get_learning_rate(batch_counters[client_id])
                    current_theta, gradient = sgd_update(current_theta, X_batch, Y_batch, lr)
                    all_gradients.append(gradient)
                    batch_counters[client_id] += 1
            
            mean_gradient = np.mean(all_gradients, axis=0) if all_gradients else np.zeros(current_theta.shape)
            round_gradients[client_id] = mean_gradient
            local_gradient_trajectories[client_id].append(mean_gradient.copy())
            local_thetas_trajectories[client_id].append(current_theta.copy())
            updated_client_thetas[client_id] = current_theta
        
        global_theta = np.sum(updated_client_thetas * data_weights[:, np.newaxis], axis=0)
        global_theta_trajectory.append(global_theta.copy())
        
        if true_theta is not None:
            parameter_error = np.linalg.norm(global_theta - true_theta)
            parameter_error_trajectory.append(parameter_error)
        
        individual_gradients = []
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_gradient = compute_gradient(X_client, Y_client, global_theta)
            individual_gradient_trajectories[client_id].append(client_gradient.copy())
            individual_gradients.append(client_gradient)
        
        summed_gradient = np.sum(individual_gradients, axis=0)
        summed_gradient_trajectory.append(summed_gradient.copy())
        
        global_loss = compute_global_loss(clients_data, global_theta, NUM_CLIENTS)
        global_loss_trajectory.append(global_loss)
        
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_loss = compute_loss(X_client, Y_client, global_theta)
            client_loss_trajectories[client_id].append(client_loss)
        
        client_thetas = np.tile(global_theta, (NUM_CLIENTS, 1))
    
    return {
        'global_theta_trajectory': global_theta_trajectory,
        'individual_gradient_trajectories': individual_gradient_trajectories,
        'summed_gradient_trajectory': summed_gradient_trajectory,
        'global_loss_trajectory': global_loss_trajectory,
        'local_gradient_trajectories': local_gradient_trajectories,
        'local_thetas_trajectories': local_thetas_trajectories,
        'client_loss_trajectories': client_loss_trajectories,
        'parameter_error_trajectory': parameter_error_trajectory,
        'true_theta': true_theta
    }

def run_fedprox(clients_data, data_weights, rounds, true_theta=None):
    """Run FedProx algorithm with proximal term"""
    np.random.seed(RANDOM_SEED)
    client_thetas = np.random.normal(INITIAL_THETA_MEAN, INITIAL_THETA_STD, 
                                   (NUM_CLIENTS, INITIAL_THETA_SIZE)).astype(DTYPE)
    
    global_theta_trajectory = []
    individual_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    summed_gradient_trajectory = []
    global_loss_trajectory = []
    local_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    local_thetas_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    client_loss_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    parameter_error_trajectory = []
    
    batch_counters = np.ones(NUM_CLIENTS, dtype=np.int32)
    
    for round_num in range(rounds):
        client_local_steps = generate_varying_local_steps(LOCAL_STEPS, round_num, NUM_CLIENTS)
        current_global_theta = np.sum(client_thetas * data_weights[:, np.newaxis], axis=0)
        updated_client_thetas = np.zeros_like(client_thetas)
        round_gradients = {}
        
        for client_id in range(NUM_CLIENTS):
            theta_true, X, Y, _ = clients_data[client_id]
            current_theta = client_thetas[client_id].copy()
            client_local_step_count = client_local_steps[client_id]
            
            all_gradients = []
            for epoch in range(client_local_step_count):
                batches = create_consistent_batches(X, Y, BATCH_SIZE, round_num, client_id)
                for batch_idx, (X_batch, Y_batch) in enumerate(batches):
                    lr = get_learning_rate(batch_counters[client_id])
                    
                    prediction_error = Y_batch - X_batch @ current_theta
                    gradient = -2 * X_batch.T @ prediction_error / len(X_batch)
                    proximal_term = FEDPROX_MU * (current_theta - current_global_theta)
                    total_gradient = gradient + proximal_term
                    
                    current_theta = current_theta - lr * total_gradient
                    all_gradients.append(gradient)
                    batch_counters[client_id] += 1
            
            mean_gradient = np.mean(all_gradients, axis=0) if all_gradients else np.zeros(current_theta.shape)
            round_gradients[client_id] = mean_gradient
            local_gradient_trajectories[client_id].append(mean_gradient.copy())
            local_thetas_trajectories[client_id].append(current_theta.copy())
            updated_client_thetas[client_id] = current_theta
        
        global_theta = np.sum(updated_client_thetas * data_weights[:, np.newaxis], axis=0)
        global_theta_trajectory.append(global_theta.copy())
        
        if true_theta is not None:
            parameter_error = np.linalg.norm(global_theta - true_theta)
            parameter_error_trajectory.append(parameter_error)
        
        individual_gradients = []
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_gradient = compute_gradient(X_client, Y_client, global_theta)
            individual_gradient_trajectories[client_id].append(client_gradient.copy())
            individual_gradients.append(client_gradient)
        
        summed_gradient = np.sum(individual_gradients, axis=0)
        summed_gradient_trajectory.append(summed_gradient.copy())
        
        global_loss = compute_global_loss(clients_data, global_theta, NUM_CLIENTS)
        global_loss_trajectory.append(global_loss)
        
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_loss = compute_loss(X_client, Y_client, global_theta)
            client_loss_trajectories[client_id].append(client_loss)
        
        client_thetas = np.tile(global_theta, (NUM_CLIENTS, 1))
    
    return {
        'global_theta_trajectory': global_theta_trajectory,
        'individual_gradient_trajectories': individual_gradient_trajectories,
        'summed_gradient_trajectory': summed_gradient_trajectory,
        'global_loss_trajectory': global_loss_trajectory,
        'local_gradient_trajectories': local_gradient_trajectories,
        'local_thetas_trajectories': local_thetas_trajectories,
        'client_loss_trajectories': client_loss_trajectories,
        'parameter_error_trajectory': parameter_error_trajectory,
        'true_theta': true_theta
    }

def run_fednova(clients_data, data_weights, rounds, true_theta=None):
    """Run FedNova algorithm with normalized averaging"""
    np.random.seed(RANDOM_SEED)
    client_thetas = np.random.normal(INITIAL_THETA_MEAN, INITIAL_THETA_STD, 
                                   (NUM_CLIENTS, INITIAL_THETA_SIZE)).astype(DTYPE)
    
    global_theta_trajectory = []
    individual_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    summed_gradient_trajectory = []
    global_loss_trajectory = []
    local_gradient_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    local_thetas_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    client_loss_trajectories = {i: [] for i in range(NUM_CLIENTS)}
    parameter_error_trajectory = []
    
    batch_counters = np.ones(NUM_CLIENTS, dtype=np.int32)
    
    for round_num in range(rounds):
        client_local_steps = generate_varying_local_steps(LOCAL_STEPS, round_num, NUM_CLIENTS)
        global_theta_prev = np.sum(client_thetas * data_weights[:, np.newaxis], axis=0)
        updated_client_thetas = np.zeros_like(client_thetas)
        client_steps = np.zeros(NUM_CLIENTS)
        round_gradients = {}
        
        for client_id in range(NUM_CLIENTS):
            theta_true, X, Y, _ = clients_data[client_id]
            current_theta = client_thetas[client_id].copy()
            total_steps = 0
            client_local_step_count = client_local_steps[client_id]
            
            all_gradients = []
            for epoch in range(client_local_step_count):
                batches = create_consistent_batches(X, Y, BATCH_SIZE, round_num, client_id)
                for batch_idx, (X_batch, Y_batch) in enumerate(batches):
                    lr = get_learning_rate(batch_counters[client_id])
                    current_theta, gradient = sgd_update(current_theta, X_batch, Y_batch, lr)
                    all_gradients.append(gradient)
                    batch_counters[client_id] += 1
                    total_steps += 1
            
            mean_gradient = np.mean(all_gradients, axis=0) if all_gradients else np.zeros(current_theta.shape)
            round_gradients[client_id] = mean_gradient
            local_gradient_trajectories[client_id].append(mean_gradient.copy())
            local_thetas_trajectories[client_id].append(current_theta.copy())
            updated_client_thetas[client_id] = current_theta
            client_steps[client_id] = total_steps
        
        # FedNova normalized averaging
        client_deltas = updated_client_thetas - global_theta_prev
        effective_steps = data_weights * client_steps
        normalization_factor = np.sum(effective_steps)
        weighted_delta = np.sum(
            (effective_steps[:, np.newaxis] * client_deltas), axis=0
        ) / normalization_factor
        global_theta = global_theta_prev + weighted_delta
        
        global_theta_trajectory.append(global_theta.copy())
        
        if true_theta is not None:
            parameter_error = np.linalg.norm(global_theta - true_theta)
            parameter_error_trajectory.append(parameter_error)
        
        individual_gradients = []
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_gradient = compute_gradient(X_client, Y_client, global_theta)
            individual_gradient_trajectories[client_id].append(client_gradient.copy())
            individual_gradients.append(client_gradient)
        
        summed_gradient = np.sum(individual_gradients, axis=0)
        summed_gradient_trajectory.append(summed_gradient.copy())
        
        global_loss = compute_global_loss(clients_data, global_theta, NUM_CLIENTS)
        global_loss_trajectory.append(global_loss)
        
        for client_id in range(NUM_CLIENTS):
            _, X_client, Y_client, _ = clients_data[client_id]
            client_loss = compute_loss(X_client, Y_client, global_theta)
            client_loss_trajectories[client_id].append(client_loss)
        
        client_thetas = np.tile(global_theta, (NUM_CLIENTS, 1))
    
    return {
        'global_theta_trajectory': global_theta_trajectory,
        'individual_gradient_trajectories': individual_gradient_trajectories,
        'summed_gradient_trajectory': summed_gradient_trajectory,
        'global_loss_trajectory': global_loss_trajectory,
        'local_gradient_trajectories': local_gradient_trajectories,
        'local_thetas_trajectories': local_thetas_trajectories,
        'client_loss_trajectories': client_loss_trajectories,
        'parameter_error_trajectory': parameter_error_trajectory,
        'true_theta': true_theta
    }

def run_baseline_comparison(json_file, save_dir):
    """Run complete baseline comparison experiment for all step sizes"""
    os.makedirs(save_dir, exist_ok=True)
    clients_data, data_weights, true_theta = load_clients_data(json_file)
    
    algorithms = {
        'Stochastic': run_stochastic,
        'FedAvg': run_fedavg,
        'FedProx': run_fedprox,
        'FedNova': run_fednova
    }
    
    all_results = {}
    
    for local_steps in LOCAL_STEPS_VALUES:
        global LOCAL_STEPS
        LOCAL_STEPS = local_steps
        
        step_results = {}
        for method_name, algorithm in algorithms.items():
            step_results[method_name] = algorithm(clients_data, data_weights, ROUNDS, true_theta)
        
        all_results[local_steps] = step_results
        
        # Save individual results
        step_pickle_file = os.path.join(save_dir, f'results_local_steps_{local_steps}.pkl')
        with open(step_pickle_file, 'wb') as f:
            pickle.dump(step_results, f)
    
    # Save complete results
    complete_results_file = os.path.join(save_dir, 'complete_results.pkl')
    with open(complete_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Generate summary
    summary_lines = []
    for local_steps in LOCAL_STEPS_VALUES:
        summary_lines.append(f"Local Steps = {local_steps}:")
        summary_lines.append("-" * 40)
        
        for method_name, method_results in all_results[local_steps].items():
            final_loss = method_results['global_loss_trajectory'][-1]
            final_summed_gradient_norm = np.linalg.norm(method_results['summed_gradient_trajectory'][-1])
            
            summary_lines.append(f"  {method_name:12s}: Loss = {final_loss:8.6f}, Grad Norm = {final_summed_gradient_norm:8.6f}")
            
            if ('parameter_error_trajectory' in method_results and 
                len(method_results['parameter_error_trajectory']) > 0):
                final_parameter_error = method_results['parameter_error_trajectory'][-1]
                summary_lines.append(f"                 Parameter Error = {final_parameter_error:8.6f}")
        
        summary_lines.append("")
    
    summary_file = os.path.join(save_dir, 'results_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Run federated learning baseline comparison')
    parser.add_argument('--json_file', default='../data_generation/federated_datasets_combined_hetero/normal_theta_5_xvar_5.json', help='Path to clients_data.json file')
    parser.add_argument('--save_dir', default='./baselines_results', help='Directory to save results and plots')
    parser.add_argument('--rounds', type=int, default=2000, help='Number of rounds to run')
    parser.add_argument('--no_tapering', action='store_true', help='Disable learning rate tapering')
    
    args = parser.parse_args()
    
    global ROUNDS
    ROUNDS = args.rounds
    
    run_baseline_comparison(args.json_file, args.save_dir)

if __name__ == "__main__":
    main()