import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import os
from datetime import datetime
import json
import copy
import pickle

from models import get_model, get_model_parameters
from dataset import create_federated_datasets, get_dataset_info
from strategies import get_strategy
from client import FederatedClient, ClientManager
from server import FederatedServer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_config(args, algorithm: str, experiment_type: str, tapering: bool):
    return {
        'dataset': args.dataset,
        'algorithm': algorithm,
        'experiment_type': experiment_type,
        'tapering': tapering,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'num_local_epochs': args.num_local_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'partition_type': args.partition_type,
        'alpha': args.alpha,
        'mu': args.mu,
        'weighted_aggregate': args.weighted_aggregate.lower() == 'true',
        'seed': args.seed,
        'device': args.device,
        'timestamp': datetime.now().isoformat(),
        'single_dominant_lr': args.single_dominant_lr,
        'single_non_dominant_lr': args.single_non_dominant_lr,
        'dominant_decay_power': args.dominant_decay_power,
        'non_dominant_decay_power': args.non_dominant_decay_power,
        'stochastic_c': args.stochastic_c,
        'stochastic_delta': args.stochastic_delta
    }


def get_client_configs(dataset: str):
    if dataset.lower() in ['mnist', 'cifar10', 'bloodmnist', 'organsmnist']:
        return 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def setup_experiment_directories(base_path: str, experiment_type: str, dataset: str, 
                                algorithm: str, tapering: bool):
    if algorithm.lower() == 'stochastic':
        experiment_dir = os.path.join(base_path, experiment_type, dataset, f"{algorithm}")
    else:
        tapering_str = "tapered" if tapering else "constant"
        experiment_dir = os.path.join(base_path, experiment_type, dataset, f"{algorithm}_{tapering_str}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def run_single_experiment(dataset: str, algorithm: str, experiment_type: str, tapering: bool,
                         base_args, base_path: str = "./results"):
    
    args = copy.deepcopy(base_args)
    args.dataset = dataset
    args.algorithm = algorithm
    args.num_clients = get_client_configs(dataset)
    args.tapering = tapering
    
    config = create_experiment_config(args, algorithm, experiment_type, tapering)
    
    save_path = setup_experiment_directories(base_path, experiment_type, dataset, 
                                           algorithm, tapering)
    config['save_path'] = save_path
    
    set_seed(args.seed)
    
    print(f"\nStarting Experiment:")
    print(f"  Dataset: {dataset}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Experiment Type: {experiment_type}")
    print(f"  Tapering: {tapering}")
    print(f"  Results Path: {save_path}")
    
    client_manager, server = setup_federated_learning(config)
    results = run_federated_training(client_manager, server, config)
    
    config_path = os.path.join(save_path, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment completed! Results saved to: {save_path}")
    
    return results


def setup_federated_learning(config):
    dataset_info = get_dataset_info(config['dataset'])
    print(f"Dataset: {config['dataset']}")
    print(f"  Number of classes: {dataset_info['num_classes']}")
    print(f"  Input channels: {dataset_info['input_channels']}")
    print(f"  Input size: {dataset_info['input_size']}")
    
    print(f"\nCreating federated datasets...")
    
    if config['experiment_type'] == 'single_rare':
        partition_type = 'single_rare'
        print(f"Using single_rare partition: Client 0 gets rare distribution")
    else:  # same_step_size
        partition_type = config['partition_type']
        print(f"Using {partition_type} partition")
    
    client_train_loaders, client_val_loaders, test_loader = create_federated_datasets(
        dataset_name=config['dataset'],
        num_clients=config['num_clients'],
        partition_type=partition_type,
        alpha=config['alpha'],
        batch_size=config['batch_size']
    )
    
    print(f"Created {len(client_train_loaders)} client datasets")
    
    model = get_model(config['dataset'], dataset_info['num_classes'])
    print(f"\nModel architecture: {model.__class__.__name__}")
    
    clients = []
    for i in range(config['num_clients']):
        client_model = get_model(config['dataset'], dataset_info['num_classes'])
        
        client_strategy_kwargs = {
            'learning_rate': config['learning_rate'],
            'num_local_epochs': config['num_local_epochs'],
            'single_dominant_lr': config['single_dominant_lr'],
            'single_non_dominant_lr': config['single_non_dominant_lr'],
            'dominant_decay_power': config['dominant_decay_power'],
            'non_dominant_decay_power': config['non_dominant_decay_power'],
            'stochastic_c': config['stochastic_c'],
            'stochastic_delta': config['stochastic_delta']
        }
        
        if config['algorithm'].lower() in ['fedprox', 'fednova']:
            client_strategy_kwargs['mu'] = config['mu']
        
        client_strategy = get_strategy(config['algorithm'], **client_strategy_kwargs)
        
        client = FederatedClient(
            client_id=i,
            model=client_model,
            train_loader=client_train_loaders[i],
            val_loader=client_val_loaders[i],
            strategy=client_strategy,
            device=config['device'],
            save_path=config['save_path'],
            experiment_type=config['experiment_type'],
            tapering=config['tapering']
        )
        clients.append(client)
    
    server_strategy_kwargs = {
        'learning_rate': config['learning_rate'],
        'num_local_epochs': config['num_local_epochs'],
        'single_dominant_lr': config['single_dominant_lr'],
        'single_non_dominant_lr': config['single_non_dominant_lr'],
        'dominant_decay_power': config['dominant_decay_power'],
        'non_dominant_decay_power': config['non_dominant_decay_power'],
        'stochastic_c': config['stochastic_c'],
        'stochastic_delta': config['stochastic_delta']
    }
    
    if config['algorithm'].lower() in ['fedprox', 'fednova']:
        server_strategy_kwargs['mu'] = config['mu']
    
    server_strategy = get_strategy(config['algorithm'], **server_strategy_kwargs)
    
    print(f"Strategy: {server_strategy.name}")
    
    server_model = get_model(config['dataset'], dataset_info['num_classes'])
    server = FederatedServer(
        model=server_model,
        strategy=server_strategy,
        test_loader=test_loader,
        device=config['device'],
        save_path=config['save_path'],
        experiment_type=config['experiment_type'],
        tapering=config['tapering'],
        weighted_aggregate=config['weighted_aggregate'],
        dataset_info=dataset_info
    )
    
    client_manager = ClientManager(
        clients=clients,
        save_path=config['save_path'],
        experiment_type=config['experiment_type']
    )
    
    print(f"\nCreated client manager with {len(clients)} clients")
    
    return client_manager, server


def run_federated_training(client_manager, server, config):
    print(f"\nStarting Federated Learning Training")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Experiment Type: {config['experiment_type']}")
    print(f"Tapering: {config['tapering']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Clients: {config['num_clients']}")
    print(f"Rounds: {config['num_rounds']}")
    print(f"Local Epochs: {config['num_local_epochs']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Device: {config['device']}")
    
    client_stats = client_manager.get_aggregate_statistics()
    print(f"\nClient Statistics:")
    print(f"  Total training samples: {client_stats['total_train_samples']}")
    print(f"  Average samples per client: {client_stats['avg_samples_per_client']:.1f}")
    print(f"  Class distribution: {client_stats['global_class_distribution']}")
    
    print(f"\nRound 0: Initial Data Collection")
    round_0_metrics = server.collect_round_0_data(client_manager)
    
    print(f"\nInitial evaluation complete...")
    print(f"Initial Test Loss: {round_0_metrics['global_test_loss']:.4f}")
    print(f"Initial Test Accuracy: {round_0_metrics['global_test_accuracy']:.4f}")
    
    print(f"\nStarting Training Rounds")
    for round_num in range(1, config['num_rounds'] + 1):
        round_metrics = server.train_round(
            client_manager=client_manager,
            round_num=round_num,
            num_selected_clients=config['num_clients']
        )
        
        if round_num % 25 == 0:
            server.save_checkpoint(round_num, metadata=config)
    
    print(f"\nTraining Complete!")
    
    summary_stats = server.get_summary_statistics()
    print(f"Final Test Accuracy: {summary_stats['final_global_test_accuracy']:.4f}")
    print(f"Best Test Accuracy: {summary_stats['best_global_test_accuracy']:.4f} (Round {summary_stats['best_round']})")
    print(f"Final Test Loss: {summary_stats['final_global_test_loss']:.4f}")
    print(f"Best Test Loss: {summary_stats['best_global_test_loss']:.4f}")
    
    results_filename = f"results_{config['algorithm']}_{config['experiment_type']}_tapering{config['tapering']}.pkl"
    results_path = server.save_results(results_filename)
    
    return {'server': server, 'summary_stats': summary_stats, 'config': config}


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Experiment Framework')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'bloodmnist', 'organsmnist'],
                        help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='fedavg',
                        choices=['stochastic', 'fedavg', 'fedprox', 'fednova'],
                        help='Federated learning algorithm')
    parser.add_argument('--experiment-type', type=str, default='same_step_size',
                        choices=['same_step_size', 'single_rare'],
                        help='Experiment type')
    parser.add_argument('--tapering', action='store_true',
                        help='Enable tapering learning rate')
    
    parser.add_argument('--num-rounds', type=int, default=100,
                        help='Number of federated learning rounds')
    parser.add_argument('--num-local-epochs', type=int, default=5,
                        help='Number of local epochs')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Base learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    
    parser.add_argument('--mu', type=float, default=0.1,
                        help='Proximal term coefficient for FedProx and FedNova')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter for Dirichlet distribution (non-IID)')
    parser.add_argument('--partition-type', type=str, default='near_pathological',
                        choices=['near_pathological', 'iid', 'single_rare', 'dirichlet'],
                        help='Data partition type')
    parser.add_argument('--weighted-aggregate', type=str, choices=['true', 'false'], default='true',
                        help='Use data-size weights for aggregation')
    
    parser.add_argument('--single-dominant-lr', type=float, default=0.1,
                        help='Learning rate for single rare client (default: 0.1)')
    parser.add_argument('--single-non-dominant-lr', type=float, default=0.01,
                        help='Learning rate for other clients in single_rare (default: 0.01)')
    parser.add_argument('--dominant-decay-power', type=float, default=0.76,
                        help='Decay power for rare client (default: 0.76)')
    parser.add_argument('--non-dominant-decay-power', type=float, default=1.0,
                        help='Decay power for other clients (default: 1.0)')
    parser.add_argument('--stochastic-c', type=float, default=1.0,
                        help='Stochastic tapering coefficient C (default: 1.0)')
    parser.add_argument('--stochastic-delta', type=float, default=0.76,
                        help='Stochastic tapering delta parameter (default: 0.76)')
    
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--results-path', type=str, default='./results',
                        help='Base path to save results')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    args.device = device
    
    print(f"Federated Learning Experiment Framework")
    print(f"Device: {device}")
    print(f"Results will be saved to: {args.results_path}")
    
    result = run_single_experiment(
        dataset=args.dataset,
        algorithm=args.algorithm,
        experiment_type=args.experiment_type,
        tapering=args.tapering,
        base_args=args,
        base_path=args.results_path
    )
    print(f"\nExperiment completed!")
    
    return 0


if __name__ == "__main__":
    main()