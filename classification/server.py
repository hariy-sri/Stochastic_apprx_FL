import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import os
from tqdm import tqdm
import pickle

from models import get_model_parameters, set_model_parameters, get_model
from strategies import FederatedStrategy, FedNova
from client import FederatedClient, ClientManager


class FederatedServer:
    def __init__(self,
                 model: nn.Module,
                 strategy: FederatedStrategy,
                 test_loader,
                 device: str = "cpu",
                 save_path: str = "./results",
                 experiment_type: str = "same_step_size",
                 tapering: bool = False,
                 weighted_aggregate: bool = True,
                 dataset_info: dict = None):
        
        self.model = model.to(device)
        self.strategy = strategy
        self.test_loader = test_loader
        self.device = device
        self.save_path = save_path
        self.experiment_type = experiment_type
        self.tapering = tapering
        self.weighted_aggregate = weighted_aggregate
        self.dataset_info = dataset_info or {}
        
        os.makedirs(save_path, exist_ok=True)
        
        self.global_params = get_model_parameters(self.model)
        
        self.training_history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'global_test_loss': [],
            'global_test_accuracy': [],
            'global_test_class_losses': [],
            'global_test_class_accuracies': [],
            'selected_clients': [],
            'client_metrics': [],
            'learning_rates': [],
            'client_val_losses': [],
            'client_val_accuracies': [],
            'client_train_losses': [],
            'client_train_accuracies': [],
            'parameter_errors': []
        }
        
        self.checkpoints = {}
    
    def train_round(self, 
                   client_manager: ClientManager,
                   round_num: int,
                   num_selected_clients: int = None,
                   selection_probability: float = 1.0) -> Dict[str, Any]:
        
        print(f"\n--- Round {round_num} ---")
        
        selected_clients = client_manager.select_clients(
            num_clients=num_selected_clients,
            selection_probability=selection_probability
        )
        
        print(f"Selected {len(selected_clients)} clients: {[c.client_id for c in selected_clients]}")
        
        if isinstance(self.strategy, FedNova):
            self.strategy.set_global_model_params(self.global_params)
        
        for client in selected_clients:
            client.update_model_parameters(self.global_params)
        
        client_parameters = []
        client_weights = []
        client_metrics = []
        client_tau_effs = []
        client_learning_rates = []
        
        round_client_train_losses = []
        round_client_train_accuracies = []
        round_client_val_losses = []
        round_client_val_accuracies = []
        
        for client in tqdm(selected_clients, desc="Client training"):
            params, metrics = client.local_train(round_num)
            
            client_parameters.append(params)
            client_weights.append(metrics['num_samples'])
            client_metrics.append(metrics)
            client_learning_rates.append(metrics.get('learning_rate', 0.0))
            
            round_client_train_losses.append(metrics.get('train_loss', 0.0))
            round_client_train_accuracies.append(metrics.get('train_accuracy', 0.0))
            round_client_val_losses.append(metrics.get('test_loss', 0.0))
            round_client_val_accuracies.append(metrics.get('test_accuracy', 0.0))
            
            if isinstance(self.strategy, FedNova):
                client_tau_effs.append(metrics.get('tau_eff', 1.0))
        
        if isinstance(self.strategy, FedNova):
            aggregated_params = self.strategy.aggregate_parameters(
                client_parameters, client_weights, client_tau_effs, self.weighted_aggregate
            )
        else:
            aggregated_params = self.strategy.aggregate_parameters(
                client_parameters, client_weights, self.weighted_aggregate
            )
        
        previous_global_params = {name: param.clone() for name, param in self.global_params.items()}
        
        self.global_params = aggregated_params
        set_model_parameters(self.model, self.global_params)
        
        parameter_error = self._compute_parameter_error(
            previous_global_params, self.global_params, round_num
        )
        
        print("Evaluating global model...")
        if self.dataset_info and 'num_classes' in self.dataset_info:
            global_test_loss, global_test_accuracy, class_wise_losses, class_wise_accuracies = \
                self.strategy.evaluate_model_with_classwise_metrics(
                    self.model, self.test_loader, self.device, self.dataset_info['num_classes']
                )
        else:
            global_test_loss, global_test_accuracy = self.strategy.evaluate_model(self.model, self.test_loader, self.device)
            class_wise_losses = {}
            class_wise_accuracies = {}
        
        total_samples = sum(client_weights)
        global_train_loss = sum(metrics['train_loss'] * weight for metrics, weight in zip(client_metrics, client_weights)) / total_samples
        
        if all('train_accuracy' in metrics for metrics in client_metrics):
            global_train_accuracy = sum(metrics['train_accuracy'] * weight for metrics, weight in zip(client_metrics, client_weights)) / total_samples
        else:
            global_train_accuracy = 0.0
        
        self.training_history['round'].append(round_num)
        self.training_history['train_loss'].append(global_train_loss)
        self.training_history['train_accuracy'].append(global_train_accuracy)
        self.training_history['global_test_loss'].append(global_test_loss)
        self.training_history['global_test_accuracy'].append(global_test_accuracy)
        self.training_history['global_test_class_losses'].append(class_wise_losses)
        self.training_history['global_test_class_accuracies'].append(class_wise_accuracies)
        self.training_history['selected_clients'].append([c.client_id for c in selected_clients])
        self.training_history['client_metrics'].append(client_metrics)
        self.training_history['learning_rates'].append(client_learning_rates)
        self.training_history['client_train_losses'].append(round_client_train_losses)
        self.training_history['client_train_accuracies'].append(round_client_train_accuracies)
        self.training_history['client_val_losses'].append(round_client_val_losses)
        self.training_history['client_val_accuracies'].append(round_client_val_accuracies)
        self.training_history['parameter_errors'].append(parameter_error)
        
        client_manager.update_aggregated_metrics(round_num)
        
        print(f"Round {round_num} - Global Test Loss: {global_test_loss:.4f}, Global Test Accuracy: {global_test_accuracy:.4f}")
        
        if class_wise_losses and class_wise_accuracies and self.dataset_info.get('class_names'):
            print("Class-wise Test Metrics:")
            class_names = self.dataset_info['class_names']
            for class_idx, class_name in enumerate(class_names):
                class_loss = class_wise_losses.get(str(class_idx), 0.0)
                class_acc = class_wise_accuracies.get(str(class_idx), 0.0)
                print(f"  {class_name} (Class {class_idx}): Loss = {class_loss:.4f}, Accuracy = {class_acc:.4f}")
        
        print(f"Parameter Error: {parameter_error:.6f}")
        
        print("Per-Client Validation Metrics:")
        for i, client in enumerate(selected_clients):
            val_loss = round_client_val_losses[i]
            val_acc = round_client_val_accuracies[i]
            print(f"  Client {client.client_id}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        
        round_metrics = {
            'round': round_num,
            'global_train_loss': global_train_loss,
            'global_train_accuracy': global_train_accuracy,
            'global_test_loss': global_test_loss,
            'global_test_accuracy': global_test_accuracy,
            'num_selected_clients': len(selected_clients),
            'avg_learning_rate': np.mean(client_learning_rates),
            'client_metrics': client_metrics,
            'client_train_losses': round_client_train_losses,
            'client_train_accuracies': round_client_train_accuracies,
            'client_val_losses': round_client_val_losses,
            'client_val_accuracies': round_client_val_accuracies,
        }
        
        return round_metrics
    
    def _compute_parameter_error(self, previous_params: Dict, current_params: Dict, round_num: int) -> float:
        if round_num == 0:
            return 0.0
        
        param_diff = {}
        for param_name in previous_params.keys():
            if param_name in current_params:
                param_diff[param_name] = current_params[param_name] - previous_params[param_name]
        
        unscaled_norm_squared = 0.0
        for param_name, diff_tensor in param_diff.items():
            unscaled_norm_squared += torch.norm(diff_tensor).item() ** 2
        unscaled_norm = np.sqrt(unscaled_norm_squared)
        
        return unscaled_norm
    
    def get_client_metrics_by_round(self, round_num: int) -> Dict[str, List[float]]:
        if round_num >= len(self.training_history['round']):
            return {}
        
        return {
            'client_train_losses': self.training_history['client_train_losses'][round_num],
            'client_train_accuracies': self.training_history['client_train_accuracies'][round_num],
            'client_val_losses': self.training_history['client_val_losses'][round_num],
            'client_val_accuracies': self.training_history['client_val_accuracies'][round_num],
        }
    
    def get_client_metrics_across_rounds(self, client_id: int) -> Dict[str, List[float]]:
        num_rounds = len(self.training_history['round'])
        
        client_train_losses = []
        client_train_accuracies = []
        client_val_losses = []
        client_val_accuracies = []
        
        for round_idx in range(num_rounds):
            selected_clients = self.training_history['selected_clients'][round_idx]
            if client_id in selected_clients:
                client_idx = selected_clients.index(client_id)
                
                client_train_losses.append(self.training_history['client_train_losses'][round_idx][client_idx])
                client_train_accuracies.append(self.training_history['client_train_accuracies'][round_idx][client_idx])
                client_val_losses.append(self.training_history['client_val_losses'][round_idx][client_idx])
                client_val_accuracies.append(self.training_history['client_val_accuracies'][round_idx][client_idx])
            else:
                client_train_losses.append(None)
                client_train_accuracies.append(None)
                client_val_losses.append(None)
                client_val_accuracies.append(None)
        
        return {
            'rounds': self.training_history['round'],
            'train_losses': client_train_losses,
            'train_accuracies': client_train_accuracies,
            'val_losses': client_val_losses,
            'val_accuracies': client_val_accuracies,
        }
    
    def evaluate_global_model(self) -> Tuple[float, float]:
        return self.strategy.evaluate_model(self.model, self.test_loader, self.device)
    
    def save_checkpoint(self, round_num: int, metadata: Dict[str, Any] = None):
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'global_params': self.global_params,
            'training_history': self.training_history,
            'experiment_type': self.experiment_type,
            'tapering': self.tapering,
            'metadata': metadata or {}
        }
        
        checkpoint_path = os.path.join(self.save_path, f"checkpoint_round_{round_num}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        self.checkpoints[round_num] = checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_params = checkpoint['global_params']
        self.training_history = checkpoint['training_history']
        self.experiment_type = checkpoint.get('experiment_type', 'same_step_size')
        self.tapering = checkpoint.get('tapering', False)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['round']
    
    def save_results(self, filename: str = "training_results.pkl"):
        results_data = {
            'training_history': self.training_history,
            'experiment_type': self.experiment_type,
            'tapering': self.tapering,
            'weighted_aggregate': self.weighted_aggregate,
            'strategy_name': self.strategy.name,
            'device': self.device
        }
        
        results_path = os.path.join(self.save_path, filename)
        with open(results_path, 'wb') as f:
            pickle.dump(results_data, f)
        
        json_filename = filename.replace('.pkl', '.json')
        json_path = os.path.join(self.save_path, json_filename)
        
        json_summary = {
            'experiment_info': {
                'strategy_name': self.strategy.name,
                'experiment_type': self.experiment_type,
                'tapering': self.tapering,
                'weighted_aggregate': self.weighted_aggregate,
                'device': self.device,
                'dataset_info': self.dataset_info
            },
            'global_metrics': {
                'rounds': self.training_history['round'],
                'global_train_loss': self.training_history['train_loss'],
                'global_train_accuracy': self.training_history['train_accuracy'],
                'global_test_loss': self.training_history['global_test_loss'],
                'global_test_accuracy': self.training_history['global_test_accuracy'],
                'global_test_class_losses': self.training_history['global_test_class_losses'],
                'global_test_class_accuracies': self.training_history['global_test_class_accuracies'],
                'learning_rates': self.training_history['learning_rates'],
                'parameter_errors': self.training_history['parameter_errors']
            },
            'per_client_metrics': {
                'client_train_losses': self.training_history['client_train_losses'],
                'client_train_accuracies': self.training_history['client_train_accuracies'],
                'client_val_losses': self.training_history['client_val_losses'],
                'client_val_accuracies': self.training_history['client_val_accuracies']
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        print(f"Results saved to:")
        print(f"  Pickle: {results_path}")
        print(f"  JSON: {json_path}")
        return results_path
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        if not self.training_history['global_test_accuracy']:
            return {
                'final_global_test_accuracy': 0.0,
                'best_global_test_accuracy': 0.0,
                'best_round': 0,
                'final_global_test_loss': float('inf'),
                'best_global_test_loss': float('inf')
            }
        
        test_accuracies = self.training_history['global_test_accuracy']
        test_losses = self.training_history['global_test_loss']
        
        best_acc_idx = np.argmax(test_accuracies)
        best_loss_idx = np.argmin(test_losses)
        
        summary = {
            'final_global_test_accuracy': test_accuracies[-1],
            'best_global_test_accuracy': test_accuracies[best_acc_idx],
            'best_round': self.training_history['round'][best_acc_idx],
            'final_global_test_loss': test_losses[-1],
            'best_global_test_loss': test_losses[best_loss_idx]
        }
        
        if self.training_history['parameter_errors']:
            summary['final_parameter_error'] = self.training_history['parameter_errors'][-1]
        
        return summary
    
    def reset_training_history(self):
        self.training_history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'global_test_loss': [],
            'global_test_accuracy': [],
            'global_test_class_losses': [],
            'global_test_class_accuracies': [],
            'selected_clients': [],
            'client_metrics': [],
            'learning_rates': [],
            'client_val_losses': [],
            'client_val_accuracies': [],
            'client_train_losses': [],
            'client_train_accuracies': [],
            'parameter_errors': []
        }
        
        self.checkpoints = {}
    
    def collect_round_0_data(self, client_manager: 'ClientManager') -> Dict[str, Any]:
        print(f"\n--- Round 0 (Initial Data Collection) ---")
        
        all_clients = client_manager.clients
        
        for client in all_clients:
            client.update_model_parameters(self.global_params)
        
        client_metrics = []
        round_client_train_losses = []
        round_client_train_accuracies = []
        round_client_val_losses = []
        round_client_val_accuracies = []
        
        for client in all_clients:
            print(f"Collecting initial data for Client {client.client_id}")
            
            val_loss, val_accuracy = client.evaluate(client.val_loader)
            
            client_metric = {
                'client_id': client.client_id,
                'test_loss': val_loss,
                'test_accuracy': val_accuracy,
                'num_samples': client.num_train_samples
            }
            
            client_metrics.append(client_metric)
            round_client_train_losses.append(0.0)
            round_client_train_accuracies.append(0.0)
            round_client_val_losses.append(val_loss)
            round_client_val_accuracies.append(val_accuracy)
        
        if self.dataset_info and 'num_classes' in self.dataset_info:
            global_test_loss, global_test_accuracy, class_wise_losses, class_wise_accuracies = \
                self.strategy.evaluate_model_with_classwise_metrics(
                    self.model, self.test_loader, self.device, self.dataset_info['num_classes']
                )
        else:
            global_test_loss, global_test_accuracy = self.strategy.evaluate_model(
                self.model, self.test_loader, self.device
            )
            class_wise_losses = {}
            class_wise_accuracies = {}
        
        self.training_history['round'].append(0)
        self.training_history['train_loss'].append(0.0)
        self.training_history['train_accuracy'].append(0.0)
        self.training_history['global_test_loss'].append(global_test_loss)
        self.training_history['global_test_accuracy'].append(global_test_accuracy)
        self.training_history['global_test_class_losses'].append(class_wise_losses)
        self.training_history['global_test_class_accuracies'].append(class_wise_accuracies)
        self.training_history['selected_clients'].append([c.client_id for c in all_clients])
        self.training_history['client_metrics'].append(client_metrics)
        self.training_history['learning_rates'].append([0.0] * len(all_clients))
        self.training_history['client_train_losses'].append(round_client_train_losses)
        self.training_history['client_train_accuracies'].append(round_client_train_accuracies)
        self.training_history['client_val_losses'].append(round_client_val_losses)
        self.training_history['client_val_accuracies'].append(round_client_val_accuracies)
        self.training_history['parameter_errors'].append(0.0)
        
        total_samples = sum(m.get('num_samples', 0) for m in client_metrics)
        avg_val_loss = sum(m.get('test_loss', 0) * m.get('num_samples', 0) for m in client_metrics) / max(1, total_samples)
        avg_val_accuracy = sum(m.get('test_accuracy', 0) * m.get('num_samples', 0) for m in client_metrics) / max(1, total_samples)
        
        round_0_data = {
            'round': 0,
            'global_train_loss': 0.0,
            'global_train_accuracy': 0.0,
            'avg_val_loss': avg_val_loss,
            'avg_val_accuracy': avg_val_accuracy,
            'global_test_loss': global_test_loss,
            'global_test_accuracy': global_test_accuracy,
            'client_metrics': client_metrics,
            'client_train_losses': round_client_train_losses,
            'client_train_accuracies': round_client_train_accuracies,
            'client_val_losses': round_client_val_losses,
            'client_val_accuracies': round_client_val_accuracies,
        }
        
        print(f"Round 0 - Global Test Loss: {global_test_loss:.4f}, Global Test Accuracy: {global_test_accuracy:.4f}")
        print(f"Round 0 - Avg Client Val Loss: {avg_val_loss:.4f}, Avg Client Val Accuracy: {avg_val_accuracy:.4f}")
        
        return round_0_data
    
    def __str__(self):
        return f"FederatedServer with {self.strategy.name} strategy ({self.experiment_type}, tapering={self.tapering}, weighted_aggregate={self.weighted_aggregate})"
    
    def __repr__(self):
        return self.__str__()