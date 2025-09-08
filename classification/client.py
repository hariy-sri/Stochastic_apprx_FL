import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import copy
import numpy as np
import os
import pickle
from models import get_model_parameters, set_model_parameters
from strategies import FederatedStrategy


class FederatedClient:
    def __init__(self, 
                 client_id: int,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 strategy: FederatedStrategy,
                 device: str = "cpu",
                 save_path: str = "./results",
                 experiment_type: str = "same_step_size",
                 tapering: bool = False):
        
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.strategy = strategy
        self.device = device
        self.save_path = save_path
        self.experiment_type = experiment_type
        self.tapering = tapering
        
        self.round_history = []
        self.num_train_samples = len(train_loader.dataset)
        self.num_val_samples = len(val_loader.dataset)
        
        self.current_params = get_model_parameters(self.model)
        
        self._create_save_directories()
        
    def _create_save_directories(self):
        os.makedirs(os.path.join(self.save_path, "client_metrics", self.experiment_type), exist_ok=True)
        
    def update_model_parameters(self, new_params: Dict[str, torch.Tensor]):
        set_model_parameters(self.model, new_params)
        self.current_params = new_params
        
    def local_train(self, round_num: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        updated_params, metrics = self.strategy.local_update(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            client_id=self.client_id,
            server_round=round_num,
            experiment_type=self.experiment_type,
            tapering=self.tapering
        )
        
        self.current_params = updated_params
        
        self._save_round_data(round_num, metrics)
        
        metrics.update({
            'client_id': self.client_id,
            'round': round_num,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples
        })
        
        self.round_history.append(metrics)
        
        return updated_params, metrics
    
    def _save_round_data(self, round_num: int, metrics: Dict[str, float]):
        metrics_file = os.path.join(
            self.save_path, "client_metrics", self.experiment_type,
            f"client_{self.client_id}_round_{round_num}.pkl"
        )
        with open(metrics_file, "wb") as f:
            pickle.dump(metrics, f)
    
    def evaluate(self, data_loader=None) -> Tuple[float, float]:
        if data_loader is None:
            data_loader = self.val_loader
            
        loss, accuracy = self.strategy.evaluate_model(self.model, data_loader, self.device)
        return loss, accuracy
    
    def get_data_statistics(self) -> Dict[str, any]:
        class_counts = {}
        total_samples = 0
        
        for data, targets in self.train_loader:
            for target in targets:
                target_item = target.item()
                class_counts[target_item] = class_counts.get(target_item, 0) + 1
                total_samples += 1
        
        class_distribution = {k: v/total_samples for k, v in class_counts.items()}
        
        return {
            'client_id': self.client_id,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'class_counts': class_counts,
            'class_distribution': class_distribution,
            'total_samples': total_samples
        }
    
    def get_round_history(self) -> List[Dict[str, float]]:
        return self.round_history
    
    def get_parameter_norm(self) -> float:
        total_norm = 0.0
        for param_tensor in self.current_params.values():
            total_norm += torch.norm(param_tensor).item() ** 2
        return total_norm ** 0.5
    
    def reset_model(self, initial_params: Dict[str, torch.Tensor]):
        self.update_model_parameters(initial_params)
        self.round_history = []
        
    def __str__(self):
        return f"Client {self.client_id}: {self.num_train_samples} train samples, {self.num_val_samples} val samples"
    
    def __repr__(self):
        return self.__str__()


class ClientManager:
    def __init__(self, 
                 clients: List[FederatedClient],
                 save_path: str = "./results",
                 experiment_type: str = "same_step_size"):
        
        self.clients = clients
        self.num_clients = len(clients)
        self.save_path = save_path
        self.experiment_type = experiment_type
        
        self.aggregated_metrics = {
            'client_train_losses': {i: [] for i in range(self.num_clients)},
            'client_train_accuracies': {i: [] for i in range(self.num_clients)},
            'client_test_losses': {i: [] for i in range(self.num_clients)},
            'client_test_accuracies': {i: [] for i in range(self.num_clients)}
        }
        
        self._create_save_directories()
        
    def _create_save_directories(self):
        os.makedirs(os.path.join(self.save_path, "client_metrics_aggregated", self.experiment_type), exist_ok=True)
        
    def select_clients(self, 
                      num_clients: int = None, 
                      selection_probability: float = 1.0) -> List[FederatedClient]:
        return self.clients
    
    def get_client_weights(self, selected_clients: List[FederatedClient]) -> List[float]:
        return [client.num_train_samples for client in selected_clients]
    
    def update_aggregated_metrics(self, round_num: int):
        for client in self.clients:
            client_id = client.client_id
            
            if client.round_history:
                latest_metrics = client.round_history[-1]
                
                self.aggregated_metrics['client_train_losses'][client_id].append(
                    latest_metrics.get('train_loss', 0.0)
                )
                self.aggregated_metrics['client_train_accuracies'][client_id].append(0.0)
                
                test_loss, test_accuracy = client.evaluate()
                self.aggregated_metrics['client_test_losses'][client_id].append(test_loss)
                self.aggregated_metrics['client_test_accuracies'][client_id].append(test_accuracy)
        
        self._save_aggregated_metrics()
    
    def _save_aggregated_metrics(self):
        metric_files = {
            'client_train_losses': os.path.join(self.save_path, "client_metrics_aggregated", self.experiment_type, "train_losses.pkl"),
            'client_train_accuracies': os.path.join(self.save_path, "client_metrics_aggregated", self.experiment_type, "train_accuracies.pkl"),
            'client_test_losses': os.path.join(self.save_path, "client_metrics_aggregated", self.experiment_type, "test_losses.pkl"),
            'client_test_accuracies': os.path.join(self.save_path, "client_metrics_aggregated", self.experiment_type, "test_accuracies.pkl")
        }
        
        for metric_name, file_path in metric_files.items():
            with open(file_path, "wb") as f:
                pickle.dump(self.aggregated_metrics[metric_name], f)
    
    def get_all_client_statistics(self) -> List[Dict[str, any]]:
        return [client.get_data_statistics() for client in self.clients]
    
    def get_aggregate_statistics(self) -> Dict[str, any]:
        all_stats = self.get_all_client_statistics()
        
        total_train_samples = sum(stats['num_train_samples'] for stats in all_stats)
        total_val_samples = sum(stats['num_val_samples'] for stats in all_stats)
        
        aggregate_class_counts = {}
        for stats in all_stats:
            for class_id, count in stats['class_counts'].items():
                aggregate_class_counts[class_id] = aggregate_class_counts.get(class_id, 0) + count
        
        global_class_distribution = {k: v/total_train_samples for k, v in aggregate_class_counts.items()}
        
        return {
            'total_clients': self.num_clients,
            'total_train_samples': total_train_samples,
            'total_val_samples': total_val_samples,
            'aggregate_class_counts': aggregate_class_counts,
            'global_class_distribution': global_class_distribution,
            'avg_samples_per_client': total_train_samples / self.num_clients
        }
    
    def evaluate_all_clients(self, data_loader=None) -> Dict[str, float]:
        all_losses = []
        all_accuracies = []
        all_weights = []
        
        for client in self.clients:
            loss, accuracy = client.evaluate(data_loader)
            all_losses.append(loss)
            all_accuracies.append(accuracy)
            all_weights.append(client.num_train_samples)
        
        total_weight = sum(all_weights)
        weighted_loss = sum(loss * weight for loss, weight in zip(all_losses, all_weights)) / total_weight
        weighted_accuracy = sum(acc * weight for acc, weight in zip(all_accuracies, all_weights)) / total_weight
        
        return {
            'avg_loss': weighted_loss,
            'avg_accuracy': weighted_accuracy,
            'min_loss': min(all_losses),
            'max_loss': max(all_losses),
            'min_accuracy': min(all_accuracies),
            'max_accuracy': max(all_accuracies),
            'std_loss': np.std(all_losses),
            'std_accuracy': np.std(all_accuracies)
        }
    
    def reset_all_clients(self, initial_params: Dict[str, torch.Tensor]):
        for client in self.clients:
            client.reset_model(initial_params)
        
        self.aggregated_metrics = {
            'client_train_losses': {i: [] for i in range(self.num_clients)},
            'client_train_accuracies': {i: [] for i in range(self.num_clients)},
            'client_test_losses': {i: [] for i in range(self.num_clients)},
            'client_test_accuracies': {i: [] for i in range(self.num_clients)}
        }
    
    def __len__(self):
        return self.num_clients
    
    def __getitem__(self, index):
        return self.clients[index]
    
    def __str__(self):
        return f"ClientManager with {self.num_clients} clients"
    
    def __repr__(self):
        return self.__str__()