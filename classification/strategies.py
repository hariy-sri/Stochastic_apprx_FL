import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import random
from collections import OrderedDict
from models import get_model_parameters, set_model_parameters, calculate_model_difference, apply_model_difference


def get_random_epochs(client_id: int, server_round: int, max_epochs: int, seed: int = 42) -> int:
    unique_seed = seed + client_id * 1000 + server_round
    rng = random.Random(unique_seed)
    return rng.randint(1, max_epochs)


class FederatedStrategy:
    def __init__(self, learning_rate: float = 0.01, num_local_epochs: int = 1,
                 single_dominant_lr: float = 0.1, single_non_dominant_lr: float = 0.01,
                 dominant_decay_power: float = 0.76, non_dominant_decay_power: float = 1.0,
                 stochastic_c: float = 1.0, stochastic_delta: float = 0.76):
        self.learning_rate = learning_rate
        self.num_local_epochs = num_local_epochs
        self.single_dominant_lr = single_dominant_lr
        self.single_non_dominant_lr = single_non_dominant_lr
        self.dominant_decay_power = dominant_decay_power
        self.non_dominant_decay_power = non_dominant_decay_power
        self.stochastic_c = stochastic_c
        self.stochastic_delta = stochastic_delta
        
    def aggregate_parameters(self, client_parameters: List[Dict[str, torch.Tensor]], 
                           client_weights: List[float]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def local_update(self, model: nn.Module, train_loader, val_loader, 
                     device: str, client_id: int, server_round: int = 1, 
                     experiment_type: str = "same_step_size", tapering: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        raise NotImplementedError

    def evaluate_model(self, model: nn.Module, data_loader, device: str) -> Tuple[float, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                if len(target.shape) > 1 and target.shape[1] == 1:
                    target = target.squeeze(1)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    def evaluate_model_with_classwise_metrics(self, model: nn.Module, data_loader, device: str, num_classes: int) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Class-wise metrics
        class_losses = {str(i): 0.0 for i in range(num_classes)}
        class_correct = {str(i): 0 for i in range(num_classes)}
        class_total = {str(i): 0 for i in range(num_classes)}
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                if len(target.shape) > 1 and target.shape[1] == 1:
                    target = target.squeeze(1)
                output = model(data)
                losses = criterion(output, target)
                
                total_loss += losses.sum().item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
                
                # Update class-wise metrics
                for i in range(data.size(0)):
                    class_idx = str(target[i].item())
                    if class_idx in class_losses:
                        class_losses[class_idx] += losses[i].item()
                        class_total[class_idx] += 1
                        if pred[i] == target[i]:
                            class_correct[class_idx] += 1
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate class-wise averages
        class_wise_losses = {}
        class_wise_accuracies = {}
        for class_idx in class_losses:
            if class_total[class_idx] > 0:
                class_wise_losses[class_idx] = class_losses[class_idx] / class_total[class_idx]
                class_wise_accuracies[class_idx] = class_correct[class_idx] / class_total[class_idx]
            else:
                class_wise_losses[class_idx] = 0.0
                class_wise_accuracies[class_idx] = 0.0
        
        return avg_loss, accuracy, class_wise_losses, class_wise_accuracies


class FedAvg(FederatedStrategy):
    def __init__(self, learning_rate: float = 0.01, num_local_epochs: int = 1, **kwargs):
        super().__init__(learning_rate, num_local_epochs, **kwargs)
        self.name = "FedAvg"
    
    def aggregate_parameters(self, client_parameters: List[Dict[str, torch.Tensor]], 
                           client_weights: List[float], weighted_aggregate: bool = True) -> Dict[str, torch.Tensor]:
        if not client_parameters:
            return {}
        
        if weighted_aggregate:
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]
        else:
            normalized_weights = [1.0 / len(client_parameters)] * len(client_parameters)
        
        aggregated_params = {}
        for param_name in client_parameters[0].keys():
            aggregated_params[param_name] = torch.zeros_like(client_parameters[0][param_name])
        
        for client_params, weight in zip(client_parameters, normalized_weights):
            for param_name, param_value in client_params.items():
                aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    def local_update(self, model: nn.Module, train_loader, val_loader, 
                     device: str, client_id: int, server_round: int = 1, 
                     experiment_type: str = "same_step_size", tapering: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model.train()
        
        if not tapering:
            actual_epochs = get_random_epochs(client_id, server_round, self.num_local_epochs)
        else:
            actual_epochs = self.num_local_epochs
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        
        for epoch in range(actual_epochs):
            if tapering:
                global_epoch_index = (server_round - 1) * self.num_local_epochs + epoch + 1
                lr = self._calculate_learning_rate_per_epoch(client_id, global_epoch_index, experiment_type)
            else:
                lr = self._calculate_learning_rate(client_id, server_round, experiment_type)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                if len(target.shape) > 1 and target.shape[1] == 1:
                    target = target.squeeze(1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            total_correct += epoch_correct
        
        final_params = get_model_parameters(model)
        val_loss, val_accuracy = self.evaluate_model(model, val_loader, device)
        
        if tapering:
            final_global_epoch = (server_round - 1) * self.num_local_epochs + actual_epochs
            final_lr = self._calculate_learning_rate_per_epoch(client_id, final_global_epoch, experiment_type)
        else:
            final_lr = self._calculate_learning_rate(client_id, server_round, experiment_type)
        
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_accuracy': total_correct / total_samples,
            'test_loss': val_loss,
            'test_accuracy': val_accuracy,
            'num_samples': total_samples,
            'learning_rate': final_lr,
            'actual_epochs': actual_epochs
        }
        
        return final_params, metrics
    
    def _calculate_learning_rate(self, client_id: int, server_round: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            return self.learning_rate
        else:  # same_step_size
            return self.learning_rate

    def _calculate_learning_rate_per_epoch(self, client_id: int, current_epoch_index: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            return self.learning_rate
        else:  # same_step_size
            return (self.learning_rate * self.stochastic_c) / (current_epoch_index ** self.stochastic_delta)


class FedProx(FederatedStrategy):
    def __init__(self, learning_rate: float = 0.01, num_local_epochs: int = 1, mu: float = 0.01, **kwargs):
        super().__init__(learning_rate, num_local_epochs, **kwargs)
        self.name = "FedProx"
        self.mu = mu
    
    def aggregate_parameters(self, client_parameters: List[Dict[str, torch.Tensor]], 
                           client_weights: List[float], weighted_aggregate: bool = True) -> Dict[str, torch.Tensor]:
        if not client_parameters:
            return {}
        
        if weighted_aggregate:
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]
        else:
            normalized_weights = [1.0 / len(client_parameters)] * len(client_parameters)
        
        aggregated_params = {}
        for param_name in client_parameters[0].keys():
            aggregated_params[param_name] = torch.zeros_like(client_parameters[0][param_name])
        
        for client_params, weight in zip(client_parameters, normalized_weights):
            for param_name, param_value in client_params.items():
                aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    def local_update(self, model: nn.Module, train_loader, val_loader, 
                     device: str, client_id: int, server_round: int = 1, 
                     experiment_type: str = "same_step_size", tapering: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model.train()
        
        if not tapering:
            actual_epochs = get_random_epochs(client_id, server_round, self.num_local_epochs)
        else:
            actual_epochs = self.num_local_epochs
        
        global_params = [val.clone() for val in model.parameters()]
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        
        for epoch in range(actual_epochs):
            if tapering:
                global_epoch_index = (server_round - 1) * self.num_local_epochs + epoch + 1
                lr = self._calculate_learning_rate_per_epoch(client_id, global_epoch_index, experiment_type)
            else:
                lr = self._calculate_learning_rate(client_id, server_round, experiment_type)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                if len(target.shape) > 1 and target.shape[1] == 1:
                    target = target.squeeze(1)
                
                optimizer.zero_grad()
                output = model(data)
                
                loss = criterion(output, target)
                
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_params):
                    proximal_term += (self.mu / 2) * torch.norm(local_param - global_param, 2) ** 2
                
                total_loss_with_prox = loss + proximal_term
                total_loss_with_prox.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            total_correct += epoch_correct
        
        final_params = get_model_parameters(model)
        val_loss, val_accuracy = self.evaluate_model(model, val_loader, device)
        
        if tapering:
            final_global_epoch = (server_round - 1) * self.num_local_epochs + actual_epochs
            final_lr = self._calculate_learning_rate_per_epoch(client_id, final_global_epoch, experiment_type)
        else:
            final_lr = self._calculate_learning_rate(client_id, server_round, experiment_type)
        
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_accuracy': total_correct / total_samples,
            'test_loss': val_loss,
            'test_accuracy': val_accuracy,
            'num_samples': total_samples,
            'learning_rate': final_lr,
            'actual_epochs': actual_epochs
        }
        
        return final_params, metrics
    
    def _calculate_learning_rate(self, client_id: int, server_round: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            return self.learning_rate
        else:  # same_step_size
            return self.learning_rate

    def _calculate_learning_rate_per_epoch(self, client_id: int, current_epoch_index: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            return self.learning_rate
        else:  # same_step_size
            return (self.learning_rate * self.stochastic_c) / (current_epoch_index ** self.stochastic_delta)


class FedNova(FederatedStrategy):
    def __init__(self, learning_rate: float = 0.01, num_local_epochs: int = 1, mu: float = 0.01, **kwargs):
        super().__init__(learning_rate, num_local_epochs, **kwargs)
        self.name = "FedNova"
        self.mu = mu
        self.global_model_params = None
    
    def set_global_model_params(self, params: Dict[str, torch.Tensor]):
        self.global_model_params = params
    
    def aggregate_parameters(self, client_parameters: List[Dict[str, torch.Tensor]], 
                           client_weights: List[float],
                           client_tau_effs: List[float], weighted_aggregate: bool = True) -> Dict[str, torch.Tensor]:
        if not client_parameters or self.global_model_params is None:
            return {}
        
        if weighted_aggregate:
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]
        else:
            total_tau = sum(client_tau_effs)
            normalized_weights = [tau / total_tau for tau in client_tau_effs]
        
        normalized_deltas = []
        for client_params, tau_eff in zip(client_parameters, client_tau_effs):
            delta = {}
            for param_name in client_params:
                param_diff = client_params[param_name] - self.global_model_params[param_name]
                delta[param_name] = param_diff / tau_eff
            normalized_deltas.append(delta)
        
        avg_normalized_delta = {}
        for param_name in self.global_model_params:
            avg_normalized_delta[param_name] = torch.zeros_like(self.global_model_params[param_name])
            for normalized_delta, weight in zip(normalized_deltas, normalized_weights):
                avg_normalized_delta[param_name] += weight * normalized_delta[param_name]
        
        tau_eff_server = sum(tau_eff * weight for tau_eff, weight in zip(client_tau_effs, normalized_weights))
        
        updated_params = {}
        for param_name in self.global_model_params:
            updated_params[param_name] = (self.global_model_params[param_name] + 
                                        tau_eff_server * avg_normalized_delta[param_name])
        
        return updated_params
    
    def local_update(self, model: nn.Module, train_loader, val_loader, 
                     device: str, client_id: int, server_round: int = 1, 
                     experiment_type: str = "same_step_size", tapering: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model.train()
        
        if not tapering:
            actual_epochs = get_random_epochs(client_id, server_round, self.num_local_epochs)
        else:
            actual_epochs = self.num_local_epochs
        
        global_params = [val.clone() for val in model.parameters()]
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        local_steps = 0
        
        for epoch in range(actual_epochs):
            if tapering:
                global_epoch_index = (server_round - 1) * self.num_local_epochs + epoch + 1
                lr = self._calculate_learning_rate_per_epoch(client_id, global_epoch_index, experiment_type)
            else:
                lr = self._calculate_learning_rate(client_id, server_round, experiment_type)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                if len(target.shape) > 1 and target.shape[1] == 1:
                    target = target.squeeze(1)
                
                optimizer.zero_grad()
                output = model(data)
                
                loss = criterion(output, target)
                
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_params):
                    proximal_term += (self.mu / 2) * torch.norm(local_param - global_param, 2) ** 2
                
                total_loss_with_prox = loss + proximal_term
                total_loss_with_prox.backward()
                optimizer.step()
                
                local_steps += 1
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            total_correct += epoch_correct
        
        tau_eff = local_steps
        
        final_params = get_model_parameters(model)
        val_loss, val_accuracy = self.evaluate_model(model, val_loader, device)
        
        if tapering:
            final_global_epoch = (server_round - 1) * self.num_local_epochs + actual_epochs
            final_lr = self._calculate_learning_rate_per_epoch(client_id, final_global_epoch, experiment_type)
        else:
            final_lr = self._calculate_learning_rate(client_id, server_round, experiment_type)
        
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_accuracy': total_correct / total_samples,
            'test_loss': val_loss,
            'test_accuracy': val_accuracy,
            'num_samples': total_samples,
            'tau_eff': tau_eff,
            'learning_rate': final_lr,
            'actual_epochs': actual_epochs
        }
        
        return final_params, metrics
    
    def _calculate_learning_rate(self, client_id: int, server_round: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            return self.learning_rate
        else:  # same_step_size
            return self.learning_rate

    def _calculate_learning_rate_per_epoch(self, client_id: int, current_epoch_index: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            return self.learning_rate
        else:  # same_step_size
            return (self.learning_rate * self.stochastic_c) / (current_epoch_index ** self.stochastic_delta)


class Stochastic(FederatedStrategy):
    def __init__(self, learning_rate: float = 0.01, num_local_epochs: int = 1, **kwargs):
        super().__init__(learning_rate, num_local_epochs, **kwargs)
        self.name = "Stochastic"
    
    def aggregate_parameters(self, client_parameters: List[Dict[str, torch.Tensor]], 
                           client_weights: List[float], weighted_aggregate: bool = True) -> Dict[str, torch.Tensor]:
        if not client_parameters:
            return {}
        
        if weighted_aggregate:
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]
        else:
            normalized_weights = [1.0 / len(client_parameters)] * len(client_parameters)
        
        aggregated_params = {}
        for param_name in client_parameters[0].keys():
            aggregated_params[param_name] = torch.zeros_like(client_parameters[0][param_name])
        
        for client_params, weight in zip(client_parameters, normalized_weights):
            for param_name, param_value in client_params.items():
                aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    def local_update(self, model: nn.Module, train_loader, val_loader, 
                     device: str, client_id: int, server_round: int = 1, 
                     experiment_type: str = "same_step_size", tapering: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model.train()
        
        actual_epochs = self.num_local_epochs
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        
        for epoch in range(actual_epochs):
            global_epoch_index = (server_round - 1) * self.num_local_epochs + epoch + 1
            lr = self._calculate_learning_rate_per_epoch(client_id, global_epoch_index, experiment_type)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                if len(target.shape) > 1 and target.shape[1] == 1:
                    target = target.squeeze(1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            total_correct += epoch_correct
        
        final_params = get_model_parameters(model)
        val_loss, val_accuracy = self.evaluate_model(model, val_loader, device)
        
        final_global_epoch = (server_round - 1) * self.num_local_epochs + actual_epochs
        final_lr = self._calculate_learning_rate_per_epoch(client_id, final_global_epoch, experiment_type)
        
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_accuracy': total_correct / total_samples,
            'test_loss': val_loss,
            'test_accuracy': val_accuracy,
            'num_samples': total_samples,
            'learning_rate': final_lr,
            'actual_epochs': actual_epochs
        }
        
        return final_params, metrics
    
    def _calculate_learning_rate_per_epoch(self, client_id: int, current_epoch_index: int, experiment_type: str) -> float:
        if experiment_type == "single_rare":
            if client_id == 0:
                base_lr = self.single_dominant_lr
                decay_power = self.dominant_decay_power
            else:
                base_lr = self.single_non_dominant_lr
                decay_power = self.non_dominant_decay_power
        else:  # same_step_size
            base_lr = self.learning_rate
            decay_power = self.stochastic_delta
        
        return (base_lr * self.stochastic_c) / (current_epoch_index ** decay_power)


def get_strategy(strategy_name: str, **kwargs) -> FederatedStrategy:
    if strategy_name.lower() == 'fedavg':
        return FedAvg(**kwargs)
    elif strategy_name.lower() == 'fedprox':
        return FedProx(**kwargs)
    elif strategy_name.lower() == 'fednova':
        return FedNova(**kwargs)
    elif strategy_name.lower() == 'stochastic':
        return Stochastic(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")