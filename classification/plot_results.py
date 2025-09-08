import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Any
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')
ACADEMIC_DPI = 300
ACADEMIC_FONT_SIZE = 10
ACADEMIC_TITLE_SIZE = 12
ACADEMIC_LABEL_SIZE = 10
ACADEMIC_LEGEND_SIZE = 9

INDIVIDUAL_PLOT_SIZE = (10, 5)
COMPARISON_PLOT_SIZE = (12, 4) 
COMBINED_PLOT_SIZE = (12, 5)
CROSS_ALGORITHM_SIZE = (12, 5)


def format_axis_decimals(ax, metric_key):
    """Format y-axis to show maximum 2 decimal places for loss and accuracy metrics"""
    if any(keyword in metric_key.lower() for keyword in ['loss', 'accuracy']):
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


def adjust_accuracy_ylimits(ax, metric_key):
    """Adjust y-axis limits for accuracy plots to focus on the data range"""
    if 'accuracy' in metric_key.lower():
        all_y_data = []
        for line in ax.get_lines():
            y_data = line.get_ydata()
            if len(y_data) > 0:
                all_y_data.extend(y_data)
        
        if all_y_data:
            data_min = min(all_y_data)
            data_max = max(all_y_data)
            data_range = data_max - data_min
            
            padding = max(data_range * 0.1, 0.03)
            
            new_min = max(0, data_min - padding)
            new_max = min(1, data_max + padding)
            
            y_min, y_max = ax.get_ylim()
            current_range = y_max - y_min
            
            if (current_range > 0.7 or data_range < 0.2 or 
                (new_max - new_min) < current_range * 0.6):
                ax.set_ylim(new_min, new_max)


def load_experiment_data(results_path: str, experiment_type: str, dataset: str) -> Dict[str, Any]:
    """Load experiment data from structured results directory"""
    
    experiment_dir = os.path.join(results_path, experiment_type, dataset)
    
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    algorithms = {
        'stochastic': [''], 
        'fedavg': ['_constant', '_tapered'],
        'fedprox': ['_constant', '_tapered'],
        'fednova': ['_constant', '_tapered']
    }
    
    experiment_data = {}
    
    for algorithm, variants in algorithms.items():
        experiment_data[algorithm] = {}
        
        for variant in variants:
            algorithm_dir = os.path.join(experiment_dir, f"{algorithm}{variant}")
            
            if os.path.exists(algorithm_dir):
                variant_name = 'tapered' if variant == '_tapered' else 'constant'
                
                experiment_data[algorithm][variant_name] = {
                    'config': None,
                    'results': None
                }
                
                for filename in os.listdir(algorithm_dir):
                    if filename.startswith('results_') and filename.endswith('.pkl'):
                        results_file = os.path.join(algorithm_dir, filename)
                        with open(results_file, 'rb') as f:
                            loaded_data = pickle.load(f)
                            experiment_data[algorithm][variant_name]['results'] = loaded_data
                            print(f"Loaded {results_file}")
                            break
    
    return experiment_data


def extract_metrics_over_rounds(experiment_data: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    """Extract metrics from experiment data"""
    
    metrics_data = {}
    
    for algorithm, variants in experiment_data.items():
        for variant_name, data in variants.items():
            if algorithm == 'stochastic':
                alg_key = 'stochastic'
            else:
                alg_key = f"{algorithm}_{variant_name}"
            
            print(f"Processing {alg_key}...")
            
            if 'results' not in data or data['results'] is None or 'training_history' not in data['results']:
                print(f"Warning: No training_history found for {alg_key}")
                continue
            
            history = data['results']['training_history']
            
            metrics_data[alg_key] = {
                'rounds': history.get('round', []),
                'variant': variant_name,
                'algorithm_base': algorithm
            }
            
            # Extract global metrics
            metrics_data[alg_key]['train_loss'] = history.get('train_loss', [])
            metrics_data[alg_key]['train_accuracy'] = history.get('train_accuracy', [])
            metrics_data[alg_key]['global_test_loss'] = history.get('global_test_loss', [])
            metrics_data[alg_key]['global_test_accuracy'] = history.get('global_test_accuracy', [])
            
            # Extract parameter errors
            metrics_data[alg_key]['parameter_errors'] = history.get('parameter_errors', [])
            
            # Store structured client metrics
            metrics_data[alg_key]['client_train_losses'] = history.get('client_train_losses', [])
            metrics_data[alg_key]['client_train_accuracies'] = history.get('client_train_accuracies', [])
            metrics_data[alg_key]['client_val_losses'] = history.get('client_val_losses', [])
            metrics_data[alg_key]['client_val_accuracies'] = history.get('client_val_accuracies', [])
            
            # Extract per-client metrics
            client_train_losses = history.get('client_train_losses', [])
            client_train_accuracies = history.get('client_train_accuracies', [])
            client_val_losses = history.get('client_val_losses', [])
            client_val_accuracies = history.get('client_val_accuracies', [])
            
            if client_train_losses:
                num_clients = len(client_train_losses[0]) if client_train_losses else 0
                
                for client_id in range(num_clients):
                    metrics_data[alg_key][f'client_{client_id}_train_loss'] = [
                        round_data[client_id] for round_data in client_train_losses
                    ]
                    metrics_data[alg_key][f'client_{client_id}_train_accuracy'] = [
                        round_data[client_id] for round_data in client_train_accuracies
                    ]
                    metrics_data[alg_key][f'client_{client_id}_test_loss'] = [
                        round_data[client_id] for round_data in client_val_losses
                    ]
                    metrics_data[alg_key][f'client_{client_id}_test_accuracy'] = [
                        round_data[client_id] for round_data in client_val_accuracies
                    ]
    
    return metrics_data


def plot_paired_metrics(metrics_data: Dict[str, Dict[str, List]], 
                       algorithm: str, output_dir: str, 
                       experiment_type: str, dataset: str,
                       metric_type: str):
    """Plot paired accuracy and loss metrics in 1x2 layout"""
    
    if algorithm not in metrics_data:
        print(f"Warning: No data found for algorithm {algorithm}")
        return
    
    data = metrics_data[algorithm]
    rounds = data.get('rounds', [])
    
    if not rounds:
        print(f"Warning: No rounds data for algorithm {algorithm}")
        return
    
    # Limit to 100 rounds
    original_rounds = rounds.copy()
    if len(rounds) > 100:
        if rounds[0] == 0:
            rounds = rounds[1:101]
        else:
            rounds = rounds[:100]
    
    num_clients = 0
    for key in data.keys():
        if key.startswith('client_') and key.endswith('_train_loss'):
            num_clients += 1
    
    if num_clients == 0:
        print(f"Warning: No client data for algorithm {algorithm}")
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_clients))
    
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=INDIVIDUAL_PLOT_SIZE)
    
    if metric_type == 'training':
        acc_title = 'Training Accuracy'
        loss_title = 'Training Loss'
        client_acc_suffix = '_train_accuracy'
        client_loss_suffix = '_train_loss'
        global_acc_key = None
        global_loss_key = None
    elif metric_type == 'validation':
        acc_title = 'Validation Accuracy'
        loss_title = 'Validation Loss'
        client_acc_suffix = '_test_accuracy'
        client_loss_suffix = '_test_loss'
        global_acc_key = None
        global_loss_key = None
    elif metric_type == 'global_test':
        acc_title = 'Global Test Accuracy'
        loss_title = 'Global Test Loss'
        client_acc_suffix = None
        client_loss_suffix = None
        global_acc_key = 'global_test_accuracy'
        global_loss_key = 'global_test_loss'
    
    clients_plotted_acc = 0
    clients_plotted_loss = 0
    
    if client_acc_suffix and client_loss_suffix:
        for client_id in range(num_clients):
            client_acc_key = f'client_{client_id}{client_acc_suffix}'
            client_loss_key = f'client_{client_id}{client_loss_suffix}'
            
            if client_acc_key in data and data[client_acc_key]:
                client_acc_data = data[client_acc_key]
                if len(client_acc_data) == len(original_rounds):
                    if len(original_rounds) > 100:
                        if original_rounds[0] == 0:
                            client_acc_data = client_acc_data[1:101]
                        else:
                            client_acc_data = client_acc_data[:100]
                ax_acc.plot(rounds[:len(client_acc_data)], client_acc_data, 
                           label=f'Client {client_id + 1}', color=colors[client_id], 
                           marker='o', markersize=2, alpha=0.8, linewidth=1.5)
                clients_plotted_acc += 1
            
            if client_loss_key in data and data[client_loss_key]:
                client_loss_data = data[client_loss_key]
                if len(client_loss_data) == len(original_rounds):
                    if len(original_rounds) > 100:
                        if original_rounds[0] == 0:
                            client_loss_data = client_loss_data[1:101]
                        else:
                            client_loss_data = client_loss_data[:100]
                ax_loss.plot(rounds[:len(client_loss_data)], client_loss_data, 
                            label=f'Client {client_id + 1}', color=colors[client_id], 
                            marker='o', markersize=2, alpha=0.8, linewidth=1.5)
                clients_plotted_loss += 1
    
    if global_acc_key and global_loss_key:
        if global_acc_key in data and data[global_acc_key]:
            global_acc = data[global_acc_key]
            if len(global_acc) == len(original_rounds):
                if len(original_rounds) > 100:
                    if original_rounds[0] == 0:
                        global_acc = global_acc[1:101]
                    else:
                        global_acc = global_acc[:100]
            ax_acc.plot(rounds[:len(global_acc)], global_acc, 
                       label='Global Test', color='blue', linewidth=3, alpha=0.9, marker='s', markersize=4)
            clients_plotted_acc = 1
        
        if global_loss_key in data and data[global_loss_key]:
            global_loss = data[global_loss_key]
            if len(global_loss) == len(original_rounds):
                if len(original_rounds) > 100:
                    if original_rounds[0] == 0:
                        global_loss = global_loss[1:101]
                    else:
                        global_loss = global_loss[:100]
            ax_loss.plot(rounds[:len(global_loss)], global_loss, 
                        label='Global Test', color='blue', linewidth=3, alpha=0.9, marker='s', markersize=4)
            clients_plotted_loss = 1
    else:
        # Add average lines for training/validation
        if clients_plotted_acc > 0:
            avg_acc = []
            for round_idx in range(len(rounds)):
                round_values = []
                for client_id in range(num_clients):
                    client_key = f'client_{client_id}{client_acc_suffix}'
                    if client_key in data:
                        client_data = data[client_key]
                        if len(client_data) == len(original_rounds):
                            if len(original_rounds) > 100 and original_rounds[0] == 0:
                                data_idx = round_idx + 1
                            else:
                                data_idx = round_idx
                        else:
                            data_idx = round_idx
                        
                        if data_idx < len(client_data):
                            round_values.append(client_data[data_idx])
                
                if round_values:
                    avg_acc.append(sum(round_values) / len(round_values))
            
            if avg_acc:
                ax_acc.plot(rounds[:len(avg_acc)], avg_acc, label='Average', 
                           color='black', linestyle='--', linewidth=2, alpha=0.9)
        
        if clients_plotted_loss > 0:
            avg_loss = []
            for round_idx in range(len(rounds)):
                round_values = []
                for client_id in range(num_clients):
                    client_key = f'client_{client_id}{client_loss_suffix}'
                    if client_key in data:
                        client_data = data[client_key]
                        if len(client_data) == len(original_rounds):
                            if len(original_rounds) > 100 and original_rounds[0] == 0:
                                data_idx = round_idx + 1
                            else:
                                data_idx = round_idx
                        else:
                            data_idx = round_idx
                        
                        if data_idx < len(client_data):
                            round_values.append(client_data[data_idx])
                
                if round_values:
                    avg_loss.append(sum(round_values) / len(round_values))
            
            if avg_loss:
                ax_loss.plot(rounds[:len(avg_loss)], avg_loss, label='Average', 
                            color='black', linestyle='--', linewidth=2, alpha=0.9)
    
    if clients_plotted_acc > 0:
        ax_acc.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
        ax_acc.set_ylabel(acc_title, fontsize=ACADEMIC_LABEL_SIZE)
        ax_acc.set_title(acc_title, fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
        ax_acc.grid(True, alpha=0.3)
        handles, labels = ax_acc.get_legend_handles_labels()
        if len(handles) > 1:
            ax_acc.legend(fontsize=ACADEMIC_LEGEND_SIZE, loc='best')
        format_axis_decimals(ax_acc, 'accuracy')
        adjust_accuracy_ylimits(ax_acc, 'accuracy')
    
    if clients_plotted_loss > 0:
        ax_loss.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
        ax_loss.set_ylabel(loss_title, fontsize=ACADEMIC_LABEL_SIZE)
        ax_loss.set_title(loss_title, fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        handles, labels = ax_loss.get_legend_handles_labels()
        if len(handles) > 1:
            ax_loss.legend(fontsize=ACADEMIC_LEGEND_SIZE, loc='best')
        format_axis_decimals(ax_loss, 'loss')
    
    fig.suptitle(f'{metric_type.replace("_", " ").title()} - {dataset.upper()}', 
                fontsize=ACADEMIC_TITLE_SIZE + 1, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    filename = f"{algorithm}_{metric_type}_metrics.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=ACADEMIC_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filepath}")


def plot_individual_algorithm(metrics_data: Dict[str, Dict[str, List]], 
                             algorithm: str, output_dir: str, 
                             experiment_type: str, dataset: str):
    """Plot individual algorithm parameter error metrics"""
    
    if algorithm not in metrics_data:
        print(f"Warning: No data found for algorithm {algorithm}")
        return
    
    data = metrics_data[algorithm]
    rounds = data.get('rounds', [])
    
    if not rounds:
        print(f"Warning: No rounds data for algorithm {algorithm}")
        return
    
    # Limit to 100 rounds
    original_rounds = rounds.copy()
    if len(rounds) > 100:
        if rounds[0] == 0:
            rounds = rounds[1:101]
        else:
            rounds = rounds[:100]
    
    # Plot parameter error
    fig, ax = plt.subplots(1, 1, figsize=INDIVIDUAL_PLOT_SIZE)
    
    metric_data = data.get('parameter_errors', [])
    
    if metric_data:
        if len(metric_data) == len(original_rounds) - 1:
            if len(original_rounds) > 100:
                metric_data = metric_data[:100]
        elif len(metric_data) == len(original_rounds):
            if len(original_rounds) > 100:
                if original_rounds[0] == 0:
                    metric_data = metric_data[1:101]
                else:
                    metric_data = metric_data[:100]
        
        plot_rounds = rounds[:len(metric_data)]
        plot_data = metric_data[:len(plot_rounds)]
        
        if plot_data and plot_rounds:
            ax.plot(plot_rounds, plot_data, label='Parameter Error', 
                   color='blue', linewidth=3, alpha=0.9, marker='o', markersize=4)
            
            ax.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
            ax.set_ylabel(r'$||\bar{w}_{n+1} - \bar{w}_n||$', fontsize=ACADEMIC_LABEL_SIZE)
            ax.set_title(rf'Parameter Error - {dataset.upper()}', 
                        fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"{algorithm}_parameter_error.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=ACADEMIC_DPI, bbox_inches='tight')
            print(f"Saved: {filepath}")
        else:
            print(f"Skipped parameter error - no data to plot")
    else:
        print(f"Skipped parameter error - no data found")
    
    plt.close()


def plot_cross_algorithm_comparison(metrics_data: Dict[str, Dict[str, List]], 
                                   output_dir: str, experiment_type: str, dataset: str):
    """Plot cross-algorithm comparison for specific metrics"""
    
    comparison_dir = os.path.join(output_dir, 'cross_algorithm_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_metrics = [
        ('global_test_accuracy', 'Global Test Accuracy'),
        ('global_test_loss', 'Global Test Loss'),
        ('parameter_error', 'Parameter Error')
    ]
    
    algorithm_data = {}
    for alg_key, data in metrics_data.items():
        if data.get('rounds'):
            algorithm_data[alg_key] = data
    
    if not algorithm_data:
        print("Warning: No algorithm data found for cross-algorithm comparison")
        return
    
    algorithm_colors = {
        'fedavg': '#1f77b4',
        'fedprox': '#ff7f0e',
        'fednova': '#2ca02c',
        'stochastic': '#d62728'
    }
    
    variant_styles = {
        'constant': '-',
        'tapered': '--'
    }
    
    for metric_key, metric_title in comparison_metrics:
        fig, ax = plt.subplots(1, 1, figsize=CROSS_ALGORITHM_SIZE)
        
        lines_plotted = 0
        
        for alg_key, alg_data in algorithm_data.items():
            rounds = alg_data.get('rounds', [])
            variant = alg_data.get('variant', 'constant')
            algorithm_base = alg_data.get('algorithm_base', '')
            
            if not rounds:
                continue
            
            original_rounds = rounds.copy()
            if len(rounds) > 100:
                if rounds[0] == 0:
                    rounds = rounds[1:101]
                else:
                    rounds = rounds[:100]
            
            metric_data = None
            
            if metric_key == 'global_test_accuracy':
                metric_data = alg_data.get('global_test_accuracy', [])
            elif metric_key == 'global_test_loss':
                metric_data = alg_data.get('global_test_loss', [])
            elif metric_key == 'parameter_error':
                metric_data = alg_data.get('parameter_errors', [])
            
            if metric_data:
                if len(metric_data) == len(original_rounds) - 1:
                    if len(original_rounds) > 100:
                        metric_data = metric_data[:100]
                elif len(metric_data) == len(original_rounds):
                    if len(original_rounds) > 100:
                        if original_rounds[0] == 0:
                            metric_data = metric_data[1:101]
                        else:
                            metric_data = metric_data[:100]
                elif len(metric_data) > len(rounds):
                    metric_data = metric_data[:len(rounds)]
                
                plot_rounds = rounds[:len(metric_data)]
                plot_data = metric_data[:len(plot_rounds)]
                
                if plot_data and plot_rounds:
                    color = algorithm_colors.get(algorithm_base, '#000000')
                    style = variant_styles.get(variant, '-')
                    
                    alg_display = algorithm_base.replace('_', ' ').title()
                    if alg_display == 'Fedavg':
                        alg_display = 'FedAvg'
                    elif alg_display == 'Fedprox':
                        alg_display = 'FedProx'
                    elif alg_display == 'Fednova':
                        alg_display = 'FedNova'
                    
                    if algorithm_base == 'stochastic':
                        label = 'Stochastic'
                    else:
                        label = f'{alg_display} ({variant.title()})'
                    
                    ax.plot(plot_rounds, plot_data, label=label, color=color, 
                           linestyle=style, linewidth=2, alpha=0.8, marker='o', markersize=3)
                    lines_plotted += 1
        
        if lines_plotted > 0:
            ax.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
            
            # Use mathematical notation for parameter error
            if metric_key == 'parameter_error':
                ax.set_ylabel(r'$||\bar{w}_{n+1} - \bar{w}_n||$', fontsize=ACADEMIC_LABEL_SIZE)
                ax.set_title(rf'Parameter Error - Cross-Algorithm Comparison - {dataset.upper()}', 
                            fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
            else:
                ax.set_ylabel(metric_title, fontsize=ACADEMIC_LABEL_SIZE)
                ax.set_title(f'{metric_title} - Cross-Algorithm Comparison - {dataset.upper()}', 
                            fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
            
            format_axis_decimals(ax, metric_key)
            adjust_accuracy_ylimits(ax, metric_key)
            
            handles, labels = ax.get_legend_handles_labels()
            if handles and len(handles) > 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=ACADEMIC_LEGEND_SIZE)
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"{metric_key}_cross_algorithm_comparison.png"
            filepath = os.path.join(comparison_dir, filename)
            plt.savefig(filepath, dpi=ACADEMIC_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"Saved cross-algorithm comparison: {filepath}")
        else:
            print(f"Skipped {metric_key} - no data to plot")
            plt.close()


def plot_cross_algorithm_paired_metrics(metrics_data: Dict[str, Dict[str, List]], 
                                       output_dir: str, experiment_type: str, dataset: str,
                                       metric_type: str):
    """Plot cross-algorithm comparison for paired accuracy and loss metrics in 1x2 layout"""
    
    comparison_dir = os.path.join(output_dir, 'cross_algorithm_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    algorithm_data = {}
    for alg_key, data in metrics_data.items():
        if data.get('rounds'):
            algorithm_data[alg_key] = data
    
    if not algorithm_data:
        print("Warning: No algorithm data found for cross-algorithm comparison")
        return
    
    algorithm_colors = {
        'fedavg': '#1f77b4',
        'fedprox': '#ff7f0e',
        'fednova': '#2ca02c',
        'stochastic': '#d62728'
    }
    
    variant_styles = {
        'constant': '-',
        'tapered': '--'
    }
    
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=CROSS_ALGORITHM_SIZE)
    
    if metric_type == 'global_test':
        acc_title = 'Global Test Accuracy'
        loss_title = 'Global Test Loss'
        acc_field = 'global_test_accuracy'
        loss_field = 'global_test_loss'
    
    lines_plotted_acc = 0
    lines_plotted_loss = 0
    
    for alg_key, alg_data in algorithm_data.items():
        rounds = alg_data.get('rounds', [])
        variant = alg_data.get('variant', 'constant')
        algorithm_base = alg_data.get('algorithm_base', '')
        
        if not rounds:
            continue
        
        original_rounds = rounds.copy()
        if len(rounds) > 100:
            if rounds[0] == 0:
                rounds = rounds[1:101]
            else:
                rounds = rounds[:100]
        
        acc_data = alg_data.get(acc_field, [])
        loss_data = alg_data.get(loss_field, [])
        
        # Plot accuracy data
        if acc_data:
            if len(acc_data) == len(original_rounds):
                if len(original_rounds) > 100:
                    if original_rounds[0] == 0:
                        acc_data = acc_data[1:101]
                    else:
                        acc_data = acc_data[:100]
            elif len(acc_data) > len(rounds):
                acc_data = acc_data[:len(rounds)]
            
            plot_rounds = rounds[:len(acc_data)]
            plot_data = acc_data[:len(plot_rounds)]
            
            if plot_data and plot_rounds:
                color = algorithm_colors.get(algorithm_base, '#000000')
                style = variant_styles.get(variant, '-')
                
                alg_display = algorithm_base.replace('_', ' ').title()
                if alg_display == 'Fedavg':
                    alg_display = 'FedAvg'
                elif alg_display == 'Fedprox':
                    alg_display = 'FedProx'
                elif alg_display == 'Fednova':
                    alg_display = 'FedNova'
                
                if algorithm_base == 'stochastic':
                    label = 'Stochastic'
                else:
                    label = f'{alg_display} ({variant.title()})'
                
                ax_acc.plot(plot_rounds, plot_data, label=label, color=color, 
                           linestyle=style, linewidth=2, alpha=0.8, marker='o', markersize=3)
                lines_plotted_acc += 1
        
        # Plot loss data
        if loss_data:
            if len(loss_data) == len(original_rounds):
                if len(original_rounds) > 100:
                    if original_rounds[0] == 0:
                        loss_data = loss_data[1:101]
                    else:
                        loss_data = loss_data[:100]
            elif len(loss_data) > len(rounds):
                loss_data = loss_data[:len(rounds)]
            
            plot_rounds = rounds[:len(loss_data)]
            plot_data = loss_data[:len(plot_rounds)]
            
            if plot_data and plot_rounds:
                color = algorithm_colors.get(algorithm_base, '#000000')
                style = variant_styles.get(variant, '-')
                
                alg_display = algorithm_base.replace('_', ' ').title()
                if alg_display == 'Fedavg':
                    alg_display = 'FedAvg'
                elif alg_display == 'Fedprox':
                    alg_display = 'FedProx'
                elif alg_display == 'Fednova':
                    alg_display = 'FedNova'
                
                if algorithm_base == 'stochastic':
                    label = 'Stochastic'
                else:
                    label = f'{alg_display} ({variant.title()})'
                
                ax_loss.plot(plot_rounds, plot_data, label=label, color=color, 
                            linestyle=style, linewidth=2, alpha=0.8, marker='o', markersize=3)
                lines_plotted_loss += 1
    
    if lines_plotted_acc > 0:
        ax_acc.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
        ax_acc.set_ylabel(acc_title, fontsize=ACADEMIC_LABEL_SIZE)
        ax_acc.set_title(acc_title, fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
        ax_acc.grid(True, alpha=0.3)
        handles, labels = ax_acc.get_legend_handles_labels()
        if len(handles) > 1:
            ax_acc.legend(fontsize=ACADEMIC_LEGEND_SIZE, loc='best')
        format_axis_decimals(ax_acc, 'accuracy')
        adjust_accuracy_ylimits(ax_acc, 'accuracy')
    
    if lines_plotted_loss > 0:
        ax_loss.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
        ax_loss.set_ylabel(loss_title, fontsize=ACADEMIC_LABEL_SIZE)
        ax_loss.set_title(loss_title, fontsize=ACADEMIC_TITLE_SIZE, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        handles, labels = ax_loss.get_legend_handles_labels()
        if len(handles) > 1:
            ax_loss.legend(fontsize=ACADEMIC_LEGEND_SIZE, loc='best')
        format_axis_decimals(ax_loss, 'loss')
    
    fig.suptitle(f'{metric_type.replace("_", " ").title()} Metrics - Cross-Algorithm Comparison - {dataset.upper()}', 
                fontsize=ACADEMIC_TITLE_SIZE + 1, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    filename = f"{metric_type}_cross_algorithm_comparison.png"
    filepath = os.path.join(comparison_dir, filename)
    plt.savefig(filepath, dpi=ACADEMIC_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cross-algorithm paired comparison: {filepath}")


def generate_all_plots(results_path: str, experiment_type: str, dataset: str, output_dir: str):
    """Generate all plots"""
    
    print(f"Generating plots for: {experiment_type} - {dataset}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        experiment_data = load_experiment_data(results_path, experiment_type, dataset)
        metrics_data = extract_metrics_over_rounds(experiment_data)
        
        if not metrics_data:
            print("No metrics data found")
            return
        
        print(f"Found data for algorithms: {list(metrics_data.keys())}")
        
        individual_dir = os.path.join(output_dir, 'individual')
        os.makedirs(individual_dir, exist_ok=True)
        
        for algorithm in metrics_data.keys():
            plot_paired_metrics(metrics_data, algorithm, individual_dir, experiment_type, dataset, 'training')
            plot_paired_metrics(metrics_data, algorithm, individual_dir, experiment_type, dataset, 'validation')
            plot_paired_metrics(metrics_data, algorithm, individual_dir, experiment_type, dataset, 'global_test')
            plot_individual_algorithm(metrics_data, algorithm, individual_dir, experiment_type, dataset)
        
        plot_cross_algorithm_paired_metrics(metrics_data, output_dir, experiment_type, dataset, 'global_test')
        plot_cross_algorithm_comparison(metrics_data, output_dir, experiment_type, dataset)
        
        print(f"All plots generated in: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-type', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--results-path', default='./results')
    parser.add_argument('--output-dir', default=None)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_path, 'plots', args.experiment_type, args.dataset)
    
    generate_all_plots(args.results_path, args.experiment_type, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()