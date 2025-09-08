import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Any

plt.style.use('seaborn-v0_8-whitegrid')
ACADEMIC_DPI = 300
ACADEMIC_FONT_SIZE = 10
ACADEMIC_TITLE_SIZE = 12
ACADEMIC_LABEL_SIZE = 10
ACADEMIC_LEGEND_SIZE = 9

PLOT_SIZE = (12, 5)


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
    """Load experiment data from structured results directory - only stochastic"""
    
    experiment_dir = os.path.join(results_path, experiment_type, dataset)
    
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    # Only load stochastic
    algorithm_dir = os.path.join(experiment_dir, "stochastic")
    
    if not os.path.exists(algorithm_dir):
        raise ValueError(f"Stochastic directory not found: {algorithm_dir}")
    
    # Load main results
    for filename in os.listdir(algorithm_dir):
        if filename.startswith('results_') and filename.endswith('.pkl'):
            results_file = os.path.join(algorithm_dir, filename)
            with open(results_file, 'rb') as f:
                loaded_data = pickle.load(f)
                print(f"Loaded {results_file}")
                return loaded_data
    
    raise ValueError(f"No results file found in {algorithm_dir}")


def calculate_global_accuracy_excluding_rare(class_accuracies_list: List[Dict], 
                                           class_samples: Dict, 
                                           rare_class: int) -> List[float]:
    """
    Calculate global test accuracy excluding the rare class.
    
    Args:
        class_accuracies_list: List of per-round class accuracies (dict with class_id -> accuracy)
        class_samples: Dict with class_id -> number of samples for each class
        rare_class: The rare class to exclude
    
    Returns:
        List of global accuracies excluding the rare class for each round
    """
    global_acc_excluding_rare = []
    
    for round_class_accs in class_accuracies_list:
        if not isinstance(round_class_accs, dict):
            global_acc_excluding_rare.append(0.0)
            continue
        
        total_samples = 0
        weighted_acc = 0.0
        
        for class_id_str, accuracy in round_class_accs.items():
            class_id = int(class_id_str)
            if class_id != rare_class and class_id_str in class_samples:
                samples = class_samples[class_id_str]
                weighted_acc += accuracy * samples
                total_samples += samples
        
        if total_samples > 0:
            global_acc_excluding_rare.append(weighted_acc / total_samples)
        else:
            global_acc_excluding_rare.append(0.0)
    
    return global_acc_excluding_rare


def extract_metrics(experiment_data: Dict[str, Any], rare_class: int) -> Dict[str, List]:
    """Extract the three required metrics from experiment data"""
    
    if 'training_history' not in experiment_data:
        raise ValueError("No training_history found in experiment data")
    
    history = experiment_data['training_history']
    
    # Extract basic data
    rounds = history.get('round', [])
    global_test_accuracy = history.get('global_test_accuracy', [])
    global_test_class_accuracies = history.get('global_test_class_accuracies', [])
    
    # Extract rare class accuracy
    rare_class_accuracy = []
    if global_test_class_accuracies:
        for round_class_accs in global_test_class_accuracies:
            if isinstance(round_class_accs, dict) and str(rare_class) in round_class_accs:
                rare_class_accuracy.append(round_class_accs[str(rare_class)])
            else:
                rare_class_accuracy.append(0.0)
    
    # Calculate global accuracy excluding rare class
    # Note: This is a simplified calculation - you may need to provide actual class sample counts
    # For now, assuming equal samples per class for demonstration
    global_acc_excluding_rare = []
    if global_test_class_accuracies:
        for round_class_accs in global_test_class_accuracies:
            if isinstance(round_class_accs, dict):
                # Simple average of all classes except rare class
                non_rare_accs = [acc for class_id, acc in round_class_accs.items() 
                               if int(class_id) != rare_class]
                if non_rare_accs:
                    global_acc_excluding_rare.append(sum(non_rare_accs) / len(non_rare_accs))
                else:
                    global_acc_excluding_rare.append(0.0)
            else:
                global_acc_excluding_rare.append(0.0)
    
    return {
        'rounds': rounds,
        'global_test_accuracy': global_test_accuracy,
        'global_test_accuracy_excluding_rare': global_acc_excluding_rare,
        'rare_class_test_accuracy': rare_class_accuracy
    }


def plot_three_line_comparison(metrics: Dict[str, List], 
                             dataset: str, 
                             rare_class: int, 
                             output_path: str):
    """Plot the three-line comparison: global (with rare), global (without rare), rare class"""
    
    rounds = metrics['rounds']
    global_acc = metrics['global_test_accuracy']
    global_acc_no_rare = metrics['global_test_accuracy_excluding_rare']
    rare_class_acc = metrics['rare_class_test_accuracy']
    
    if not rounds:
        print("No rounds data available")
        return
    
    # Skip first 5 rounds and limit to 100 rounds
    start_idx = 5 if len(rounds) > 5 else len(rounds)
    end_idx = min(len(rounds), start_idx + 100)
    
    plot_rounds = rounds[start_idx:end_idx]
    
    # Adjust data arrays
    if len(global_acc) > start_idx:
        plot_global_acc = global_acc[start_idx:end_idx]
    else:
        plot_global_acc = []
    
    if len(global_acc_no_rare) > start_idx:
        plot_global_acc_no_rare = global_acc_no_rare[start_idx:end_idx]
    else:
        plot_global_acc_no_rare = []
    
    if len(rare_class_acc) > start_idx:
        plot_rare_class_acc = rare_class_acc[start_idx:end_idx]
    else:
        plot_rare_class_acc = []
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=PLOT_SIZE)
    
    # Plot global test accuracy (including rare)
    if plot_global_acc:
        ax.plot(plot_rounds[:len(plot_global_acc)], plot_global_acc, 
               label='Global Test Metric (Including Rare)', 
               color='blue', linewidth=3, alpha=0.9, marker='s', markersize=4)
    
    # Plot global test accuracy (excluding rare)
    if plot_global_acc_no_rare:
        ax.plot(plot_rounds[:len(plot_global_acc_no_rare)], plot_global_acc_no_rare, 
               label='Global Test Metric (Excluding Rare)', 
               color='green', linewidth=3, alpha=0.9, marker='^', markersize=4)
    
    # Plot rare class test accuracy
    if plot_rare_class_acc:
        ax.plot(plot_rounds[:len(plot_rare_class_acc)], plot_rare_class_acc, 
               label=f'Class {rare_class} Test Metric', 
               color='red', linewidth=3, alpha=0.9, marker='o', markersize=4)
    
    # Customize the plot
    ax.set_xlabel('Round', fontsize=ACADEMIC_LABEL_SIZE)
    ax.set_ylabel('Test Accuracy', fontsize=ACADEMIC_LABEL_SIZE)
    ax.set_title(f'Global vs Rare Class - {dataset.upper()}', 
                fontsize=ACADEMIC_TITLE_SIZE + 1, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=ACADEMIC_LEGEND_SIZE, loc='best')
    
    # Format axis
    format_axis_decimals(ax, 'accuracy')
    adjust_accuracy_ylimits(ax, 'accuracy')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=ACADEMIC_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot global vs rare class comparison')
    parser.add_argument('--experiment-type', required=True, help='Experiment type')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--rare-class', type=int, required=True, help='Rare class number (0-9)')
    parser.add_argument('--results-path', default='./results', help='Path to results directory')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not (0 <= args.rare_class <= 9):
        raise ValueError("Rare class must be between 0 and 9")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_path, 'plots', args.experiment_type, 
                                      args.dataset, f'rare_class_{args.rare_class}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print(f"Loading data for: {args.experiment_type} - {args.dataset} - Class {args.rare_class}")
        
        # Load experiment data
        experiment_data = load_experiment_data(args.results_path, args.experiment_type, args.dataset)
        
        # Extract the three required metrics
        metrics = extract_metrics(experiment_data, args.rare_class)
        
        # Generate the plot
        output_path = os.path.join(args.output_dir, f"global_vs_class_{args.rare_class}_comparison.png")
        plot_three_line_comparison(metrics, args.dataset, args.rare_class, output_path)
        
        print(f"Plot generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()