import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import sys
import argparse
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
ACADEMIC_DPI = 300
ACADEMIC_FONT_SIZE = 10
ACADEMIC_TITLE_SIZE = 12
ACADEMIC_LABEL_SIZE = 10
ACADEMIC_LEGEND_SIZE = 10

# Focus only on N=5
N_VALUES = [5]
NUM_CLIENTS = 10

# Journal-optimized figure sizes
COMPARISON_1X2_SIZE = (8, 3)

def format_axis_decimals(ax, metric_key):
    """Format y-axis to show maximum 2 decimal places for loss and accuracy metrics"""
    if any(keyword in metric_key.lower() for keyword in ['loss', 'accuracy']):
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

def configure_academic_plot(fig, axes=None, title_size=None, label_size=None, legend_size=None):
    """Configure a plot with academic formatting settings"""
    title_size = title_size or ACADEMIC_TITLE_SIZE
    label_size = label_size or ACADEMIC_LABEL_SIZE
    legend_size = legend_size or ACADEMIC_LEGEND_SIZE
    
    if hasattr(fig, 'suptitle') and fig._suptitle:
        fig._suptitle.set_fontsize(title_size)
    
    if axes is not None:
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        for ax in axes:
            if ax is not None:
                ax.title.set_fontsize(title_size)
                ax.xaxis.label.set_fontsize(label_size)
                ax.yaxis.label.set_fontsize(label_size)
                ax.tick_params(axis='both', which='major', labelsize=ACADEMIC_FONT_SIZE)
                
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_fontsize(legend_size)

def get_distribution_label(param_val):
    """Helper function to get proper distribution label"""
    if isinstance(param_val, str):
        if 'uniform' in param_val:
            spread = param_val.split('_')[-1].replace('.json', '')
            return f'Uniform {spread}'
        elif 'normal' in param_val:
            if 'theta' in param_val and 'xvar' in param_val:
                parts = param_val.replace('.json', '').split('_')
                theta_val = parts[2] if len(parts) > 2 else 'N/A'
                xvar_val = parts[4] if len(parts) > 4 else 'N/A'
                return f'W~$\\mathcal{{N}}$(0,{theta_val}²), X~$\\mathcal{{N}}$(0,{xvar_val}²)'
            elif 'snr' in param_val:
                snr_val = param_val.split('_')[-1].replace('.json', '')
                return f'SNR-{snr_val}dB'
            else:
                spread = param_val.split('_')[-1].replace('.json', '')
                return f'$\\mathcal{{N}}$(0,{spread}²)'
        elif 'heterogeneous' in param_val:
            return 'Heterogeneous'
        else:
            spread = param_val.split('_')[-1].replace('.json', '')
            return f'Dataset {spread}'
    else:
        display_val = 0.760 if abs(param_val - 0.750) < 0.001 else param_val
        return f'$\\delta$={display_val:.3f}'

def plot_comparative_analysis(results, parameter_values, save_dir, overall_title, skip_initial_rounds=5):
    """Create comparative 1x2 plots across experiments"""
    N = 5
    colors = plt.cm.viridis(np.linspace(0, 1, len(parameter_values)))
    
    # Comparative Plot: Parameter Error, Gradient Convergence
    fig, axes = plt.subplots(1, 2, figsize=COMPARISON_1X2_SIZE)
    fig.suptitle(overall_title)
    
    handles, labels = [], []
    
    # Parameter Error Comparison
    ax1 = axes[0]
    for param_idx, param_val in enumerate(parameter_values):
        if param_val in results and N in results[param_val]:
            parameter_errors = results[param_val][N]['parameter_error_trajectory']
            parameter_errors_plot = parameter_errors[skip_initial_rounds:]
            rounds_plot = np.arange(skip_initial_rounds + 1, len(parameter_errors) + 1)
            label = get_distribution_label(param_val)
            line = ax1.plot(rounds_plot, parameter_errors_plot, 
                   label=label, color=colors[param_idx], alpha=0.8, linewidth=1.5)[0]
            handles.append(line)
            labels.append(label)
    
    ax1.set_title('Parameter Error Comparison')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel(r'$||\bar{w}_{nN} - w^*||$')
    ax1.grid(True, alpha=0.3)
    format_axis_decimals(ax1, 'error')
    
    # Gradient Convergence Comparison
    ax2 = axes[1]
    for param_idx, param_val in enumerate(parameter_values):
        if param_val in results and N in results[param_val]:
            sum_gradient_norms = results[param_val][N]['sum_gradients_norm_trajectory']
            sum_gradient_norms_plot = sum_gradient_norms[skip_initial_rounds:]
            rounds_plot = np.arange(skip_initial_rounds + 1, len(sum_gradient_norms) + 1)
            label = get_distribution_label(param_val)
            ax2.plot(rounds_plot, sum_gradient_norms_plot, 
                    label=label, color=colors[param_idx], alpha=0.8, linewidth=1.5)
    
    ax2.set_title('Collective Gradient Norm Comparison')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel(r'$||\sum_{i=1}^{10} p^{(i)}h^{(i)}(\bar{w}_{nN})||$')
    ax2.grid(True, alpha=0.3)
    format_axis_decimals(ax2, 'gradient')
    
    # Add shared legend at bottom
    fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4), 
               bbox_to_anchor=(0.5, -0.12), fontsize=ACADEMIC_LEGEND_SIZE)
    
    configure_academic_plot(fig, axes)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f"{save_dir}/comparative_analysis.png", dpi=ACADEMIC_DPI, bbox_inches='tight')
    plt.close()

def plot_progressive_results_n5_comparative_only(results, save_dir, study_name, parameter_values, overall_title, skip_initial_rounds=5):
    """Main plotting function for N=5 progressive results - comparative plots only"""
    print(f"Generating {study_name} comparative plots for N=5 only")
    print(f"Skipping first {skip_initial_rounds} rounds...")
    print(f"Processing {len(parameter_values)} experiments")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create only comparative plots
    print("Creating comparative plots across experiments...")
    plot_comparative_analysis(results, parameter_values, save_dir, overall_title, skip_initial_rounds)
    
    print(f"{study_name} comparative plots saved in {save_dir}")

def find_pickle_files(folder_path):
    """Find all pickle files in the given folder"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return []
    
    pickle_files = list(folder.glob("*.pkl"))
    if not pickle_files:
        print(f"No pickle files found in {folder_path}")
        return []
    
    print(f"Found {len(pickle_files)} pickle files:")
    for pf in pickle_files:
        print(f"  - {pf.name}")
    
    return pickle_files

def load_results_from_pickle(pickle_path):
    """Load results from a pickle file"""
    try:
        with open(pickle_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded results from {pickle_path}")
        return results
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None

def determine_study_info(pickle_path):
    """Determine study name from pickle file path"""
    path = Path(pickle_path)
    
    if 'progressive' in path.name.lower():
        study_name = "Progressive Heterogeneity Study"
    elif 'taper' in path.name.lower():
        study_name = "Tapering Study"
    else:
        study_name = "Study"
    
    return study_name

def is_tapering_study(pickle_path):
    """Check if this is a tapering study"""
    return 'taper' in str(pickle_path).lower()

def main():
    parser = argparse.ArgumentParser(description='Plot N=5 Progressive/Tapering comparative results from pickle files')
    parser.add_argument('folder', help='Folder containing pickle files')
    parser.add_argument('--skip-rounds', type=int, default=5, 
                        help='Number of initial rounds to skip in trajectory plots (default: 5)')
    parser.add_argument('--output-dir', help='Output directory for plots (default: folder/plots_n5_comparative)')
    parser.add_argument('--tapering-first-n', type=int, default=5,
                        help='For tapering studies: use only first N parameter values sorted numerically (default: 5)')
    
    args = parser.parse_args()
    
    pickle_files = find_pickle_files(args.folder)
    if not pickle_files:
        return
    
    # Filter for progressive and tapering results
    target_files = []
    for pickle_path in pickle_files:
        if 'progressive' in str(pickle_path).lower() or 'taper' in str(pickle_path).lower():
            target_files.append(pickle_path)
    
    if not target_files:
        print("No progressive or tapering pickle files found!")
        return
    
    print(f"Found {len(target_files)} target pickle files")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.folder, "plots_n5_comparative")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    for pickle_path in target_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pickle_path.name}")
        print(f"{'='*60}")
        
        results = load_results_from_pickle(pickle_path)
        if results is None:
            continue
        
        # Check if N=5 data exists
        has_n5_data = False
        for param_val, param_data in results.items():
            if 5 in param_data:
                has_n5_data = True
                break
        
        if not has_n5_data:
            print(f"No N=5 data found in {pickle_path.name}, skipping...")
            continue
        
        study_name = determine_study_info(pickle_path)
        parameter_values = list(results.keys())
        
        # Handle tapering studies - limit to first N experiments
        if is_tapering_study(pickle_path):
            original_count = len(parameter_values)
            if args.tapering_first_n > 0 and args.tapering_first_n < len(parameter_values):
                try:
                    # Try to sort numerically
                    parameter_values_sorted = sorted(parameter_values, key=float)
                    parameter_values = parameter_values_sorted[:args.tapering_first_n]
                    print(f"Tapering study: Using first {args.tapering_first_n} of {original_count} parameters")
                except (ValueError, TypeError):
                    # If numerical sorting fails, just take first N as-is
                    parameter_values = parameter_values[:args.tapering_first_n]
                    print(f"Tapering study: Using first {args.tapering_first_n} of {original_count} parameters (unsorted)")
            elif args.tapering_first_n <= 0:
                print(f"Warning: Invalid --tapering-first-n value {args.tapering_first_n}, using all parameters")
            else:
                print(f"Tapering study: Requested {args.tapering_first_n} parameters but only {len(parameter_values)} available, using all")
        
        print(f"Found {len(parameter_values)} parameter values: {parameter_values}")
        
        # Ask for overall plot title
        print(f"\nEnter overall plot title for {pickle_path.name}:")
        overall_title = input("Overall title: ").strip()
        if not overall_title:
            overall_title = f"{study_name}"
            print(f"Using default title: {overall_title}")
        
        pickle_output_dir = os.path.join(output_dir, pickle_path.stem)
        
        plot_progressive_results_n5_comparative_only(
            results=results,
            save_dir=pickle_output_dir,
            study_name=study_name,
            parameter_values=parameter_values,
            overall_title=overall_title,
            skip_initial_rounds=args.skip_rounds
        )
        
        print(f"Completed plotting for {pickle_path.name}")
    
    print(f"\n{'='*60}")
    print("All N=5 comparative plotting completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()