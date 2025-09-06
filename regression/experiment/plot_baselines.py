import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import argparse
import pickle

# Academic paper settings for compact plots
plt.style.use('seaborn-v0_8-whitegrid')
ACADEMIC_DPI = 300
ACADEMIC_FONT_SIZE = 10
ACADEMIC_TITLE_SIZE = 12
ACADEMIC_LABEL_SIZE = 10
ACADEMIC_LEGEND_SIZE = 11

# Journal-optimized figure sizes
COMPARISON_1X3_SIZE = (12, 3)
INDIVIDUAL_PLOT_SIZE = (6, 3)

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

def plot_parameter_convergence_n5(results, save_dir, plot_title, include_stochastic=False):
    """Plot 1x3 parameter convergence for N=5 with all algorithms"""
    print("Creating parameter convergence plot...")
    
    plot_order = ['FedNova', 'FedProx', 'FedAvg']
    if include_stochastic:
        plot_order.append('Stochastic')
    
    # Get true_theta if available
    true_theta = None
    for method_name in plot_order:
        if (method_name in results and 'true_theta' in results[method_name] and 
            results[method_name]['true_theta'] is not None):
            true_theta = results[method_name]['true_theta']
            break
    
    fig, axes = plt.subplots(1, 3, figsize=COMPARISON_1X3_SIZE)
    fig.suptitle(plot_title)
    
    component_colors = ['blue', 'green', 'purple']
    component_names = ['Component 1', 'Component 2', 'Component 3']
    
    handles, labels = [], []
    
    for i in range(3):  # For each parameter component
        ax = axes[i]
        
        # Plot each algorithm
        for method_name in plot_order:
            if method_name in results:
                trajectory = np.array(results[method_name]['global_theta_trajectory'])
                rounds = np.arange(1, len(trajectory) + 1)
                line = ax.plot(rounds, trajectory[:, i], label=method_name, 
                              alpha=0.8, linewidth=1.5, marker='o', markersize=1.5)[0]
                
                # Collect legend from first subplot only
                if i == 0:
                    handles.append(line)
                    labels.append(method_name)
        
        # Plot true theta if available
        if true_theta is not None:
            true_line = ax.axhline(y=true_theta[i], color=component_colors[i], linestyle='--', 
                                  linewidth=1.5, alpha=0.7, label=f'$w^*[{i+1}]$')
            
            # Collect true theta legend from each subplot
            if i == 0:
                handles.append(true_line)
                labels.append(f'$w^*[1]$')
            elif i == 1:
                handles.append(true_line)
                labels.append(f'$w^*[2]$')
            elif i == 2:
                handles.append(true_line)
                labels.append(f'$w^*[3]$')
        
        ax.set_title(f'{component_names[i]}')
        ax.set_xlabel('Rounds')
        ax.set_ylabel(rf'$\bar{{w}}_{{nN}}[{i+1}]$')
        ax.grid(True, alpha=0.3)
        format_axis_decimals(ax, 'parameter')
    
    # Add shared legend at bottom
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), 
               bbox_to_anchor=(0.5, -0.08), fontsize=ACADEMIC_LEGEND_SIZE)
    
    configure_academic_plot(fig, axes)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(os.path.join(save_dir, 'parameter_convergence_n5.png'), dpi=ACADEMIC_DPI, bbox_inches='tight')
    plt.close()
    print("Saved parameter_convergence_n5.png")

def plot_gradient_norm_convergence_n5(results, save_dir, plot_title, include_stochastic=False):
    """Plot 1x1 collective gradient norm convergence for N=5 with all algorithms"""
    print("Creating gradient norm convergence plot...")
    
    plot_order = ['FedNova', 'FedProx', 'FedAvg']
    if include_stochastic:
        plot_order.append('Stochastic')
    
    fig, ax = plt.subplots(figsize=INDIVIDUAL_PLOT_SIZE)
    ax.set_title(plot_title)
    
    for method_name in plot_order:
        if method_name in results:
            trajectory = np.array(results[method_name]['summed_gradient_trajectory'])
            gradient_norms = np.linalg.norm(trajectory, axis=1)
            rounds = np.arange(1, len(gradient_norms) + 1)
            ax.plot(rounds, gradient_norms, label=method_name, 
                   alpha=0.8, linewidth=1.5, marker='o', markersize=1.5)
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel(r'$||\sum_{i=1}^{10} p^{(i)}h^{(i)}(\bar{w}_{nN})||$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    format_axis_decimals(ax, 'gradient')
    
    configure_academic_plot(fig, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_norm_convergence_n5.png'), dpi=ACADEMIC_DPI, bbox_inches='tight')
    plt.close()
    print("Saved gradient_norm_convergence_n5.png")

def generate_n5_plots(pickle_file, save_dir, include_stochastic=False):
    """Load N=5 results and generate the two required plots"""
    print("=== FEDERATED LEARNING N=5 BASELINE PLOTTING ===")
    print(f"Loading results from: {pickle_file}")
    print(f"Saving plots to: {save_dir}")
    print(f"Include stochastic algorithm: {include_stochastic}")
    
    # Load results
    with open(pickle_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Loaded results for local steps: {list(all_results.keys())}")
    
    # Extract N=5 results
    if 5 not in all_results:
        print("Error: No results found for N=5 (local_steps=5)")
        return
    
    results_n5 = all_results[5]
    print(f"Found algorithms for N=5: {list(results_n5.keys())}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get user input for plot titles
    print(f"\nEnter plot titles:")
    
    print("Parameter convergence plot title:")
    param_title = input("Title: ").strip()
    if not param_title:
        param_title = "Parameter Convergence (N=5)"
        print(f"Using default title: {param_title}")
    
    print("Gradient norm convergence plot title:")
    grad_title = input("Title: ").strip()
    if not grad_title:
        grad_title = "Collective Gradient Norm Convergence (N=5)"
        print(f"Using default title: {grad_title}")
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING N=5 PLOTS")
    print(f"{'='*60}")
    
    plot_parameter_convergence_n5(results_n5, save_dir, param_title, include_stochastic)
    plot_gradient_norm_convergence_n5(results_n5, save_dir, grad_title, include_stochastic)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY FOR N=5")
    print(f"{'='*60}")
    
    for method_name, method_results in results_n5.items():
        final_loss = method_results['global_loss_trajectory'][-1]
        final_summed_gradient_norm = np.linalg.norm(method_results['summed_gradient_trajectory'][-1])
        
        final_parameter_error = None
        if ('parameter_error_trajectory' in method_results and 
            len(method_results['parameter_error_trajectory']) > 0):
            final_parameter_error = method_results['parameter_error_trajectory'][-1]
        
        print(f"{method_name}:")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Final Summed Gradient Norm: {final_summed_gradient_norm:.6f}")
        if final_parameter_error is not None:
            print(f"  Final Parameter Error: {final_parameter_error:.6f}")
    
    # Save summary
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("N=5 RESULTS SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append("")
    
    for method_name, method_results in results_n5.items():
        final_loss = method_results['global_loss_trajectory'][-1]
        final_summed_gradient_norm = np.linalg.norm(method_results['summed_gradient_trajectory'][-1])
        
        final_parameter_error = None
        if ('parameter_error_trajectory' in method_results and 
            len(method_results['parameter_error_trajectory']) > 0):
            final_parameter_error = method_results['parameter_error_trajectory'][-1]
        
        summary_lines.append(f"{method_name}:")
        summary_lines.append(f"  Final Loss: {final_loss:.6f}")
        summary_lines.append(f"  Final Summed Gradient Norm: {final_summed_gradient_norm:.6f}")
        if final_parameter_error is not None:
            summary_lines.append(f"  Final Parameter Error: {final_parameter_error:.6f}")
        summary_lines.append("")
    
    summary_file = os.path.join(save_dir, 'n5_results_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSaved results summary to: {summary_file}")
    
    print(f"\nAll N=5 plots saved to: {save_dir}")
    print("Generated plots:")
    print("  - parameter_convergence_n5.png")
    print("  - gradient_norm_convergence_n5.png")
    print(f"  - n5_results_summary.txt")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Generate N=5 plots from federated learning baseline results')
    parser.add_argument('--pickle_file', default='./new_baselines_results_no_dominant/complete_results.pkl', 
                       help='Path to complete_results.pkl file')
    parser.add_argument('--save_dir', default='./plotting_output_n5', 
                       help='Directory to save plots')
    parser.add_argument('--include_stochastic', action='store_true', default=False,
                       help='Include stochastic algorithm data in plots (default: False)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pickle_file):
        print(f"Error: Pickle file not found: {args.pickle_file}")
        return
    
    generate_n5_plots(args.pickle_file, args.save_dir, args.include_stochastic)

if __name__ == "__main__":
    main()