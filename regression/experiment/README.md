### Federated Linear Regression Experiments

This folder contains scripts to run and visualize baseline federated learning (FL) experiments for linear regression on synthetic multi‑client datasets.

### Contents
- `baselines_comparison_study.py`: Runs Stochastic SGD, FedAvg, FedProx, and FedNova across multiple local step sizes and saves results.
- `plot_baseline_results.py`: Loads saved results and generates publication‑ready plots and summaries.
- `utils.py`: Common math utilities (SGD update, losses, gradients).
 - Ablation runners and plotters:
   - `ablation_study_single_dominant.py`, `plot_single_dominant.py`
   - `ablation_study_no_dominant.py`, `plot_no_dominant.py`
   - `ablation_study_dual_dominant.py`, `plot_dual_dominant.py`

### Prerequisites
- Python 3.12+
- Python packages as mentioned in the Root Readme
- A dataset JSON in the structure produced by `data_generation` (see `../data_generation/Readme.md`).

### Expected Dataset Format
The runner expects a JSON with:
- Top‑level `clients_data` mapping client ids to `{"theta", "X", "Y", "length"}`
- Optional `metadata.reference_theta` (used for parameter‑error plots/summaries)

Example location (from this repo):
- `data_generation/federated_datasets_combined_hetero/normal_theta_5_xvar_5.json`

### Run: Baseline Comparison
Runs all methods for each configured local step size, then saves per‑step and combined results.

Example:
```bash
python baselines_comparison_study.py \
  --json_file ../data_generation/federated_datasets_combined_hetero/normal_theta_5_xvar_5.json \
  --save_dir  baselines_results \
  --rounds 2000
```

Flags:
- `--json_file`: Path to dataset JSON
- `--save_dir`: Output directory for results
- `--rounds`: Number of FL rounds (overrides script default)
- `--no_tapering`: Disable learning‑rate tapering

What gets saved (per `--save_dir`):
- `results_local_steps_{N}.pkl` for each N in the configured step sizes
- `complete_results.pkl` containing all step sizes and methods
- `results_summary.txt` with a human‑readable summary

Per‑method result dict includes:
- `global_theta_trajectory`: list of global parameter vectors per round
- `individual_gradient_trajectories`: dict[client_id] -> list of gradients per round
- `summed_gradient_trajectory`: list of sum of all client gradients per round
- `global_loss_trajectory`: list of global MSE per round
- `local_gradient_trajectories`, `local_thetas_trajectories`, `client_loss_trajectories`
- `parameter_error_trajectory` (present if `reference_theta` provided)
- `true_theta` (echo of `reference_theta` if provided)

### Run: Plotting
Generates per‑step plots and cross‑step comparison plots from `complete_results.pkl`.

Example:
```bash
python plot_baseline_results.py \
  --pickle_file baselines_results/complete_results.pkl \
  --save_dir  plotting_output
```

### Run: Single‑Dominant Ablation
```bash
python ablation_study_single_dominant.py
# Non‑interactive default runs both studies (heterogeneity + tapering)
```
Outputs:
- Heterogeneity study: `heterogeneity_study_single_dominant/{combined_hetero|feature_variance|param_variance|target_variance}/progressive_results.pkl` and `single_heterogeneous_results.pkl`
- Tapering study: `tapering_study_single_dominant/tapering_results.pkl`

### Plot: Single‑Dominant
```bash
python plot_single_dominant.py \
  heterogeneity_study_single_dominant/combined_hetero \
  --output-dir plots_single_dominant \
```

### Run: No‑Dominant Ablation
```bash
python ablation_study_no_dominant.py
```
Outputs:
- Heterogeneity study: `heterogeneity_study_no_dominant/{...}/progressive_results.pkl` and `single_heterogeneous_results.pkl`
- Tapering study: `tapering_study_no_dominant/tapering_results_no_dominant.pkl`

### Plot: No‑Dominant
```bash
python plot_no_dominant.py \
  heterogeneity_study_no_dominant/combined_hetero \
  --output-dir plots_no_dominant \
```

### Run: Dual‑Dominant Ablation
```bash
python ablation_study_dual_dominant.py
```
Outputs:
- Heterogeneity study: `heterogeneity_study_dual_dominant/{...}/progressive_results.pkl` and `single_heterogeneous_results.pkl`
- Tapering study: `tapering_study_dual_dominant/tapering_results_dual_dominant.pkl`

### Plot: Dual‑Dominant
```bash
python plot_dual_dominant.py \
  heterogeneity_study_dual_dominant/combined_hetero \
  --output-dir plots_dual_dominant \
```

Notes on ablation outputs
- Each pickle key is a parameter value (dataset name or exponent), mapping to N→result‑dict.
- Result dicts include trajectories for `global_theta`, parameter error, per‑client gradients/ norms, and final metrics; exact keys differ slightly by study (e.g., `sum_gradients_*` for no‑dominant, `combined_gradient_*` for dual‑dominant).
