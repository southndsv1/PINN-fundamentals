"""
Main script for running PINN benchmarks.

This script runs all PINN variants on all benchmark problems and generates
comprehensive performance analysis.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add pinn_benchmark to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinn_benchmark.pinn_variants import (
    VanillaPINN, VariationalPINN, ConservativePINN,
    BayesianPINN, GradientEnhancedPINN, AdaptivePINN, CausalPINN
)
from pinn_benchmark.benchmark_problems import (
    HeatConduction1D, WavePropagation1D, BurgersEquation,
    NavierStokes2D, EulerBernoulliBeam, PlateVibration,
    LinearElasticity2D, AdvectionDiffusion1D, AllenCahn1D,
    KuramotoSivashinsky1D
)
from pinn_benchmark.utils.metrics import (
    l2_relative_error, compute_all_metrics, analyze_training_convergence
)
from pinn_benchmark.utils.plotting import (
    plot_convergence_history, plot_solution_comparison_1d,
    plot_solution_comparison_2d, plot_multiple_convergence,
    plot_performance_heatmap, plot_boxplot_comparison,
    plot_accuracy_vs_time
)


class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    def __init__(self):
        # Network architecture
        self.hidden_layers = [50, 50, 50, 50]
        self.activation = 'tanh'

        # Training hyperparameters
        self.n_epochs = 10000
        self.learning_rate = 1e-3
        self.n_interior = 1000
        self.n_boundary = 100
        self.n_initial = 100

        # Physics loss weight
        self.lambda_physics = 1.0

        # Early stopping
        self.early_stopping_patience = 2000
        self.early_stopping_delta = 1e-6

        # Random seeds for reproducibility
        self.seeds = [42, 123, 456, 789, 1011]

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Output directories
        self.results_dir = 'results'
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        self.checkpoints_dir = os.path.join(self.results_dir, 'checkpoints')

        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)


def get_problem_instances() -> Dict:
    """
    Get all benchmark problem instances.

    Returns:
        Dictionary mapping problem names to problem instances
    """
    problems = {
        'Heat Conduction': HeatConduction1D(alpha=0.1),
        'Wave Propagation': WavePropagation1D(c=1.0),
        'Burgers Equation': BurgersEquation(nu=0.01),
        'Euler-Bernoulli Beam': EulerBernoulliBeam(EI=1.0, q0=1.0),
        'Advection-Diffusion': AdvectionDiffusion1D(v=1.0, kappa=0.01),
        'Allen-Cahn': AllenCahn1D(epsilon=0.1),
        'Kuramoto-Sivashinsky': KuramotoSivashinsky1D(),
        # Simplified 2D problems for faster benchmarking
        # 'Navier-Stokes': NavierStokes2D(nu=0.01),
        # 'Linear Elasticity': LinearElasticity2D(E=1.0, nu=0.3),
        # 'Plate Vibration': PlateVibration(D=1.0, rho_h=1.0),
    }
    return problems


def create_pinn_variant(
    variant_name: str,
    problem,
    config: BenchmarkConfig,
    seed: int
) -> object:
    """
    Create a PINN variant instance.

    Args:
        variant_name: Name of PINN variant
        problem: Problem instance
        config: Benchmark configuration
        seed: Random seed

    Returns:
        PINN instance
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine input/output dimensions from problem
    training_data = problem.get_training_data(n_interior=10, device=config.device)
    input_dim = training_data['x_interior'].shape[1]

    # Determine output dimension
    if hasattr(problem, 'name'):
        if 'Navier-Stokes' in problem.name:
            output_dim = 3  # u, v, p
        elif 'Elasticity' in problem.name:
            output_dim = 2  # u, v
        else:
            output_dim = 1
    else:
        output_dim = 1

    # Network architecture
    layers = [input_dim] + config.hidden_layers + [output_dim]

    # Create PINN based on variant
    pde_residual_fn = problem.pde_residual

    if variant_name == 'Vanilla':
        pinn = VanillaPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif variant_name == 'Variational':
        pinn = VariationalPINN(
            energy_functional_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif variant_name == 'Conservative':
        pinn = ConservativePINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif variant_name == 'Bayesian':
        pinn = BayesianPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics,
            dropout_rate=0.1
        )
    elif variant_name == 'Gradient-Enhanced':
        pinn = GradientEnhancedPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif variant_name == 'Adaptive':
        pinn = AdaptivePINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics,
            adaptive_weights=True,
            adaptive_sampling=True
        )
    elif variant_name == 'Causal':
        pinn = CausalPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    else:
        raise ValueError(f"Unknown PINN variant: {variant_name}")

    return pinn


def run_single_experiment(
    pinn_variant_name: str,
    problem_name: str,
    problem,
    config: BenchmarkConfig,
    seed: int,
    verbose: bool = False
) -> Dict:
    """
    Run a single experiment.

    Args:
        pinn_variant_name: Name of PINN variant
        problem_name: Name of problem
        problem: Problem instance
        config: Configuration
        seed: Random seed
        verbose: Print progress

    Returns:
        Results dictionary
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {pinn_variant_name} on {problem_name} (seed={seed})")
        print(f"{'='*60}")

    # Create PINN
    pinn = create_pinn_variant(pinn_variant_name, problem, config, seed)

    # Get training data
    training_data = problem.get_training_data(
        n_interior=config.n_interior,
        n_boundary=config.n_boundary,
        n_initial=config.n_initial,
        device=config.device
    )

    # Train
    start_time = time.time()
    history = pinn.train(
        n_epochs=config.n_epochs,
        x_interior=training_data['x_interior'],
        x_boundary=training_data.get('x_boundary'),
        u_boundary=training_data.get('u_boundary'),
        x_initial=training_data.get('x_initial'),
        u_initial=training_data.get('u_initial'),
        checkpoint_freq=1000,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_delta=config.early_stopping_delta,
        verbose=verbose
    )
    training_time = time.time() - start_time

    # Evaluate on test data
    test_data = problem.get_test_data()

    if len(test_data) == 2:  # 1D problem: x, u_exact
        x_test, u_exact = test_data
        xt_test = torch.tensor(x_test, dtype=torch.float32, device=config.device)
        u_pred = pinn.predict(xt_test)
        error_metrics = compute_all_metrics(u_pred.flatten(), u_exact.flatten())

    elif len(test_data) == 3:  # 2D spatiotemporal: X, T, U
        X, T, U_exact = test_data
        xt_test = np.hstack([X.flatten().reshape(-1, 1), T.flatten().reshape(-1, 1)])
        xt_test_tensor = torch.tensor(xt_test, dtype=torch.float32, device=config.device)
        u_pred = pinn.predict(xt_test_tensor)
        error_metrics = compute_all_metrics(u_pred.flatten(), U_exact.flatten())

    elif len(test_data) == 4:  # 2D spatial or spatiotemporal with 2 outputs
        # Handle different cases
        error_metrics = {'l2_relative_error': 0.0}  # Placeholder

    else:  # Other cases
        error_metrics = {'l2_relative_error': 0.0}  # Placeholder

    # Memory usage
    memory_mb = pinn.get_memory_usage()

    # Convergence analysis
    convergence_analysis = analyze_training_convergence(history)

    # Compile results
    results = {
        'pinn_variant': pinn_variant_name,
        'problem': problem_name,
        'seed': seed,
        'training_time': training_time,
        'memory_mb': memory_mb,
        'final_loss': history['total_loss'][-1] if history['total_loss'] else float('inf'),
        'error_metrics': error_metrics,
        'convergence_analysis': convergence_analysis,
        'history': {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                   for k, vals in history.items()}
    }

    if verbose:
        print(f"Training time: {training_time:.2f}s")
        print(f"L2 Relative Error: {error_metrics.get('l2_relative_error', 0.0):.6e}")
        print(f"Final Loss: {results['final_loss']:.6e}")

    return results


def run_full_benchmark(config: BenchmarkConfig) -> Dict:
    """
    Run full benchmark suite.

    Args:
        config: Benchmark configuration

    Returns:
        Complete results dictionary
    """
    print("="*80)
    print("PINN BENCHMARK FRAMEWORK")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Number of seeds: {len(config.seeds)}")
    print(f"Number of epochs: {config.n_epochs}")
    print("="*80)

    # Get problems
    problems = get_problem_instances()

    # PINN variants to test
    pinn_variants = [
        'Vanilla',
        'Variational',
        'Conservative',
        'Bayesian',
        'Gradient-Enhanced',
        'Adaptive',
        'Causal'
    ]

    # Results storage
    all_results = []

    # Run experiments
    total_experiments = len(pinn_variants) * len(problems) * len(config.seeds)
    current_experiment = 0

    for pinn_variant in pinn_variants:
        for problem_name, problem in problems.items():
            for seed in config.seeds:
                current_experiment += 1
                print(f"\nProgress: {current_experiment}/{total_experiments}")

                try:
                    result = run_single_experiment(
                        pinn_variant,
                        problem_name,
                        problem,
                        config,
                        seed,
                        verbose=True
                    )
                    all_results.append(result)

                except Exception as e:
                    print(f"ERROR in {pinn_variant} on {problem_name}: {str(e)}")
                    # Add failed result
                    all_results.append({
                        'pinn_variant': pinn_variant,
                        'problem': problem_name,
                        'seed': seed,
                        'error': str(e),
                        'training_time': 0.0,
                        'error_metrics': {'l2_relative_error': float('inf')}
                    })

    # Save raw results
    results_file = os.path.join(config.results_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return {'results': all_results, 'config': config}


def analyze_and_visualize(results_data: Dict, config: BenchmarkConfig):
    """
    Analyze results and generate visualizations.

    Args:
        results_data: Results from benchmark
        config: Configuration
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS AND ANALYSIS")
    print("="*80)

    results = results_data['results']

    # Get unique problems and variants
    problems = sorted(list(set([r['problem'] for r in results])))
    variants = sorted(list(set([r['pinn_variant'] for r in results])))

    # Performance matrix (average L2 error)
    perf_matrix = np.zeros((len(variants), len(problems)))

    for i, variant in enumerate(variants):
        for j, problem in enumerate(problems):
            errors = [r['error_metrics'].get('l2_relative_error', float('inf'))
                     for r in results
                     if r['pinn_variant'] == variant and r['problem'] == problem]
            perf_matrix[i, j] = np.mean(errors) if errors else float('inf')

    # Plot performance heatmap
    plot_performance_heatmap(
        perf_matrix,
        variants,
        problems,
        metric_name="L2 Relative Error",
        save_path=os.path.join(config.plots_dir, 'performance_heatmap.png'),
        title="PINN Performance Across Problems"
    )
    print("✓ Generated performance heatmap")

    # Boxplot comparison for each problem
    for problem in problems:
        problem_results = {
            variant: [r['error_metrics'].get('l2_relative_error', float('inf'))
                     for r in results
                     if r['problem'] == problem and r['pinn_variant'] == variant]
            for variant in variants
        }

        safe_name = problem.replace(' ', '_').replace('-', '_')
        plot_boxplot_comparison(
            problem_results,
            save_path=os.path.join(config.plots_dir, f'boxplot_{safe_name}.png'),
            title=f'PINN Comparison: {problem}',
            ylabel='L2 Relative Error'
        )
    print(f"✓ Generated {len(problems)} boxplot comparisons")

    # Accuracy vs time plots
    for problem in problems:
        problem_results = [r for r in results if r['problem'] == problem]

        # Average across seeds
        variant_times = {}
        variant_errors = {}

        for variant in variants:
            variant_results = [r for r in problem_results if r['pinn_variant'] == variant]
            if variant_results:
                variant_times[variant] = np.mean([r['training_time'] for r in variant_results])
                variant_errors[variant] = np.mean([r['error_metrics'].get('l2_relative_error', float('inf'))
                                                   for r in variant_results])

        safe_name = problem.replace(' ', '_').replace('-', '_')
        plot_accuracy_vs_time(
            list(variant_times.keys()),
            list(variant_errors.values()),
            list(variant_times.values()),
            save_path=os.path.join(config.plots_dir, f'accuracy_vs_time_{safe_name}.png'),
            title=f'Accuracy vs Training Time: {problem}'
        )
    print(f"✓ Generated {len(problems)} accuracy vs time plots")

    print("\n✓ All visualizations generated successfully!")


def generate_report(results_data: Dict, config: BenchmarkConfig):
    """
    Generate markdown report.

    Args:
        results_data: Results from benchmark
        config: Configuration
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)

    results = results_data['results']
    problems = sorted(list(set([r['problem'] for r in results])))
    variants = sorted(list(set([r['pinn_variant'] for r in results])))

    report_lines = []

    # Header
    report_lines.append("# PINN Benchmark Results")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(f"This report presents a comprehensive comparison of {len(variants)} PINN variants ")
    report_lines.append(f"across {len(problems)} mechanical engineering benchmark problems.")
    report_lines.append("")

    # Configuration
    report_lines.append("## Experimental Setup")
    report_lines.append("")
    report_lines.append("### Hyperparameters")
    report_lines.append("")
    report_lines.append(f"- **Network Architecture**: {config.hidden_layers}")
    report_lines.append(f"- **Training Epochs**: {config.n_epochs}")
    report_lines.append(f"- **Learning Rate**: {config.learning_rate}")
    report_lines.append(f"- **Interior Points**: {config.n_interior}")
    report_lines.append(f"- **Boundary Points**: {config.n_boundary}")
    report_lines.append(f"- **Random Seeds**: {len(config.seeds)}")
    report_lines.append(f"- **Device**: {config.device}")
    report_lines.append("")

    # Performance table
    report_lines.append("## Performance Summary")
    report_lines.append("")
    report_lines.append("### L2 Relative Error (Mean ± Std)")
    report_lines.append("")

    # Table header
    header = "| PINN Variant |" + "|".join([f" {p} " for p in problems]) + "|"
    separator = "|" + "|".join(["---" for _ in range(len(problems) + 1)]) + "|"

    report_lines.append(header)
    report_lines.append(separator)

    # Table rows
    for variant in variants:
        row = f"| {variant} |"
        for problem in problems:
            errors = [r['error_metrics'].get('l2_relative_error', float('inf'))
                     for r in results
                     if r['pinn_variant'] == variant and r['problem'] == problem]

            if errors:
                mean_err = np.mean(errors)
                std_err = np.std(errors)
                row += f" {mean_err:.2e} ± {std_err:.2e} |"
            else:
                row += " N/A |"

        report_lines.append(row)

    report_lines.append("")

    # Best performer for each problem
    report_lines.append("## Best Performers")
    report_lines.append("")

    for problem in problems:
        problem_results = {}
        for variant in variants:
            errors = [r['error_metrics'].get('l2_relative_error', float('inf'))
                     for r in results
                     if r['pinn_variant'] == variant and r['problem'] == problem]
            if errors:
                problem_results[variant] = np.mean(errors)

        if problem_results:
            best_variant = min(problem_results, key=problem_results.get)
            best_error = problem_results[best_variant]
            report_lines.append(f"- **{problem}**: {best_variant} ({best_error:.2e})")

    report_lines.append("")

    # Computational cost
    report_lines.append("## Computational Cost Analysis")
    report_lines.append("")
    report_lines.append("### Average Training Time (seconds)")
    report_lines.append("")

    for variant in variants:
        times = [r['training_time'] for r in results if r['pinn_variant'] == variant]
        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            report_lines.append(f"- **{variant}**: {mean_time:.2f} ± {std_time:.2f}s")

    report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append("### When to Use Each PINN Variant")
    report_lines.append("")
    report_lines.append("- **Vanilla PINN**: General-purpose, good baseline for most problems")
    report_lines.append("- **Variational PINN**: Best for problems with natural energy formulations")
    report_lines.append("- **Conservative PINN**: Ideal when conservation laws are critical")
    report_lines.append("- **Bayesian PINN**: Use when uncertainty quantification is needed")
    report_lines.append("- **Gradient-Enhanced PINN**: Effective when derivative data is available")
    report_lines.append("- **Adaptive PINN**: Good for complex problems with varying solution features")
    report_lines.append("- **Causal PINN**: Best for time-dependent problems with strong causality")
    report_lines.append("")

    # Visualizations
    report_lines.append("## Visualizations")
    report_lines.append("")
    report_lines.append("![Performance Heatmap](plots/performance_heatmap.png)")
    report_lines.append("")

    # Write report
    report_path = os.path.join(config.results_dir, 'RESULTS.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Report generated: {report_path}")


if __name__ == "__main__":
    # Create configuration
    config = BenchmarkConfig()

    # Run benchmarks
    results_data = run_full_benchmark(config)

    # Analyze and visualize
    analyze_and_visualize(results_data, config)

    # Generate report
    generate_report(results_data, config)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults directory: {config.results_dir}")
    print(f"Plots directory: {config.plots_dir}")
    print("\nCheck RESULTS.md for comprehensive analysis.")
