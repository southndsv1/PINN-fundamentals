"""
Fast benchmark runner for quick demonstration.
Runs a subset of experiments with reduced epochs for faster execution.
"""

import os
import sys
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add pinn_benchmark to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Importing modules...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported")
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch numpy matplotlib")
    sys.exit(1)

from pinn_benchmark.pinn_variants import (
    VanillaPINN,
    VariationalPINN,
    ConservativePINN,
    BayesianPINN,
    GradientEnhancedPINN,
    AdaptivePINN,
    CausalPINN
)
from pinn_benchmark.benchmark_problems import (
    HeatConduction1D, WavePropagation1D, BurgersEquation
)
from pinn_benchmark.utils.metrics import l2_relative_error, compute_all_metrics
from pinn_benchmark.utils.plotting import (
    plot_convergence_history, plot_solution_comparison_1d,
    plot_multiple_convergence
)

print("✓ All modules imported successfully\n")

# Fast configuration
class FastConfig:
    """Fast configuration for quick benchmarking."""
    hidden_layers = [32, 32, 32]  # Smaller network
    n_epochs = 2000  # Fewer epochs
    learning_rate = 1e-3
    n_interior = 500  # Fewer collocation points
    n_boundary = 50
    n_initial = 50
    lambda_physics = 1.0
    early_stopping_patience = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = 'fast_results'
    plots_dir = os.path.join(results_dir, 'plots')
    seed = 42

    def __init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)


def run_experiment(pinn_variant_name, problem_name, problem, config):
    """Run single experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {pinn_variant_name} on {problem_name}")
    print(f"{'='*70}")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Get training data
    training_data = problem.get_training_data(
        n_interior=config.n_interior,
        n_boundary=config.n_boundary,
        n_initial=config.n_initial,
        device=config.device
    )

    # Determine input/output dimensions
    input_dim = training_data['x_interior'].shape[1]
    output_dim = 1
    layers = [input_dim] + config.hidden_layers + [output_dim]

    # Create PINN
    pde_residual_fn = problem.pde_residual

    if pinn_variant_name == 'Vanilla':
        pinn = VanillaPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif pinn_variant_name == 'Adaptive':
        pinn = AdaptivePINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics,
            adaptive_weights=True,
            adaptive_sampling=False  # Disable for speed
        )
    elif pinn_variant_name == 'Causal':
        pinn = CausalPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif pinn_variant_name == 'Variational':
        pinn = VariationalPINN(
            # energy_functional_fn is optional, uses default diffusion energy
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif pinn_variant_name == 'Conservative':
        pinn = ConservativePINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    elif pinn_variant_name == 'Bayesian':
        pinn = BayesianPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics,
            n_samples=5  # Reduced for speed
        )
    elif pinn_variant_name == 'Gradient-Enhanced':
        pinn = GradientEnhancedPINN(
            pde_residual_fn=pde_residual_fn,
            layers=layers,
            learning_rate=config.learning_rate,
            device=config.device,
            lambda_physics=config.lambda_physics
        )
    else:
        raise ValueError(f"Unknown variant: {pinn_variant_name}")

    print(f"Network architecture: {layers}")
    print(f"Device: {config.device}")
    print(f"Training for {config.n_epochs} epochs...\n")

    # Train
    start_time = time.time()
    history = pinn.train(
        n_epochs=config.n_epochs,
        x_interior=training_data['x_interior'],
        x_boundary=training_data.get('x_boundary'),
        u_boundary=training_data.get('u_boundary'),
        x_initial=training_data.get('x_initial'),
        u_initial=training_data.get('u_initial'),
        early_stopping_patience=config.early_stopping_patience,
        verbose=True
    )
    training_time = time.time() - start_time

    print(f"\n✓ Training completed in {training_time:.2f}s")
    print(f"  Final loss: {history['total_loss'][-1]:.6e}")
    print(f"  Epochs: {len(history['epochs'])}")

    # Evaluate
    print("\nEvaluating on test data...")
    test_data = problem.get_test_data(nx=100, nt=100)

    if len(test_data) == 3:  # X, T, U_exact
        X, T, U_exact = test_data
        # Test at final time
        t_final = T.max()
        x_test = X[:, -1].reshape(-1, 1)
        t_test = np.full_like(x_test, t_final)
        xt_test = np.hstack([x_test, t_test])
        xt_test_tensor = torch.tensor(xt_test, dtype=torch.float32, device=config.device)

        u_pred = pinn.predict(xt_test_tensor)
        u_exact = U_exact[:, -1].reshape(-1, 1)

        error = l2_relative_error(u_pred, u_exact)
        print(f"  L2 Relative Error: {error:.6e}")

        # Plot solution
        safe_problem = problem_name.replace(' ', '_')
        safe_variant = pinn_variant_name.replace(' ', '_')

        plot_solution_comparison_1d(
            x_test,
            u_pred,
            u_exact,
            save_path=os.path.join(config.plots_dir, f'solution_{safe_problem}_{safe_variant}.png'),
            title=f'{problem_name} - {pinn_variant_name} (t={t_final:.2f})'
        )
        print(f"  ✓ Solution plot saved")

    # Plot convergence
    safe_problem = problem_name.replace(' ', '_')
    safe_variant = pinn_variant_name.replace(' ', '_')
    plot_convergence_history(
        history,
        save_path=os.path.join(config.plots_dir, f'convergence_{safe_problem}_{safe_variant}.png'),
        title=f'{problem_name} - {pinn_variant_name}'
    )
    print(f"  ✓ Convergence plot saved")

    return {
        'pinn_variant': pinn_variant_name,
        'problem': problem_name,
        'training_time': training_time,
        'final_loss': history['total_loss'][-1],
        'epochs': len(history['epochs']),
        'l2_error': error if len(test_data) == 3 else None,
        'history': history
    }


def main():
    """Main benchmark execution."""
    print("="*70)
    print("FAST PINN BENCHMARK")
    print("="*70)
    print()

    config = FastConfig()
    print(f"Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Network: {config.hidden_layers}")
    print(f"  Collocation points: {config.n_interior}")
    print()

    # Select problems
    problems = {
        'Heat Conduction': HeatConduction1D(alpha=0.1),
        'Wave Propagation': WavePropagation1D(c=1.0),
        'Burgers Equation': BurgersEquation(nu=0.01)
    }

    # Select PINN variants - ALL 7 VARIANTS
    pinn_variants = [
        'Vanilla',
        'Variational',
        'Conservative',
        'Bayesian',
        'Gradient-Enhanced',
        'Adaptive',
        'Causal'
    ]

    # Run experiments
    all_results = []
    total_experiments = len(pinn_variants) * len(problems)
    current = 0

    for pinn_variant in pinn_variants:
        for problem_name, problem in problems.items():
            current += 1
            print(f"\nProgress: {current}/{total_experiments}")

            try:
                result = run_experiment(pinn_variant, problem_name, problem, config)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    # Generate comparison plots
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    for problem_name in problems.keys():
        # Get histories for this problem
        histories = {}
        for result in all_results:
            if result['problem'] == problem_name:
                histories[result['pinn_variant']] = result['history']

        if histories:
            safe_problem = problem_name.replace(' ', '_')
            plot_multiple_convergence(
                histories,
                save_path=os.path.join(config.plots_dir, f'comparison_{safe_problem}.png'),
                title=f'Convergence Comparison: {problem_name}'
            )
            print(f"  ✓ {problem_name} comparison plot saved")

    # Save results
    results_file = os.path.join(config.results_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        # Convert history arrays to lists for JSON serialization
        results_json = []
        for r in all_results:
            r_copy = r.copy()
            r_copy['history'] = {
                k: [float(v) for v in vals] if isinstance(vals, (list, np.ndarray)) else vals
                for k, vals in r['history'].items()
            }
            results_json.append(r_copy)
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()

    for problem_name in problems.keys():
        print(f"\n{problem_name}:")
        print("-" * 70)
        problem_results = [r for r in all_results if r['problem'] == problem_name]
        for result in sorted(problem_results, key=lambda x: x.get('l2_error', float('inf'))):
            variant = result['pinn_variant']
            time_sec = result['training_time']
            loss = result['final_loss']
            error = result.get('l2_error')

            if error is not None:
                print(f"  {variant:15s} | L2 Error: {error:.6e} | Time: {time_sec:6.2f}s | Loss: {loss:.6e}")
            else:
                print(f"  {variant:15s} | Time: {time_sec:6.2f}s | Loss: {loss:.6e}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\nResults directory: {config.results_dir}")
    print(f"Plots directory: {config.plots_dir}")
    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Total time: {sum(r['training_time'] for r in all_results):.2f}s")


if __name__ == "__main__":
    main()
