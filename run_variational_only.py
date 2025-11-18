"""
Run only Variational PINN experiments to complete the benchmark.
"""

import os
import sys
import json
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pinn_benchmark.pinn_variants import VariationalPINN
from pinn_benchmark.benchmark_problems import (
    HeatConduction1D, WavePropagation1D, BurgersEquation
)
from pinn_benchmark.utils.metrics import l2_relative_error
from pinn_benchmark.utils.plotting import plot_convergence_history, plot_solution_comparison_1d

# Configuration
class Config:
    def __init__(self):
        self.seed = 42
        self.device = 'cpu'
        self.n_epochs = 2000
        self.hidden_layers = [32, 32, 32]
        self.learning_rate = 1e-3
        self.lambda_physics = 1.0
        self.n_interior = 500
        self.n_boundary = 50
        self.n_initial = 50
        self.early_stopping_patience = 500

def run_experiment(problem_name, problem, config):
    """Run single Variational PINN experiment."""
    print(f"\n{'='*70}")
    print(f"Running: Variational on {problem_name}")
    print(f"{'='*70}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Get training data
    training_data = problem.get_training_data(
        n_interior=config.n_interior,
        n_boundary=config.n_boundary,
        n_initial=config.n_initial,
        device=config.device
    )

    input_dim = training_data['x_interior'].shape[1]
    output_dim = 1
    layers = [input_dim] + config.hidden_layers + [output_dim]

    # Create Variational PINN
    pinn = VariationalPINN(
        layers=layers,
        learning_rate=config.learning_rate,
        device=config.device,
        lambda_physics=config.lambda_physics
    )

    print(f"Network architecture: {layers}")
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
    print(f"\nEvaluating on test data...")
    test_data = problem.get_test_data()

    if len(test_data) == 3:
        X, T, U_exact = test_data
        xt_test = np.hstack([X.flatten().reshape(-1, 1), T.flatten().reshape(-1, 1)])
        xt_test_tensor = torch.tensor(xt_test, dtype=torch.float32, device=config.device)

        u_pred = pinn.predict(xt_test_tensor)
        error = l2_relative_error(U_exact.flatten(), u_pred.flatten())

        print(f"  L2 Relative Error: {error:.6e}")

        # Save plots
        os.makedirs('fast_results/plots', exist_ok=True)
        safe_name = problem_name.replace(" ", "_")
        plot_solution_comparison_1d(
            X, T, U_exact, u_pred.reshape(X.shape),
            save_path=f'fast_results/plots/solution_{safe_name}_Variational.png',
            title=f'Variational PINN: {problem_name}'
        )
        plot_convergence_history(
            history,
            save_path=f'fast_results/plots/convergence_{safe_name}_Variational.png',
            title=f'{problem_name} - Variational'
        )
        print(f"  ✓ Plots saved")
    else:
        error = None

    return {
        'pinn_variant': 'Variational',
        'problem': problem_name,
        'training_time': training_time,
        'final_loss': history['total_loss'][-1],
        'epochs': len(history['epochs']),
        'l2_error': error,
        'history': history
    }

def main():
    print("="*70)
    print("VARIATIONAL PINN EXPERIMENTS")
    print("="*70)

    config = Config()

    problems = {
        'Heat Conduction': HeatConduction1D(alpha=0.1),
        'Wave Propagation': WavePropagation1D(c=1.0),
        'Burgers Equation': BurgersEquation(nu=0.01)
    }

    results = []
    for i, (problem_name, problem) in enumerate(problems.items(), 1):
        print(f"\nProgress: {i}/{len(problems)}")
        try:
            result = run_experiment(problem_name, problem, config)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    for result in results:
        print(f"{result['problem']}: L2 Error = {result['l2_error']:.6e}, Time = {result['training_time']:.2f}s")

    print(f"\n✓ All Variational PINN experiments complete!")

if __name__ == '__main__':
    main()
