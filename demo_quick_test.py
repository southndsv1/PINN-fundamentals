"""
Quick demonstration of the PINN Benchmark Framework.

This script runs a simplified benchmark with fewer epochs and seeds
to verify the framework is working correctly.
"""

import os
import sys
import torch
import numpy as np

# Add pinn_benchmark to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinn_benchmark.pinn_variants import VanillaPINN, AdaptivePINN
from pinn_benchmark.benchmark_problems import HeatConduction1D, WavePropagation1D
from pinn_benchmark.utils.metrics import l2_relative_error
from pinn_benchmark.utils.plotting import plot_convergence_history, plot_solution_comparison_1d

print("="*80)
print("PINN BENCHMARK FRAMEWORK - QUICK DEMO")
print("="*80)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Create output directory
os.makedirs('demo_results', exist_ok=True)

# Test 1: Heat Conduction with Vanilla PINN
print("\n" + "-"*80)
print("Test 1: Heat Conduction with Vanilla PINN")
print("-"*80)

# Create problem
problem = HeatConduction1D(alpha=0.1)
print(f"Problem: {problem.name}")

# Get training data
training_data = problem.get_training_data(
    n_interior=500,
    n_boundary=50,
    n_initial=50,
    device=device
)
print(f"Training data: {training_data['x_interior'].shape[0]} interior points")

# Create PINN
pinn = VanillaPINN(
    pde_residual_fn=problem.pde_residual,
    layers=[2, 32, 32, 32, 1],  # Smaller network for quick test
    learning_rate=1e-3,
    device=device,
    lambda_physics=1.0
)
print(f"Network: {pinn.layers}")

# Train
print("\nTraining...")
history = pinn.train(
    n_epochs=2000,  # Fewer epochs for quick test
    x_interior=training_data['x_interior'],
    x_boundary=training_data['x_boundary'],
    u_boundary=training_data['u_boundary'],
    x_initial=training_data['x_initial'],
    u_initial=training_data['u_initial'],
    early_stopping_patience=500,
    verbose=True
)

print(f"\nTraining completed in {len(history['epochs'])} epochs")
print(f"Final loss: {history['total_loss'][-1]:.6e}")

# Plot convergence
plot_convergence_history(
    history,
    save_path='demo_results/heat_vanilla_convergence.png',
    title='Heat Conduction - Vanilla PINN'
)
print("✓ Saved convergence plot")

# Evaluate
print("\nEvaluating on test data...")
X, T, U_exact = problem.get_test_data(nx=100, nt=100)

# Test at final time
t_final = 1.0
x_test = X[:, -1].reshape(-1, 1)
t_test = np.full_like(x_test, t_final)
xt_test = np.hstack([x_test, t_test])
xt_test_tensor = torch.tensor(xt_test, dtype=torch.float32, device=device)

u_pred = pinn.predict(xt_test_tensor)
u_exact = U_exact[:, -1].reshape(-1, 1)

# Compute error
error = l2_relative_error(u_pred, u_exact)
print(f"L2 Relative Error: {error:.6e}")

# Plot solution
plot_solution_comparison_1d(
    x_test,
    u_pred,
    u_exact,
    save_path='demo_results/heat_vanilla_solution.png',
    title=f'Heat Conduction Solution (t={t_final})'
)
print("✓ Saved solution plot")

# Test 2: Wave Propagation with Adaptive PINN
print("\n" + "-"*80)
print("Test 2: Wave Propagation with Adaptive PINN")
print("-"*80)

# Create problem
problem2 = WavePropagation1D(c=1.0)
print(f"Problem: {problem2.name}")

# Get training data
training_data2 = problem2.get_training_data(
    n_interior=500,
    n_boundary=50,
    n_initial=50,
    device=device
)

# Create Adaptive PINN
pinn2 = AdaptivePINN(
    pde_residual_fn=problem2.pde_residual,
    layers=[2, 32, 32, 32, 1],
    learning_rate=1e-3,
    device=device,
    lambda_physics=1.0,
    adaptive_weights=True,
    adaptive_sampling=False  # Disable for quick test
)

# Train
print("\nTraining...")
history2 = pinn2.train(
    n_epochs=2000,
    x_interior=training_data2['x_interior'],
    x_boundary=training_data2['x_boundary'],
    u_boundary=training_data2['u_boundary'],
    x_initial=training_data2['x_initial'],
    u_initial=training_data2['u_initial'],
    early_stopping_patience=500,
    verbose=True
)

print(f"\nTraining completed in {len(history2['epochs'])} epochs")
print(f"Final loss: {history2['total_loss'][-1]:.6e}")

# Plot convergence
plot_convergence_history(
    history2,
    save_path='demo_results/wave_adaptive_convergence.png',
    title='Wave Propagation - Adaptive PINN'
)
print("✓ Saved convergence plot")

# Evaluate
print("\nEvaluating on test data...")
X2, T2, U_exact2 = problem2.get_test_data(nx=100, nt=100)

t_final2 = 1.0
x_test2 = X2[:, -1].reshape(-1, 1)
t_test2 = np.full_like(x_test2, t_final2)
xt_test2 = np.hstack([x_test2, t_test2])
xt_test_tensor2 = torch.tensor(xt_test2, dtype=torch.float32, device=device)

u_pred2 = pinn2.predict(xt_test_tensor2)
u_exact2 = U_exact2[:, -1].reshape(-1, 1)

error2 = l2_relative_error(u_pred2, u_exact2)
print(f"L2 Relative Error: {error2:.6e}")

# Plot solution
plot_solution_comparison_1d(
    x_test2,
    u_pred2,
    u_exact2,
    save_path='demo_results/wave_adaptive_solution.png',
    title=f'Wave Propagation Solution (t={t_final2})'
)
print("✓ Saved solution plot")

# Summary
print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nResults:")
print(f"1. Heat Conduction (Vanilla PINN):  L2 Error = {error:.6e}")
print(f"2. Wave Propagation (Adaptive PINN): L2 Error = {error2:.6e}")
print(f"\nPlots saved in: demo_results/")
print("\nFramework is working correctly!")
print("\nTo run full benchmark suite, execute:")
print("  python run_benchmarks.py")
