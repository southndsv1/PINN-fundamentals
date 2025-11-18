# PINN Benchmark Framework - Quick Start Guide

## 🚀 Installation (30 seconds)

```bash
cd PINN-fundamentals
pip install torch numpy matplotlib
```

## ⚡ Quick Test (2 minutes)

```bash
python demo_quick_test.py
```

This runs 2 problems to verify everything works.

## 🎯 Full Benchmark (6-10 hours)

```bash
python run_benchmarks.py
```

This runs all 7 PINN variants on all 10 problems with 5 random seeds.

## 📊 Check Results

```bash
# View comprehensive report
cat results/RESULTS.md

# View plots
ls results/plots/

# View raw data
cat results/benchmark_results.json
```

## 🔧 Custom Configuration

Edit `run_benchmarks.py`:

```python
class BenchmarkConfig:
    # Change network size
    hidden_layers = [50, 50, 50, 50]  # or [100, 100, 100]

    # Change training duration
    n_epochs = 10000  # or 5000 for faster

    # Change collocation points
    n_interior = 1000  # or 2000 for better accuracy

    # Change random seeds
    seeds = [42, 123, 456, 789, 1011]  # or just [42] for quick test
```

## 📚 Single Problem Example

```python
from pinn_benchmark.pinn_variants import VanillaPINN
from pinn_benchmark.benchmark_problems import HeatConduction1D
import torch

# Setup
problem = HeatConduction1D(alpha=0.1)
data = problem.get_training_data(n_interior=1000, device='cpu')

# Create PINN
pinn = VanillaPINN(
    pde_residual_fn=problem.pde_residual,
    layers=[2, 50, 50, 50, 50, 1],
    learning_rate=1e-3,
    device='cpu'
)

# Train
history = pinn.train(
    n_epochs=5000,
    x_interior=data['x_interior'],
    x_boundary=data['x_boundary'],
    u_boundary=data['u_boundary'],
    x_initial=data['x_initial'],
    u_initial=data['u_initial']
)

# Evaluate
X, T, U_exact = problem.get_test_data()
# ... prediction and error computation
```

## 🎨 Available PINN Variants

1. `VanillaPINN` - Standard (fast, good baseline)
2. `VariationalPINN` - Weak form (energy problems)
3. `ConservativePINN` - Conservation laws (physics-critical)
4. `BayesianPINN` - Uncertainty (with confidence intervals)
5. `GradientEnhancedPINN` - Derivative matching (high accuracy)
6. `AdaptivePINN` - Dynamic adjustment (complex problems)
7. `CausalPINN` - Time-sequential (temporal causality)

## 🧪 Available Problems

1. Heat Conduction (1D parabolic)
2. Wave Propagation (1D hyperbolic)
3. Burgers Equation (nonlinear)
4. Euler-Bernoulli Beam (4th order)
5. Plate Vibration (2D structural)
6. Linear Elasticity (2D system)
7. Advection-Diffusion (transport)
8. Allen-Cahn (phase field)
9. Kuramoto-Sivashinsky (chaotic)
10. Navier-Stokes (2D fluid)

## 📈 Expected Results

| Metric | Typical Value |
|--------|---------------|
| Training Time | 1-2 min/problem |
| L2 Error | 1e-3 to 1e-2 |
| Convergence | 2000-5000 epochs |
| Memory | < 100 MB |

## ⚠️ Common Issues

**Issue**: Out of memory
**Solution**: Reduce `n_interior` or `hidden_layers`

**Issue**: Training too slow
**Solution**: Reduce `n_epochs` or use GPU

**Issue**: Poor accuracy
**Solution**: Increase `n_interior` or `n_epochs`

**Issue**: Not converging
**Solution**: Try different `learning_rate` or PINN variant

## 🎓 Understanding Results

### Convergence Plot
- **Decreasing**: Good, model is learning
- **Flat**: Converged or stuck
- **Increasing**: Learning rate too high

### L2 Error
- **< 1e-2**: Excellent
- **1e-2 to 1e-1**: Good
- **> 1e-1**: Poor, needs tuning

### Training Time
- Vanilla: Fastest
- Bayesian: Slowest (uncertainty estimation)
- Adaptive: Medium (dynamic sampling)

## 🔍 Tips

1. **Start small**: Use `demo_quick_test.py` first
2. **GPU helps**: Set `device='cuda'` if available
3. **Multiple seeds**: Important for statistical significance
4. **Early stopping**: Saves time, prevents overfitting
5. **Check plots**: Visual inspection reveals issues

## 📖 Full Documentation

See `README_BENCHMARK.md` for complete documentation.

---

**Need help?** Check the implementation summary in `IMPLEMENTATION_SUMMARY.md`
