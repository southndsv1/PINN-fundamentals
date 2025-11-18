## PINN Benchmark Framework

A comprehensive framework for benchmarking different Physics-Informed Neural Network (PINN) variants on mechanical engineering problems.

## Overview

This framework implements and compares **7 PINN variants** on **10 classic mechanical engineering benchmark problems**, providing detailed performance analysis, convergence studies, and computational cost comparisons.

## PINN Variants

1. **Vanilla PINN**: Standard implementation with MSE data loss + PDE residual loss
2. **Variational PINN**: Uses weak form with integration by parts, minimizing energy functional
3. **Conservative PINN**: Explicitly enforces conservation laws (mass, momentum, energy)
4. **Bayesian PINN**: Adds dropout layers for uncertainty quantification with confidence intervals
5. **Gradient-Enhanced PINN**: Includes derivative matching in loss function
6. **Adaptive PINN**: Dynamic weight adjustment and adaptive collocation point sampling
7. **Causal PINN**: Respects temporal causality for time-dependent problems

## Benchmark Problems

### 1. Heat Conduction (1D)
- **PDE**: ∂u/∂t = α∂²u/∂x²
- **Initial condition**: u(x,0) = sin(πx)
- **Boundary conditions**: u(0,t) = u(1,t) = 0
- **Analytical solution**: u(x,t) = sin(πx) × exp(-π²αt)

### 2. Wave Propagation (1D)
- **PDE**: ∂²u/∂t² = c²∂²u/∂x²
- **Initial displacement**: u(x,0) = sin(πx)
- **Initial velocity**: ∂u/∂t(x,0) = 0
- **Boundary conditions**: u(0,t) = u(1,t) = 0

### 3. Burgers' Equation
- **PDE**: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
- **Nonlinear advection-diffusion with shock wave formation**

### 4. Euler-Bernoulli Beam
- **PDE**: d⁴w/dx⁴ = q(x)/EI
- **Beam deflection under distributed load**
- **Boundary conditions**: Simply supported

### 5. Plate Vibration (Kirchhoff)
- **PDE**: ∇⁴w + (ρh/D)∂²w/∂t² = 0
- **2D structural dynamics**

### 6. Linear Elasticity (2D)
- **Stress-strain relationships**
- **Displacement fields under loading**

### 7. Advection-Diffusion
- **PDE**: ∂u/∂t + v·∇u = κ∇²u
- **Transport phenomena**

### 8. Allen-Cahn Equation
- **PDE**: ∂u/∂t = ε²∇²u + u - u³
- **Phase field modeling**

### 9. Kuramoto-Sivashinsky
- **PDE**: ∂u/∂t + ∇⁴u + ∇²u + ½(∇u)² = 0
- **Chaotic dynamics**

### 10. Navier-Stokes (2D)
- **Steady flow equations**
- **Velocity and pressure fields**

## Installation

```bash
# Clone repository
cd PINN-fundamentals

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Benchmark Suite

```bash
python run_benchmarks.py
```

This will:
1. Run each PINN variant on each problem with 5 different random seeds
2. Track convergence history (data loss, physics loss)
3. Measure wall-clock training time and memory usage
4. Compute L2 relative error against analytical solutions
5. Save all results in `results/benchmark_results.json`
6. Generate comprehensive visualizations in `results/plots/`
7. Create detailed report in `results/RESULTS.md`

### Run Single Experiment

```python
from pinn_benchmark.pinn_variants import VanillaPINN
from pinn_benchmark.benchmark_problems import HeatConduction1D

# Create problem
problem = HeatConduction1D(alpha=0.1)

# Get training data
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
    n_epochs=10000,
    x_interior=data['x_interior'],
    x_boundary=data['x_boundary'],
    u_boundary=data['u_boundary'],
    x_initial=data['x_initial'],
    u_initial=data['u_initial']
)

# Evaluate
X, T, U_exact = problem.get_test_data()
u_pred = pinn.predict(...)
```

## Configuration

Edit `BenchmarkConfig` in `run_benchmarks.py`:

```python
class BenchmarkConfig:
    # Network architecture
    hidden_layers = [50, 50, 50, 50]  # 4 hidden layers, 50 neurons each

    # Training
    n_epochs = 10000
    learning_rate = 1e-3

    # Collocation points
    n_interior = 1000
    n_boundary = 100
    n_initial = 100

    # Physics loss weight
    lambda_physics = 1.0

    # Random seeds
    seeds = [42, 123, 456, 789, 1011]
```

## Outputs

### 1. Benchmark Results (`results/benchmark_results.json`)
Complete raw data for all experiments including:
- Training time
- Memory usage
- Loss history
- Error metrics
- Convergence analysis

### 2. Visualizations (`results/plots/`)
- **Performance Heatmap**: Shows which PINN works best for which problem
- **Convergence Plots**: Loss vs epochs for each variant/problem combination
- **Boxplots**: Performance comparison across variants for each problem
- **Accuracy vs Time**: Computational efficiency analysis
- **Error Heatmaps**: Spatial distribution of prediction errors

### 3. Comprehensive Report (`results/RESULTS.md`)
Markdown report containing:
- Executive summary
- Performance comparison tables
- Best performers for each problem
- Computational cost analysis
- Recommendations for which PINN to use when
- Insights about why certain approaches work better

## Key Features

### 1. Modular Design
- Clean separation of PINN variants, problems, and utilities
- Easy to add new PINN variants or benchmark problems
- Extensible base classes

### 2. Comprehensive Metrics
- L2 relative error
- Maximum absolute error
- Mean absolute error
- RMSE
- R² score
- Convergence rate analysis
- Boundary condition satisfaction checks

### 3. Advanced Features
- **Latin Hypercube Sampling** for collocation points
- **Automatic differentiation** using PyTorch autograd
- **Learning rate scheduling** with ReduceLROnPlateau
- **Early stopping** based on validation loss
- **Checkpointing** every 1000 epochs
- **GPU acceleration** (automatic if available)

### 4. Reproducibility
- Multiple random seeds for statistical significance
- Deterministic seeding
- Complete configuration logging

## Benchmark Results Summary

Expected performance characteristics:

| PINN Variant | Best For | Typical Accuracy | Training Time |
|--------------|----------|------------------|---------------|
| Vanilla | General problems | Good | Fast |
| Variational | Energy-based systems | Excellent | Medium |
| Conservative | Conservation-critical | Excellent | Medium |
| Bayesian | Uncertainty needed | Good | Slow |
| Gradient-Enhanced | With derivative data | Excellent | Fast |
| Adaptive | Complex geometries | Very Good | Medium |
| Causal | Time-dependent | Very Good | Medium |

## Architecture Details

```
pinn_benchmark/
├── pinn_variants/
│   ├── base_pinn.py           # Base class with common functionality
│   ├── vanilla_pinn.py         # Standard PINN
│   ├── variational_pinn.py     # Weak form
│   ├── conservative_pinn.py    # Conservation laws
│   ├── bayesian_pinn.py        # Uncertainty quantification
│   ├── gradient_enhanced_pinn.py  # Derivative matching
│   ├── adaptive_pinn.py        # Adaptive weights/sampling
│   └── causal_pinn.py          # Temporal causality
├── benchmark_problems/
│   ├── heat_conduction.py      # 1D heat equation
│   ├── wave_propagation.py     # 1D wave equation
│   ├── burgers_equation.py     # Nonlinear Burgers
│   ├── navier_stokes.py        # 2D fluid dynamics
│   ├── euler_bernoulli_beam.py # Beam deflection
│   ├── plate_vibration.py      # Plate dynamics
│   ├── elasticity.py           # 2D elasticity
│   ├── advection_diffusion.py  # Transport
│   ├── allen_cahn.py           # Phase field
│   └── kuramoto_sivashinsky.py # Chaotic dynamics
└── utils/
    ├── metrics.py              # Performance metrics
    └── plotting.py             # Visualization utilities
```

## Implementation Details

### Network Architecture
- **Layers**: 4 hidden layers × 50 neurons
- **Activation**: tanh
- **Initialization**: Xavier/Glorot normal
- **Optimizer**: Adam
- **Learning rate**: 1e-3 with ReduceLROnPlateau scheduling

### Training Strategy
- **Collocation points**: 1000 interior, 100 boundary
- **Epochs**: 10,000 (with early stopping)
- **Early stopping**: Patience=2000, delta=1e-6
- **Physics weight**: λ = 1.0 (tunable)

### Validation
- Each problem has analytical or high-fidelity numerical solution
- Proper scaling and normalization
- Boundary condition satisfaction checks
- Conservation property verification

## Adding New Components

### Add New PINN Variant

```python
from pinn_benchmark.pinn_variants import BasePINN

class MyCustomPINN(BasePINN):
    def compute_pde_residual(self, x):
        # Implement PDE residual computation
        pass

    def compute_loss(self, x_interior, **kwargs):
        # Implement custom loss function
        pass
```

### Add New Benchmark Problem

```python
class MyProblem:
    def __init__(self, param1=1.0):
        self.param1 = param1
        self.name = "My Problem"

    def analytical_solution(self, x, t):
        # Return exact solution
        pass

    def pde_residual(self, network, xt):
        # Return PDE residual
        pass

    def get_training_data(self, n_interior, device='cpu'):
        # Return training data dict
        pass

    def get_test_data(self):
        # Return test data
        pass
```

## Performance Tips

1. **GPU Acceleration**: Framework automatically uses CUDA if available
2. **Batch Size**: Adjust `n_interior` based on memory constraints
3. **Learning Rate**: Start with 1e-3, reduce if unstable
4. **Early Stopping**: Prevents overfitting and saves time
5. **Adaptive Sampling**: Use Adaptive PINN for problems with sharp features

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{pinn_benchmark_2024,
  title={PINN Benchmark Framework},
  author={PINN Benchmark Team},
  year={2024},
  url={https://github.com/...}
}
```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational physics*, 378, 686-707.

2. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208-228.

3. Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific machine learning through physics–informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92(3), 88.

## License

MIT License

## Contact

For questions or contributions, please open an issue or submit a pull request.
