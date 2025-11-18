"""
Allen-Cahn Equation (Phase Field Model)

PDE: ∂u/∂t = ε²∇²u + u - u³
where ε controls interface thickness

1D version: ∂u/∂t = ε²∂²u/∂x² + u - u³

Initial condition: u(x,0) = x² * cos(πx)
Boundary conditions: Periodic or fixed
Domain: x ∈ [-1,1], t ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class AllenCahn1D:
    """Allen-Cahn equation benchmark problem."""

    def __init__(self, epsilon: float = 0.1):
        """
        Initialize problem.

        Args:
            epsilon: Interface thickness parameter
        """
        self.epsilon = epsilon
        self.name = "Allen-Cahn Equation"
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 0.5

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        No simple analytical solution. Use approximate solution.

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Approximate solution
        """
        # Approximate: decaying oscillation
        return (x**2 * np.cos(np.pi * x)) * np.exp(-t)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Initial condition."""
        return x**2 * np.cos(np.pi * x)

    def pde_residual(self, network: torch.nn.Module, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: ∂u/∂t - ε²∂²u/∂x² - u + u³

        Args:
            network: Neural network
            xt: Points [x, t]

        Returns:
            PDE residual
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        x.requires_grad = True
        t.requires_grad = True

        xt_grad = torch.cat([x, t], dim=1)
        u = network(xt_grad)

        # Derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        # PDE residual (nonlinear reaction-diffusion)
        residual = u_t - self.epsilon**2 * u_xx - u + u**3

        return residual

    def get_training_data(
        self,
        n_interior: int = 1000,
        n_boundary: int = 100,
        n_initial: int = 100,
        device: str = 'cpu'
    ) -> dict:
        """Generate training data."""
        # Interior points
        x_int = np.random.uniform(self.x_min, self.x_max, (n_interior, 1))
        t_int = np.random.uniform(self.t_min, self.t_max, (n_interior, 1))
        xt_interior = np.hstack([x_int, t_int])

        # Boundary points (Neumann BC: u_x = 0)
        t_bc = np.random.uniform(self.t_min, self.t_max, (n_boundary, 1))
        xt_bc_left = np.hstack([np.full((n_boundary, 1), self.x_min), t_bc])
        xt_bc_right = np.hstack([np.full((n_boundary, 1), self.x_max), t_bc])
        xt_boundary = np.vstack([xt_bc_left, xt_bc_right])
        u_boundary = np.zeros((2 * n_boundary, 1))  # Simplified BC

        # Initial condition
        x_ic = np.random.uniform(self.x_min, self.x_max, (n_initial, 1))
        t_ic = np.zeros((n_initial, 1))
        xt_initial = np.hstack([x_ic, t_ic])
        u_initial = self.initial_condition(x_ic)

        data = {
            'x_interior': torch.tensor(xt_interior, dtype=torch.float32, device=device, requires_grad=True),
            'x_boundary': torch.tensor(xt_boundary, dtype=torch.float32, device=device),
            'u_boundary': torch.tensor(u_boundary, dtype=torch.float32, device=device),
            'x_initial': torch.tensor(xt_initial, dtype=torch.float32, device=device),
            'u_initial': torch.tensor(u_initial, dtype=torch.float32, device=device),
            'bounds': np.array([[self.x_min, self.x_max], [self.t_min, self.t_max]])
        }

        return data

    def get_test_data(self, nx: int = 100, nt: int = 100):
        """Generate test data."""
        x = np.linspace(self.x_min, self.x_max, nx)
        t = np.linspace(self.t_min, self.t_max, nt)
        X, T = np.meshgrid(x, t)
        U_exact = self.analytical_solution(X, T)
        return X, T, U_exact
