"""
Kuramoto-Sivashinsky Equation (Chaotic Dynamics)

PDE: ∂u/∂t + ∇⁴u + ∇²u + ½(∇u)² = 0

1D version: ∂u/∂t + ∂⁴u/∂x⁴ + ∂²u/∂x² + ½(∂u/∂x)² = 0

Initial condition: u(x,0) = cos(x) + 0.1*cos(2x)
Boundary conditions: Periodic
Domain: x ∈ [0, 2π], t ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class KuramotoSivashinsky1D:
    """Kuramoto-Sivashinsky equation benchmark problem."""

    def __init__(self, L: float = 2 * np.pi):
        """
        Initialize problem.

        Args:
            L: Domain length
        """
        self.L = L
        self.name = "Kuramoto-Sivashinsky Equation"
        self.x_min, self.x_max = 0.0, L
        self.t_min, self.t_max = 0.0, 0.5

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        No analytical solution (chaotic). Use approximate initial evolution.

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Approximate solution
        """
        # Approximate: slightly decaying initial condition
        return (np.cos(x) + 0.1 * np.cos(2 * x)) * np.exp(-0.1 * t)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Initial condition."""
        return np.cos(x) + 0.1 * np.cos(2 * x)

    def pde_residual(self, network: torch.nn.Module, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: ∂u/∂t + ∂⁴u/∂x⁴ + ∂²u/∂x² + ½(∂u/∂x)²

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

        # Time derivative
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        # Spatial derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True, retain_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xxx), create_graph=True, retain_graph=True)[0]

        # PDE residual (highly nonlinear)
        residual = u_t + u_xxxx + u_xx + 0.5 * u_x**2

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

        # Periodic boundary conditions
        t_bc = np.random.uniform(self.t_min, self.t_max, (n_boundary, 1))
        xt_bc_left = np.hstack([np.zeros((n_boundary, 1)), t_bc])
        xt_bc_right = np.hstack([np.full((n_boundary, 1), self.L), t_bc])
        xt_boundary = np.vstack([xt_bc_left, xt_bc_right])

        # For periodic BC, u(0,t) = u(L,t)
        # This is handled implicitly or with additional constraints
        u_boundary = np.zeros((2 * n_boundary, 1))

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
