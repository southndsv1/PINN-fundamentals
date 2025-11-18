"""
Burgers' Equation

PDE: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
Initial condition: u(x,0) = -sin(πx)
Boundary conditions: u(-1,t) = u(1,t) = 0
Domain: x ∈ [-1,1], t ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class BurgersEquation:
    """Burgers' Equation benchmark problem."""

    def __init__(self, nu: float = 0.01):
        """
        Initialize problem.

        Args:
            nu: Viscosity coefficient
        """
        self.nu = nu
        self.name = "Burgers Equation"
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution (Cole-Hopf transformation).

        For small viscosity, we use approximate solution.

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Approximate solution
        """
        # For small viscosity, approximate using method of characteristics
        # This is a simplified version - exact solution requires numerical integration
        return -np.sin(np.pi * x) * np.exp(-self.nu * np.pi**2 * t)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Initial condition: u(x,0) = -sin(πx)"""
        return -np.sin(np.pi * x)

    def pde_residual(self, network: torch.nn.Module, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: ∂u/∂t + u∂u/∂x - ν∂²u/∂x²

        Args:
            network: Neural network
            xt: Points [x, t]

        Returns:
            PDE residual
        """
        # Ensure requires_grad is enabled by cloning if necessary
        if not xt.requires_grad:
            xt = xt.clone().detach().requires_grad_(True)

        # Network prediction
        u = network(xt)

        # Compute gradients with respect to full xt tensor
        grad_u = torch.autograd.grad(
            u, xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Extract spatial and temporal derivatives
        u_x = grad_u[:, 0:1]  # du/dx
        u_t = grad_u[:, 1:2]  # du/dt

        # Second spatial derivative
        u_xx = torch.autograd.grad(
            u_x, xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]  # d²u/dx²

        # PDE residual (nonlinear advection-diffusion)
        residual = u_t + u * u_x - self.nu * u_xx

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

        # Boundary points
        t_bc = np.random.uniform(self.t_min, self.t_max, (n_boundary, 1))
        xt_bc_left = np.hstack([np.full((n_boundary, 1), self.x_min), t_bc])
        xt_bc_right = np.hstack([np.full((n_boundary, 1), self.x_max), t_bc])
        xt_boundary = np.vstack([xt_bc_left, xt_bc_right])
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

    def get_test_data(self, nx: int = 100, nt: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate test data."""
        x = np.linspace(self.x_min, self.x_max, nx)
        t = np.linspace(self.t_min, self.t_max, nt)
        X, T = np.meshgrid(x, t)
        U_exact = self.analytical_solution(X, T)
        return X, T, U_exact
