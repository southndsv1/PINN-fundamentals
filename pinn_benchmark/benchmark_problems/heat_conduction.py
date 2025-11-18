"""
1D Heat Conduction Problem

PDE: ∂u/∂t = α∂²u/∂x²
Initial condition: u(x,0) = sin(πx)
Boundary conditions: u(0,t) = u(1,t) = 0
Domain: x ∈ [0,1], t ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple, Callable


class HeatConduction1D:
    """1D Heat Conduction benchmark problem."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize problem.

        Args:
            alpha: Thermal diffusivity
        """
        self.alpha = alpha
        self.name = "Heat Conduction 1D"
        self.x_min, self.x_max = 0.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution: u(x,t) = sin(πx) * exp(-π²αt)

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Exact solution
        """
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * self.alpha * t)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """
        Initial condition: u(x,0) = sin(πx)

        Args:
            x: Spatial coordinates

        Returns:
            Initial values
        """
        return np.sin(np.pi * x)

    def boundary_condition(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Boundary conditions: u(0,t) = u(1,t) = 0

        Args:
            t: Time coordinates

        Returns:
            Values at x=0 and x=1
        """
        return np.zeros_like(t), np.zeros_like(t)

    def pde_residual(self, network: torch.nn.Module, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: ∂u/∂t - α∂²u/∂x²

        Args:
            network: Neural network
            xt: Points [x, t]

        Returns:
            PDE residual
        """
        # Ensure requires_grad is enabled
        if not xt.requires_grad:
            xt.requires_grad = True

        # Network prediction
        u = network(xt)

        # Extract coordinates for gradient computation
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        # First derivatives
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # PDE residual
        residual = u_t - self.alpha * u_xx

        return residual

    def get_training_data(
        self,
        n_interior: int = 1000,
        n_boundary: int = 100,
        n_initial: int = 100,
        device: str = 'cpu'
    ) -> dict:
        """
        Generate training data.

        Args:
            n_interior: Number of interior collocation points
            n_boundary: Number of boundary points
            n_initial: Number of initial condition points
            device: Device to use

        Returns:
            Dictionary with training data
        """
        # Interior points (Latin Hypercube Sampling approximation)
        x_int = np.random.uniform(self.x_min, self.x_max, (n_interior, 1))
        t_int = np.random.uniform(self.t_min, self.t_max, (n_interior, 1))
        xt_interior = np.hstack([x_int, t_int])

        # Boundary points
        t_bc = np.random.uniform(self.t_min, self.t_max, (n_boundary, 1))
        x_bc_left = np.zeros((n_boundary, 1))
        x_bc_right = np.ones((n_boundary, 1))

        xt_bc_left = np.hstack([x_bc_left, t_bc])
        xt_bc_right = np.hstack([x_bc_right, t_bc])
        xt_boundary = np.vstack([xt_bc_left, xt_bc_right])

        u_bc_left, u_bc_right = self.boundary_condition(t_bc)
        u_boundary = np.vstack([u_bc_left, u_bc_right])

        # Initial condition points
        x_ic = np.random.uniform(self.x_min, self.x_max, (n_initial, 1))
        t_ic = np.zeros((n_initial, 1))
        xt_initial = np.hstack([x_ic, t_ic])
        u_initial = self.initial_condition(x_ic)

        # Convert to tensors
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
        """
        Generate test data for evaluation.

        Args:
            nx: Number of spatial points
            nt: Number of temporal points

        Returns:
            x, t, u_exact meshgrids
        """
        x = np.linspace(self.x_min, self.x_max, nx)
        t = np.linspace(self.t_min, self.t_max, nt)
        X, T = np.meshgrid(x, t)

        U_exact = self.analytical_solution(X, T)

        return X, T, U_exact
