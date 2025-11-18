"""
1D Wave Propagation Problem

PDE: ∂²u/∂t² = c²∂²u/∂x²
Initial displacement: u(x,0) = sin(πx)
Initial velocity: ∂u/∂t(x,0) = 0
Boundary conditions: u(0,t) = u(1,t) = 0
Domain: x ∈ [0,1], t ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class WavePropagation1D:
    """1D Wave Propagation benchmark problem."""

    def __init__(self, c: float = 1.0):
        """
        Initialize problem.

        Args:
            c: Wave speed
        """
        self.c = c
        self.name = "Wave Propagation 1D"
        self.x_min, self.x_max = 0.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

    def analytical_solution(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution: u(x,t) = sin(πx)cos(πct)

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Exact solution
        """
        return np.sin(np.pi * x) * np.cos(np.pi * self.c * t)

    def initial_displacement(self, x: np.ndarray) -> np.ndarray:
        """Initial displacement: u(x,0) = sin(πx)"""
        return np.sin(np.pi * x)

    def initial_velocity(self, x: np.ndarray) -> np.ndarray:
        """Initial velocity: ∂u/∂t(x,0) = 0"""
        return np.zeros_like(x)

    def pde_residual(self, network: torch.nn.Module, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: ∂²u/∂t² - c²∂²u/∂x²

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

        # Second derivatives
        u_tt = torch.autograd.grad(
            u_t, t,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # PDE residual
        residual = u_tt - self.c**2 * u_xx

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
        xt_bc_left = np.hstack([np.zeros((n_boundary, 1)), t_bc])
        xt_bc_right = np.hstack([np.ones((n_boundary, 1)), t_bc])
        xt_boundary = np.vstack([xt_bc_left, xt_bc_right])
        u_boundary = np.zeros((2 * n_boundary, 1))

        # Initial displacement
        x_ic = np.random.uniform(self.x_min, self.x_max, (n_initial, 1))
        t_ic = np.zeros((n_initial, 1))
        xt_initial = np.hstack([x_ic, t_ic])
        u_initial = self.initial_displacement(x_ic)

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
