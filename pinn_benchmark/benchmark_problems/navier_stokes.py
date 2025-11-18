"""
2D Steady Navier-Stokes (Simplified Lid-Driven Cavity)

Equations:
u∂u/∂x + v∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
u∂v/∂x + v∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
∂u/∂x + ∂v/∂y = 0 (continuity)

Domain: x,y ∈ [0,1]
Boundary: u=v=0 on walls, u=1 on top lid
"""

import torch
import numpy as np
from typing import Tuple


class NavierStokes2D:
    """2D Steady Navier-Stokes benchmark problem."""

    def __init__(self, nu: float = 0.01, U_lid: float = 1.0):
        """
        Initialize problem.

        Args:
            nu: Kinematic viscosity
            U_lid: Lid velocity
        """
        self.nu = nu
        self.U_lid = U_lid
        self.name = "Navier-Stokes 2D"
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0

    def analytical_solution(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        No simple analytical solution for lid-driven cavity.
        Use reference numerical solution or simplified approximation.

        Returns:
            u, v, p (velocity and pressure fields)
        """
        # Simplified approximate solution for demonstration
        u = self.U_lid * y * (2 - y)
        v = np.zeros_like(x)
        p = np.zeros_like(x)
        return u, v, p

    def pde_residual(self, network: torch.nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residuals for momentum and continuity equations.

        Network outputs: [u, v, p]

        Returns:
            Combined residual
        """
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        x.requires_grad = True
        y.requires_grad = True

        xy_grad = torch.cat([x, y], dim=1)
        out = network(xy_grad)

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        # Compute gradients
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]

        # Momentum equations
        momentum_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        # Continuity equation
        continuity = u_x + v_y

        # Combined residual
        residual = torch.cat([momentum_x, momentum_y, continuity], dim=1)

        return residual

    def get_training_data(
        self,
        n_interior: int = 1000,
        n_boundary: int = 100,
        device: str = 'cpu'
    ) -> dict:
        """Generate training data."""
        # Interior points
        x_int = np.random.uniform(self.x_min, self.x_max, (n_interior, 1))
        y_int = np.random.uniform(self.y_min, self.y_max, (n_interior, 1))
        xy_interior = np.hstack([x_int, y_int])

        # Boundary points
        # Bottom, top, left, right
        x_bottom = np.random.uniform(self.x_min, self.x_max, (n_boundary, 1))
        y_bottom = np.zeros((n_boundary, 1))

        x_top = np.random.uniform(self.x_min, self.x_max, (n_boundary, 1))
        y_top = np.ones((n_boundary, 1))

        x_left = np.zeros((n_boundary, 1))
        y_left = np.random.uniform(self.y_min, self.y_max, (n_boundary, 1))

        x_right = np.ones((n_boundary, 1))
        y_right = np.random.uniform(self.y_min, self.y_max, (n_boundary, 1))

        xy_boundary = np.vstack([
            np.hstack([x_bottom, y_bottom]),
            np.hstack([x_top, y_top]),
            np.hstack([x_left, y_left]),
            np.hstack([x_right, y_right])
        ])

        # Boundary values [u, v, p]
        u_bottom = np.zeros((n_boundary, 1))
        u_top = np.full((n_boundary, 1), self.U_lid)
        u_left = np.zeros((n_boundary, 1))
        u_right = np.zeros((n_boundary, 1))

        v_boundary = np.zeros((4 * n_boundary, 1))
        p_boundary = np.zeros((4 * n_boundary, 1))  # Reference pressure

        u_boundary = np.vstack([u_bottom, u_top, u_left, u_right])
        uvp_boundary = np.hstack([u_boundary, v_boundary, p_boundary])

        data = {
            'x_interior': torch.tensor(xy_interior, dtype=torch.float32, device=device, requires_grad=True),
            'x_boundary': torch.tensor(xy_boundary, dtype=torch.float32, device=device),
            'u_boundary': torch.tensor(uvp_boundary, dtype=torch.float32, device=device),
            'bounds': np.array([[self.x_min, self.x_max], [self.y_min, self.y_max]])
        }

        return data

    def get_test_data(self, nx: int = 50, ny: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate test data."""
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        X, Y = np.meshgrid(x, y)
        U, V, P = self.analytical_solution(X, Y)
        return X, Y, U, V, P
