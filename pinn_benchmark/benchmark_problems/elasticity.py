"""
2D Linear Elasticity

Governing equations (plane stress):
∂σ_xx/∂x + ∂σ_xy/∂y = 0
∂σ_xy/∂x + ∂σ_yy/∂y = 0

Stress-strain relations:
σ_xx = E/(1-ν²) * (ε_xx + ν*ε_yy)
σ_yy = E/(1-ν²) * (ε_yy + ν*ε_xx)
σ_xy = E/(2(1+ν)) * ε_xy

Strain-displacement:
ε_xx = ∂u/∂x, ε_yy = ∂v/∂y, ε_xy = (∂u/∂y + ∂v/∂x)/2

Domain: x,y ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class LinearElasticity2D:
    """2D Linear Elasticity benchmark problem."""

    def __init__(self, E: float = 1.0, nu: float = 0.3):
        """
        Initialize problem.

        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        self.E = E
        self.nu = nu
        self.name = "Linear Elasticity 2D"
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0

    def analytical_solution(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analytical solution for simple loading case.

        Returns:
            u, v (displacement fields)
        """
        # Simple extension in x-direction
        u = 0.01 * x
        v = -0.01 * self.nu * y
        return u, v

    def pde_residual(self, network: torch.nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual (equilibrium equations).

        Network outputs: [u, v] (displacements)

        Returns:
            Residual
        """
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        x.requires_grad = True
        y.requires_grad = True

        xy_grad = torch.cat([x, y], dim=1)
        out = network(xy_grad)

        u = out[:, 0:1]
        v = out[:, 1:2]

        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

        # Strains
        eps_xx = u_x
        eps_yy = v_y
        eps_xy = 0.5 * (u_y + v_x)

        # Stresses (plane stress)
        C = self.E / (1 - self.nu**2)
        sigma_xx = C * (eps_xx + self.nu * eps_yy)
        sigma_yy = C * (eps_yy + self.nu * eps_xx)
        sigma_xy = self.E / (2 * (1 + self.nu)) * eps_xy

        # Stress gradients
        sigma_xx_x = torch.autograd.grad(sigma_xx, x, grad_outputs=torch.ones_like(sigma_xx), create_graph=True, retain_graph=True)[0]
        sigma_xy_y = torch.autograd.grad(sigma_xy, y, grad_outputs=torch.ones_like(sigma_xy), create_graph=True, retain_graph=True)[0]
        sigma_xy_x = torch.autograd.grad(sigma_xy, x, grad_outputs=torch.ones_like(sigma_xy), create_graph=True, retain_graph=True)[0]
        sigma_yy_y = torch.autograd.grad(sigma_yy, y, grad_outputs=torch.ones_like(sigma_yy), create_graph=True, retain_graph=True)[0]

        # Equilibrium equations
        eq_x = sigma_xx_x + sigma_xy_y
        eq_y = sigma_xy_x + sigma_yy_y

        residual = torch.cat([eq_x, eq_y], dim=1)

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

        # Boundary points - fixed left edge
        x_left = np.zeros((n_boundary, 1))
        y_left = np.random.uniform(self.y_min, self.y_max, (n_boundary, 1))
        xy_boundary = np.hstack([x_left, y_left])
        uv_boundary = np.zeros((n_boundary, 2))

        data = {
            'x_interior': torch.tensor(xy_interior, dtype=torch.float32, device=device, requires_grad=True),
            'x_boundary': torch.tensor(xy_boundary, dtype=torch.float32, device=device),
            'u_boundary': torch.tensor(uv_boundary, dtype=torch.float32, device=device),
            'bounds': np.array([[self.x_min, self.x_max], [self.y_min, self.y_max]])
        }

        return data

    def get_test_data(self, nx: int = 30, ny: int = 30):
        """Generate test data."""
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        X, Y = np.meshgrid(x, y)
        U, V = self.analytical_solution(X, Y)
        return X, Y, U, V
