"""
Kirchhoff Plate Vibration

PDE: ∇⁴w + (ρh/D)∂²w/∂t² = p(x,y,t)/D
where w is displacement, ρ is density, h is thickness, D is flexural rigidity

Simplified problem: Free vibration of simply supported plate
∇⁴w + (ρh/D)∂²w/∂t² = 0
Domain: x,y ∈ [0,1], t ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class PlateVibration:
    """Kirchhoff plate vibration benchmark problem."""

    def __init__(self, D: float = 1.0, rho_h: float = 1.0):
        """
        Initialize problem.

        Args:
            D: Flexural rigidity
            rho_h: Mass per unit area (ρh)
        """
        self.D = D
        self.rho_h = rho_h
        self.name = "Plate Vibration"
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

    def analytical_solution(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution for simply supported plate:
        w(x,y,t) = sin(πx)sin(πy)cos(ωt)
        where ω² = 2π⁴D/(ρh)
        """
        omega_sq = 2 * np.pi**4 * self.D / self.rho_h
        omega = np.sqrt(omega_sq)
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)

    def initial_condition(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial displacement."""
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def pde_residual(self, network: torch.nn.Module, xyt: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: ∇⁴w + (ρh/D)∂²w/∂t²

        Args:
            network: Neural network
            xyt: Points [x, y, t]

        Returns:
            PDE residual
        """
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]

        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        xyt_grad = torch.cat([x, y, t], dim=1)
        w = network(xyt_grad)

        # Time derivatives
        w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]
        w_tt = torch.autograd.grad(w_t, t, grad_outputs=torch.ones_like(w_t), create_graph=True, retain_graph=True)[0]

        # Spatial derivatives
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0]

        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True, retain_graph=True)[0]
        w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True, retain_graph=True)[0]

        w_xxx = torch.autograd.grad(w_xx, x, grad_outputs=torch.ones_like(w_xx), create_graph=True, retain_graph=True)[0]
        w_yyy = torch.autograd.grad(w_yy, y, grad_outputs=torch.ones_like(w_yy), create_graph=True, retain_graph=True)[0]

        w_xxxx = torch.autograd.grad(w_xxx, x, grad_outputs=torch.ones_like(w_xxx), create_graph=True, retain_graph=True)[0]
        w_yyyy = torch.autograd.grad(w_yyy, y, grad_outputs=torch.ones_like(w_yyy), create_graph=True, retain_graph=True)[0]

        w_xxyy = torch.autograd.grad(w_xx, y, grad_outputs=torch.ones_like(w_xx), create_graph=True, retain_graph=True)[0]
        w_xxyy = torch.autograd.grad(w_xxyy, y, grad_outputs=torch.ones_like(w_xxyy), create_graph=True, retain_graph=True)[0]

        # Biharmonic operator
        laplacian_squared = w_xxxx + 2 * w_xxyy + w_yyyy

        # PDE residual
        residual = laplacian_squared + (self.rho_h / self.D) * w_tt

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
        y_int = np.random.uniform(self.y_min, self.y_max, (n_interior, 1))
        t_int = np.random.uniform(self.t_min, self.t_max, (n_interior, 1))
        xyt_interior = np.hstack([x_int, y_int, t_int])

        # Boundary points (edges at all times)
        t_bc = np.random.uniform(self.t_min, self.t_max, (n_boundary, 1))
        boundaries = []
        for edge in [[0.0, 'x'], [1.0, 'x'], [0.0, 'y'], [1.0, 'y']]:
            if edge[1] == 'x':
                x_bc = np.full((n_boundary, 1), edge[0])
                y_bc = np.random.uniform(self.y_min, self.y_max, (n_boundary, 1))
            else:
                x_bc = np.random.uniform(self.x_min, self.x_max, (n_boundary, 1))
                y_bc = np.full((n_boundary, 1), edge[0])
            boundaries.append(np.hstack([x_bc, y_bc, t_bc]))

        xyt_boundary = np.vstack(boundaries)
        w_boundary = np.zeros((len(xyt_boundary), 1))

        # Initial condition
        x_ic = np.random.uniform(self.x_min, self.x_max, (n_initial, 1))
        y_ic = np.random.uniform(self.y_min, self.y_max, (n_initial, 1))
        t_ic = np.zeros((n_initial, 1))
        xyt_initial = np.hstack([x_ic, y_ic, t_ic])
        w_initial = self.initial_condition(x_ic, y_ic)

        data = {
            'x_interior': torch.tensor(xyt_interior, dtype=torch.float32, device=device, requires_grad=True),
            'x_boundary': torch.tensor(xyt_boundary, dtype=torch.float32, device=device),
            'u_boundary': torch.tensor(w_boundary, dtype=torch.float32, device=device),
            'x_initial': torch.tensor(xyt_initial, dtype=torch.float32, device=device),
            'u_initial': torch.tensor(w_initial, dtype=torch.float32, device=device),
            'bounds': np.array([[self.x_min, self.x_max], [self.y_min, self.y_max], [self.t_min, self.t_max]])
        }

        return data

    def get_test_data(self, nx: int = 30, ny: int = 30, nt: int = 30):
        """Generate test data."""
        x = np.linspace(self.x_min, self.x_max, nx)
        y = np.linspace(self.y_min, self.y_max, ny)
        t = np.linspace(self.t_min, self.t_max, nt)
        X, Y, T = np.meshgrid(x, y, t)
        W_exact = self.analytical_solution(X, Y, T)
        return X, Y, T, W_exact
