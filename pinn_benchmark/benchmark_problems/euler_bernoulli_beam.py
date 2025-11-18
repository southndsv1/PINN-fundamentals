"""
Euler-Bernoulli Beam Deflection

PDE: d⁴w/dx⁴ = q(x)/EI
where w is deflection, q is distributed load, E is Young's modulus, I is moment of inertia

Problem: Simply supported beam with uniform load
Boundary conditions: w(0) = w(L) = 0, w''(0) = w''(L) = 0
Domain: x ∈ [0,1]
"""

import torch
import numpy as np
from typing import Tuple


class EulerBernoulliBeam:
    """Euler-Bernoulli Beam deflection benchmark problem."""

    def __init__(self, EI: float = 1.0, q0: float = 1.0, L: float = 1.0):
        """
        Initialize problem.

        Args:
            EI: Flexural rigidity
            q0: Uniform load intensity
            L: Beam length
        """
        self.EI = EI
        self.q0 = q0
        self.L = L
        self.name = "Euler-Bernoulli Beam"
        self.x_min, self.x_max = 0.0, L

    def load_function(self, x: np.ndarray) -> np.ndarray:
        """Uniform distributed load."""
        return np.full_like(x, self.q0)

    def analytical_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Analytical solution for simply supported beam with uniform load:
        w(x) = (q0/(24EI)) * x * (L³ - 2Lx² + x³)

        Args:
            x: Position along beam

        Returns:
            Deflection
        """
        return (self.q0 / (24 * self.EI)) * x * (
            self.L**3 - 2 * self.L * x**2 + x**3
        )

    def pde_residual(self, network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: d⁴w/dx⁴ - q(x)/EI

        Args:
            network: Neural network
            x: Position points

        Returns:
            PDE residual
        """
        x.requires_grad = True

        w = network(x)

        # First derivative
        w_x = torch.autograd.grad(
            w, x,
            grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True
        )[0]

        # Second derivative
        w_xx = torch.autograd.grad(
            w_x, x,
            grad_outputs=torch.ones_like(w_x),
            create_graph=True,
            retain_graph=True
        )[0]

        # Third derivative
        w_xxx = torch.autograd.grad(
            w_xx, x,
            grad_outputs=torch.ones_like(w_xx),
            create_graph=True,
            retain_graph=True
        )[0]

        # Fourth derivative
        w_xxxx = torch.autograd.grad(
            w_xxx, x,
            grad_outputs=torch.ones_like(w_xxx),
            create_graph=True,
            retain_graph=True
        )[0]

        # Load
        q = torch.full_like(x, self.q0)

        # PDE residual
        residual = w_xxxx - q / self.EI

        return residual

    def get_training_data(
        self,
        n_interior: int = 1000,
        n_boundary: int = 100,
        device: str = 'cpu'
    ) -> dict:
        """Generate training data."""
        # Interior points
        x_interior = np.random.uniform(self.x_min, self.x_max, (n_interior, 1))

        # Boundary points (deflection BC)
        x_bc = np.array([[0.0], [self.L]])
        w_bc = np.zeros((2, 1))

        data = {
            'x_interior': torch.tensor(x_interior, dtype=torch.float32, device=device, requires_grad=True),
            'x_boundary': torch.tensor(x_bc, dtype=torch.float32, device=device),
            'u_boundary': torch.tensor(w_bc, dtype=torch.float32, device=device),
            'bounds': np.array([[self.x_min, self.x_max]])
        }

        return data

    def get_test_data(self, nx: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test data."""
        x = np.linspace(self.x_min, self.x_max, nx).reshape(-1, 1)
        w_exact = self.analytical_solution(x)
        return x, w_exact
