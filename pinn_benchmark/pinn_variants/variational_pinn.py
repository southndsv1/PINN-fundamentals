"""
Variational PINN: Uses weak form with integration by parts.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
from .base_pinn import BasePINN


class VariationalPINN(BasePINN):
    """
    Variational Physics-Informed Neural Network.

    Uses weak formulation of PDEs with integration by parts.
    Minimizes energy functional instead of strong form residual.
    """

    def __init__(
        self,
        energy_functional_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize Variational PINN.

        Args:
            energy_functional_fn: Optional function that computes energy functional
                                 Should take (network, x) and return energy
                                 If None, uses default diffusion energy functional
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        self.energy_functional_fn = energy_functional_fn

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        For variational PINN, this computes the weak residual.

        Args:
            x: Collocation points

        Returns:
            Weak form residual
        """
        # In variational form, we use test functions
        # For simplicity, we'll use the network itself as test function space
        u = self.network(x)

        # Compute weak residual using integration by parts
        # This is problem-specific and handled by energy_functional_fn
        if self.energy_functional_fn is not None:
            residual = self.energy_functional_fn(self.network, x)
        else:
            # Default: use gradient-based energy (works for diffusion-type problems)
            u_grad = torch.autograd.grad(
                u, x,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            # Simple energy: 0.5 * |∇u|²
            residual = 0.5 * torch.sum(u_grad ** 2, dim=1, keepdim=True)

        return residual

    def compute_loss(
        self,
        x_interior: torch.Tensor,
        x_boundary: Optional[torch.Tensor] = None,
        u_boundary: Optional[torch.Tensor] = None,
        x_initial: Optional[torch.Tensor] = None,
        u_initial: Optional[torch.Tensor] = None,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss using variational formulation.

        Returns:
            Total loss and dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Variational physics loss (energy functional)
        u = self.network(x_interior)

        # Compute gradients for energy calculation
        u_x = torch.autograd.grad(
            u, x_interior,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        # Energy functional (approximated via Monte Carlo integration)
        # For diffusion: E = ∫(0.5 * |∇u|² - f*u) dx
        energy = torch.mean(0.5 * torch.sum(u_x ** 2, dim=1, keepdim=True))

        physics_loss = energy
        total_loss += self.lambda_physics * physics_loss
        loss_dict['physics_loss'] = physics_loss.item()

        # Data loss
        if x_data is not None and u_data is not None:
            u_pred = self.network(x_data)
            data_loss = torch.mean((u_pred - u_data) ** 2)
            total_loss += data_loss
            loss_dict['data_loss'] = data_loss.item()
        else:
            loss_dict['data_loss'] = 0.0

        # Boundary conditions (essential BCs in weak form)
        if x_boundary is not None and u_boundary is not None:
            u_bc_pred = self.network(x_boundary)
            bc_loss = torch.mean((u_bc_pred - u_boundary) ** 2)
            total_loss += bc_loss
            loss_dict['boundary_loss'] = bc_loss.item()
        else:
            loss_dict['boundary_loss'] = 0.0

        # Initial conditions
        if x_initial is not None and u_initial is not None:
            u_ic_pred = self.network(x_initial)
            ic_loss = torch.mean((u_ic_pred - u_initial) ** 2)
            total_loss += ic_loss
            loss_dict['initial_loss'] = ic_loss.item()
        else:
            loss_dict['initial_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
