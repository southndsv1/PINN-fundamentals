"""
Vanilla PINN: Standard implementation with MSE data loss + PDE residual loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
from .base_pinn import BasePINN


class VanillaPINN(BasePINN):
    """
    Standard Physics-Informed Neural Network.

    Loss = λ_data * MSE(u_pred, u_true) + λ_physics * MSE(PDE_residual, 0)
           + λ_bc * MSE(BC_residual, 0) + λ_ic * MSE(IC_residual, 0)
    """

    def __init__(
        self,
        pde_residual_fn: Callable,
        **kwargs
    ):
        """
        Initialize Vanilla PINN.

        Args:
            pde_residual_fn: Function that computes PDE residual
                            Should take (network, x) and return residual
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        self.pde_residual_fn = pde_residual_fn

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual at collocation points.

        Args:
            x: Collocation points

        Returns:
            PDE residual
        """
        return self.pde_residual_fn(self.network, x)

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
        Compute total loss for Vanilla PINN.

        Returns:
            Total loss and dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Physics loss (PDE residual)
        residual = self.compute_pde_residual(x_interior)
        physics_loss = torch.mean(residual ** 2)
        total_loss += self.lambda_physics * physics_loss
        loss_dict['physics_loss'] = physics_loss.item()

        # Data loss (if available)
        if x_data is not None and u_data is not None:
            u_pred = self.network(x_data)
            data_loss = torch.mean((u_pred - u_data) ** 2)
            total_loss += data_loss
            loss_dict['data_loss'] = data_loss.item()
        else:
            loss_dict['data_loss'] = 0.0

        # Boundary condition loss
        if x_boundary is not None and u_boundary is not None:
            u_bc_pred = self.network(x_boundary)
            bc_loss = torch.mean((u_bc_pred - u_boundary) ** 2)
            total_loss += bc_loss
            loss_dict['boundary_loss'] = bc_loss.item()
        else:
            loss_dict['boundary_loss'] = 0.0

        # Initial condition loss
        if x_initial is not None and u_initial is not None:
            u_ic_pred = self.network(x_initial)
            ic_loss = torch.mean((u_ic_pred - u_initial) ** 2)
            total_loss += ic_loss
            loss_dict['initial_loss'] = ic_loss.item()
        else:
            loss_dict['initial_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
