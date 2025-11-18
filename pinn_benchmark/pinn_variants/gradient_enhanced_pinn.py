"""
Gradient-Enhanced PINN: Includes derivative matching in loss function.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable
from .base_pinn import BasePINN


class GradientEnhancedPINN(BasePINN):
    """
    Gradient-Enhanced Physics-Informed Neural Network.

    Includes derivative matching in loss function when analytical gradients are available.
    Improves accuracy by training on both function values and their derivatives.
    """

    def __init__(
        self,
        pde_residual_fn: Callable,
        lambda_gradient: float = 1.0,
        **kwargs
    ):
        """
        Initialize Gradient-Enhanced PINN.

        Args:
            pde_residual_fn: Function that computes PDE residual
            lambda_gradient: Weight for gradient matching loss
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        self.pde_residual_fn = pde_residual_fn
        self.lambda_gradient = lambda_gradient

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
        u_data: Optional[torch.Tensor] = None,
        x_gradient: Optional[torch.Tensor] = None,
        u_gradient: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss including gradient matching.

        Args:
            x_gradient: Points where gradient data is available
            u_gradient: True gradient values at x_gradient points

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

        # Data loss
        if x_data is not None and u_data is not None:
            u_pred = self.network(x_data)
            data_loss = torch.mean((u_pred - u_data) ** 2)
            total_loss += data_loss
            loss_dict['data_loss'] = data_loss.item()
        else:
            loss_dict['data_loss'] = 0.0

        # Gradient matching loss (enhanced feature)
        if x_gradient is not None and u_gradient is not None:
            u_pred = self.network(x_gradient)
            u_pred_grad = torch.autograd.grad(
                u_pred, x_gradient,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True,
                retain_graph=True
            )[0]

            gradient_loss = torch.mean((u_pred_grad - u_gradient) ** 2)
            total_loss += self.lambda_gradient * gradient_loss
            loss_dict['gradient_loss'] = gradient_loss.item()
        else:
            loss_dict['gradient_loss'] = 0.0

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

    def train_step(
        self,
        x_interior: torch.Tensor,
        x_boundary: Optional[torch.Tensor] = None,
        u_boundary: Optional[torch.Tensor] = None,
        x_initial: Optional[torch.Tensor] = None,
        u_initial: Optional[torch.Tensor] = None,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        x_gradient: Optional[torch.Tensor] = None,
        u_gradient: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step with gradient data.

        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()

        loss, loss_dict = self.compute_loss(
            x_interior, x_boundary, u_boundary,
            x_initial, u_initial, x_data, u_data,
            x_gradient, u_gradient
        )

        loss.backward()
        self.optimizer.step()

        return loss_dict
