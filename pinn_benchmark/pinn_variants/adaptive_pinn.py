"""
Adaptive PINN: Implements dynamic weight adjustment and adaptive collocation point sampling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from .base_pinn import BasePINN


class AdaptivePINN(BasePINN):
    """
    Adaptive Physics-Informed Neural Network.

    Features:
    - Dynamic weight adjustment using gradient statistics
    - Adaptive collocation point sampling based on residuals
    """

    def __init__(
        self,
        pde_residual_fn: Callable,
        adaptive_weights: bool = True,
        adaptive_sampling: bool = True,
        resample_freq: int = 100,
        **kwargs
    ):
        """
        Initialize Adaptive PINN.

        Args:
            pde_residual_fn: Function that computes PDE residual
            adaptive_weights: Enable adaptive loss weight adjustment
            adaptive_sampling: Enable adaptive collocation point sampling
            resample_freq: Frequency (in epochs) to resample collocation points
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        self.pde_residual_fn = pde_residual_fn
        self.adaptive_weights = adaptive_weights
        self.adaptive_sampling = adaptive_sampling
        self.resample_freq = resample_freq

        # Adaptive weights (initialized to 1.0)
        self.lambda_data_adaptive = 1.0
        self.lambda_bc_adaptive = 1.0
        self.lambda_ic_adaptive = 1.0

        # For adaptive sampling
        self.residual_history = []

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual at collocation points.

        Args:
            x: Collocation points

        Returns:
            PDE residual
        """
        return self.pde_residual_fn(self.network, x)

    def update_adaptive_weights(self, loss_dict: Dict[str, float]):
        """
        Update adaptive weights based on gradient statistics.

        Uses the inverse of mean gradient magnitudes to balance different loss terms.
        """
        if not self.adaptive_weights:
            return

        # Compute mean gradients for each loss component
        grad_norms = {}

        # This is a simplified version - in practice, we'd track actual gradients
        # Here we use loss magnitudes as proxies
        physics_loss = loss_dict.get('physics_loss', 1.0)
        data_loss = loss_dict.get('data_loss', 1.0)
        bc_loss = loss_dict.get('boundary_loss', 1.0)
        ic_loss = loss_dict.get('initial_loss', 1.0)

        # Adaptive weight update (prevent division by zero)
        eps = 1e-8
        if data_loss > eps:
            self.lambda_data_adaptive = physics_loss / (data_loss + eps)
        if bc_loss > eps:
            self.lambda_bc_adaptive = physics_loss / (bc_loss + eps)
        if ic_loss > eps:
            self.lambda_ic_adaptive = physics_loss / (ic_loss + eps)

        # Clamp weights to reasonable range
        self.lambda_data_adaptive = np.clip(self.lambda_data_adaptive, 0.1, 10.0)
        self.lambda_bc_adaptive = np.clip(self.lambda_bc_adaptive, 0.1, 10.0)
        self.lambda_ic_adaptive = np.clip(self.lambda_ic_adaptive, 0.1, 10.0)

    def adaptive_resample(self, x_interior: torch.Tensor, bounds: np.ndarray) -> torch.Tensor:
        """
        Resample collocation points based on residual magnitudes.

        High residual areas get more points.
        """
        if not self.adaptive_sampling or len(self.residual_history) == 0:
            return x_interior

        # Compute current residuals
        with torch.no_grad():
            residuals = self.compute_pde_residual(x_interior)
            residuals_np = torch.abs(residuals).cpu().numpy().flatten()

        # Normalize residuals to probabilities
        probs = residuals_np / (np.sum(residuals_np) + 1e-8)

        # Sample 50% of points uniformly, 50% weighted by residuals
        n_points = len(x_interior)
        n_uniform = n_points // 2
        n_adaptive = n_points - n_uniform

        # Uniform sampling
        x_uniform = self.latin_hypercube_sampling(n_uniform, bounds)

        # Weighted resampling of existing points
        indices = np.random.choice(
            len(x_interior),
            size=n_adaptive,
            replace=True,
            p=probs
        )
        x_adaptive = x_interior[indices].clone().detach()

        # Add small random perturbation
        noise = torch.randn_like(x_adaptive) * 0.01
        x_adaptive = x_adaptive + noise

        # Combine
        x_new = torch.cat([x_uniform, x_adaptive], dim=0)
        x_new.requires_grad = True

        return x_new

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
        Compute total loss with adaptive weights.

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

        # Store residuals for adaptive sampling
        self.residual_history.append(residual.detach())
        if len(self.residual_history) > 10:
            self.residual_history.pop(0)

        # Data loss with adaptive weight
        if x_data is not None and u_data is not None:
            u_pred = self.network(x_data)
            data_loss = torch.mean((u_pred - u_data) ** 2)
            total_loss += self.lambda_data_adaptive * data_loss
            loss_dict['data_loss'] = data_loss.item()
        else:
            loss_dict['data_loss'] = 0.0

        # Boundary condition loss with adaptive weight
        if x_boundary is not None and u_boundary is not None:
            u_bc_pred = self.network(x_boundary)
            bc_loss = torch.mean((u_bc_pred - u_boundary) ** 2)
            total_loss += self.lambda_bc_adaptive * bc_loss
            loss_dict['boundary_loss'] = bc_loss.item()
        else:
            loss_dict['boundary_loss'] = 0.0

        # Initial condition loss with adaptive weight
        if x_initial is not None and u_initial is not None:
            u_ic_pred = self.network(x_initial)
            ic_loss = torch.mean((u_ic_pred - u_initial) ** 2)
            total_loss += self.lambda_ic_adaptive * ic_loss
            loss_dict['initial_loss'] = ic_loss.item()
        else:
            loss_dict['initial_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        # Update adaptive weights
        self.update_adaptive_weights(loss_dict)

        return total_loss, loss_dict
