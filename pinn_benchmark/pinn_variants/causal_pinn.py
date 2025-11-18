"""
Causal PINN: For time-dependent problems respecting temporal causality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from .base_pinn import BasePINN


class CausalPINN(BasePINN):
    """
    Causal Physics-Informed Neural Network.

    For time-dependent problems, respects temporal causality by:
    - Training in time segments sequentially
    - Using causality weights that emphasize earlier times
    """

    def __init__(
        self,
        pde_residual_fn: Callable,
        n_time_segments: int = 10,
        causal_weight: float = 1.0,
        **kwargs
    ):
        """
        Initialize Causal PINN.

        Args:
            pde_residual_fn: Function that computes PDE residual
            n_time_segments: Number of time segments for sequential training
            causal_weight: Weight emphasizing causality (reduced from 10.0 to 1.0 to prevent NaN)
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(**kwargs)
        self.pde_residual_fn = pde_residual_fn
        self.n_time_segments = n_time_segments
        self.causal_weight = causal_weight
        self.current_time_window = 1.0  # Full time by default

    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual at collocation points.

        Args:
            x: Collocation points

        Returns:
            PDE residual
        """
        return self.pde_residual_fn(self.network, x)

    def compute_causality_weights(self, x: torch.Tensor, time_idx: int = -1) -> torch.Tensor:
        """
        Compute causality weights that emphasize past over future.

        Args:
            x: Collocation points (assumes last dimension is time)
            time_idx: Index of time dimension (default -1 for last)

        Returns:
            Causality weights
        """
        # Extract time coordinate (assume last dimension is time)
        if time_idx == -1:
            t = x[:, -1:]  # Properly extract last column
        else:
            t = x[:, time_idx:time_idx+1]

        # Simple linear weighting to avoid numerical issues with sqrt/exp
        # Earlier times get higher weights
        t_normalized = torch.clamp(t / (self.current_time_window + 1e-8), 0.0, 1.0)
        # Linear decay: weight = 1.0 at t=0, weight decreases as t increases
        # With causal_weight=1.0: goes from 1.0 at t=0 to 0.1 at t=1
        weights = 1.0 - 0.9 * self.causal_weight * t_normalized
        weights = torch.clamp(weights, 0.1, 1.0)  # Keep minimum weight at 0.1

        return weights

    def compute_loss(
        self,
        x_interior: torch.Tensor,
        x_boundary: Optional[torch.Tensor] = None,
        u_boundary: Optional[torch.Tensor] = None,
        x_initial: Optional[torch.Tensor] = None,
        u_initial: Optional[torch.Tensor] = None,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        apply_causality: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with causality weighting.

        Returns:
            Total loss and dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Physics loss (PDE residual) with causality weights
        residual = self.compute_pde_residual(x_interior)

        if apply_causality and x_interior.shape[1] > 1:
            # Assume last dimension is time
            weights = self.compute_causality_weights(x_interior)
            physics_loss = torch.mean(weights * residual ** 2)
        else:
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

        # Boundary condition loss
        if x_boundary is not None and u_boundary is not None:
            u_bc_pred = self.network(x_boundary)
            bc_loss = torch.mean((u_bc_pred - u_boundary) ** 2)
            total_loss += bc_loss
            loss_dict['boundary_loss'] = bc_loss.item()
        else:
            loss_dict['boundary_loss'] = 0.0

        # Initial condition loss (moderately weighted in causal training)
        if x_initial is not None and u_initial is not None:
            u_ic_pred = self.network(x_initial)
            ic_loss = torch.mean((u_ic_pred - u_initial) ** 2)
            # Use fixed weight of 2.0 instead of causal_weight to prevent gradient explosion
            total_loss += 2.0 * ic_loss
            loss_dict['initial_loss'] = ic_loss.item()
        else:
            loss_dict['initial_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def train_sequential(
        self,
        n_epochs_per_segment: int,
        x_interior_fn: Callable,  # Function to generate interior points for time window
        t_final: float,
        **kwargs
    ) -> Dict:
        """
        Train sequentially in time segments.

        Args:
            n_epochs_per_segment: Epochs to train per time segment
            x_interior_fn: Function that takes (t_min, t_max) and returns interior points
            t_final: Final time
            **kwargs: Additional arguments for training

        Returns:
            Training history
        """
        # Divide time into segments
        time_segments = np.linspace(0, t_final, self.n_time_segments + 1)

        for i in range(self.n_time_segments):
            t_min = time_segments[i]
            t_max = time_segments[i + 1]
            self.current_time_window = t_max

            print(f"Training time segment [{t_min:.3f}, {t_max:.3f}]")

            # Generate collocation points for this time window
            x_interior = x_interior_fn(t_min, t_max)

            # Train on this segment
            self.train(
                n_epochs=n_epochs_per_segment,
                x_interior=x_interior,
                verbose=False,
                **kwargs
            )

        return self.history
