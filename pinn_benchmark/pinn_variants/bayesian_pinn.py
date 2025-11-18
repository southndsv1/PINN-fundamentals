"""
Bayesian PINN: Adds uncertainty quantification using dropout or variational inference.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable, List
import numpy as np
from .base_pinn import BasePINN, MLP


class BayesianMLP(nn.Module):
    """MLP with dropout for uncertainty quantification."""

    def __init__(self, layers: List[int], activation=nn.Tanh(), dropout_rate: float = 0.1):
        super(BayesianMLP, self).__init__()
        self.layers = layers
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Build network with dropout
        self.network = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.network.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # No dropout on output layer
                self.dropouts.append(nn.Dropout(p=dropout_rate))

        # Xavier initialization
        self.init_weights()

    def init_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout."""
        for i, layer in enumerate(self.network[:-1]):
            x = self.activation(layer(x))
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        x = self.network[-1](x)
        return x


class BayesianPINN(BasePINN):
    """
    Bayesian Physics-Informed Neural Network.

    Uses dropout for uncertainty quantification.
    Provides mean predictions with confidence intervals.
    """

    def __init__(
        self,
        pde_residual_fn: Callable,
        dropout_rate: float = 0.1,
        n_samples: int = 100,
        **kwargs
    ):
        """
        Initialize Bayesian PINN.

        Args:
            pde_residual_fn: Function that computes PDE residual
            dropout_rate: Dropout probability
            n_samples: Number of samples for uncertainty estimation
            **kwargs: Additional arguments for BasePINN
        """
        # Initialize base class without building network
        self.pde_residual_fn = pde_residual_fn
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples

        # Call parent init but override network
        super().__init__(**kwargs)

        # Replace network with Bayesian version
        self.network = BayesianMLP(
            self.layers,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Recreate optimizer for new network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, verbose=False
        )

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
        Compute total loss for Bayesian PINN.

        Returns:
            Total loss and dictionary of loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # Physics loss (PDE residual) with dropout active
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

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Args:
            x: Input points

        Returns:
            Tuple of (mean predictions, standard deviation)
        """
        self.network.train()  # Keep dropout active for uncertainty estimation

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                u_pred = self.network(x)
                predictions.append(u_pred.cpu().numpy())

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        self.network.eval()

        return mean, std

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions (mean only).

        Args:
            x: Input points

        Returns:
            Mean predictions
        """
        mean, _ = self.predict_with_uncertainty(x)
        return mean
