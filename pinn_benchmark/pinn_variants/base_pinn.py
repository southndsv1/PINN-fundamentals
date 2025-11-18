"""
Base PINN class providing common functionality for all PINN variants.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
import time


class MLP(nn.Module):
    """Multi-layer perceptron for function approximation."""

    def __init__(self, layers: List[int], activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.layers = layers
        self.activation = activation

        # Build network
        self.network = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.network.append(nn.Linear(layers[i], layers[i+1]))

        # Xavier initialization
        self.init_weights()

    def init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for i, layer in enumerate(self.network[:-1]):
            x = self.activation(layer(x))
        x = self.network[-1](x)  # No activation on output layer
        return x


class BasePINN(ABC):
    """
    Base class for Physics-Informed Neural Networks.

    Provides common functionality including:
    - Network architecture setup
    - Training loop
    - Loss computation framework
    - Collocation point sampling
    - Checkpointing
    - Metrics tracking
    """

    def __init__(
        self,
        layers: List[int],
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        lambda_physics: float = 1.0,
        **kwargs
    ):
        """
        Initialize base PINN.

        Args:
            layers: List of layer sizes [input_dim, hidden1, ..., hiddenN, output_dim]
            learning_rate: Learning rate for optimizer
            device: Device to run on ('cpu' or 'cuda')
            lambda_physics: Weight for physics loss term
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lambda_physics = lambda_physics

        # Build network
        self.network = MLP(layers).to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, verbose=False
        )

        # Training history
        self.history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
            'epochs': [],
            'learning_rates': [],
            'wall_time': []
        }

        self.start_time = None

    def latin_hypercube_sampling(self, n_points: int, bounds: np.ndarray) -> torch.Tensor:
        """
        Generate collocation points using Latin Hypercube Sampling.

        Args:
            n_points: Number of points to generate
            bounds: Array of shape (n_dims, 2) with min/max for each dimension

        Returns:
            Tensor of shape (n_points, n_dims)
        """
        n_dims = len(bounds)
        points = np.zeros((n_points, n_dims))

        for i in range(n_dims):
            points[:, i] = np.random.uniform(
                bounds[i, 0], bounds[i, 1], n_points
            )
            # Apply LHS permutation
            intervals = np.linspace(bounds[i, 0], bounds[i, 1], n_points + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            points[:, i] = np.random.uniform(lower, upper)
            np.random.shuffle(points[:, i])

        return torch.tensor(points, dtype=torch.float32, device=self.device, requires_grad=True)

    def compute_gradient(self, u: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """
        Compute gradient using automatic differentiation.

        Args:
            u: Network output
            x: Input tensor
            order: Order of derivative

        Returns:
            Gradient tensor
        """
        grad = u
        for _ in range(order):
            grad = torch.autograd.grad(
                grad, x,
                grad_outputs=torch.ones_like(grad),
                create_graph=True,
                retain_graph=True
            )[0]
        return grad

    @abstractmethod
    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual at collocation points.
        Must be implemented by each PINN variant.

        Args:
            x: Collocation points

        Returns:
            PDE residual
        """
        pass

    @abstractmethod
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
        Compute total loss including physics and data terms.
        Must be implemented by each PINN variant.

        Args:
            x_interior: Interior collocation points
            x_boundary: Boundary points
            u_boundary: Boundary values
            x_initial: Initial condition points (for time-dependent)
            u_initial: Initial condition values
            x_data: Data points for supervised learning
            u_data: Data values

        Returns:
            Total loss and dictionary of loss components
        """
        pass

    def train_step(
        self,
        x_interior: torch.Tensor,
        x_boundary: Optional[torch.Tensor] = None,
        u_boundary: Optional[torch.Tensor] = None,
        x_initial: Optional[torch.Tensor] = None,
        u_initial: Optional[torch.Tensor] = None,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()

        loss, loss_dict = self.compute_loss(
            x_interior, x_boundary, u_boundary,
            x_initial, u_initial, x_data, u_data
        )

        loss.backward()
        self.optimizer.step()

        return loss_dict

    def train(
        self,
        n_epochs: int,
        x_interior: torch.Tensor,
        x_boundary: Optional[torch.Tensor] = None,
        u_boundary: Optional[torch.Tensor] = None,
        x_initial: Optional[torch.Tensor] = None,
        u_initial: Optional[torch.Tensor] = None,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        checkpoint_freq: int = 1000,
        early_stopping_patience: int = 2000,
        early_stopping_delta: float = 1e-6,
        verbose: bool = True
    ) -> Dict:
        """
        Train the PINN.

        Args:
            n_epochs: Number of training epochs
            x_interior: Interior collocation points
            x_boundary: Boundary points
            u_boundary: Boundary values
            x_initial: Initial condition points
            u_initial: Initial condition values
            x_data: Data points
            u_data: Data values
            checkpoint_freq: Frequency to save checkpoints
            early_stopping_patience: Epochs to wait for improvement
            early_stopping_delta: Minimum change to qualify as improvement
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        self.start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training step
            loss_dict = self.train_step(
                x_interior, x_boundary, u_boundary,
                x_initial, u_initial, x_data, u_data
            )

            # Update history
            total_loss = loss_dict.get('total_loss', 0.0)
            self.history['total_loss'].append(total_loss)
            self.history['data_loss'].append(loss_dict.get('data_loss', 0.0))
            self.history['physics_loss'].append(loss_dict.get('physics_loss', 0.0))
            self.history['boundary_loss'].append(loss_dict.get('boundary_loss', 0.0))
            self.history['initial_loss'].append(loss_dict.get('initial_loss', 0.0))
            self.history['epochs'].append(epoch)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['wall_time'].append(time.time() - self.start_time)

            # Learning rate scheduling
            self.scheduler.step(total_loss)

            # Early stopping
            if total_loss < best_loss - early_stopping_delta:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

            # Verbose output
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs} | Loss: {total_loss:.6e} | "
                      f"Physics: {loss_dict.get('physics_loss', 0.0):.6e} | "
                      f"Data: {loss_dict.get('data_loss', 0.0):.6e} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6e}")

        return self.history

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions at given points.

        Args:
            x: Input points

        Returns:
            Network predictions as numpy array
        """
        self.network.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            u_pred = self.network(x)
        self.network.train()
        return u_pred.cpu().numpy()

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'layers': self.layers
        }, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**2
        else:
            # Approximate CPU memory usage
            mem = sum(p.numel() * p.element_size() for p in self.network.parameters())
            return mem / 1024**2
