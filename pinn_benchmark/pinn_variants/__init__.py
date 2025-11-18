"""
PINN Variants Package

Contains implementations of different PINN architectures:
- Vanilla PINN: Standard implementation
- Variational PINN: Weak form formulation
- Conservative PINN: Enforces conservation laws
- Bayesian PINN: Uncertainty quantification
- Gradient-Enhanced PINN: Uses derivative information
- Adaptive PINN: Dynamic weights and sampling
- Causal PINN: Temporal causality for time-dependent problems
"""

from .base_pinn import BasePINN, MLP
from .vanilla_pinn import VanillaPINN
from .variational_pinn import VariationalPINN
from .conservative_pinn import ConservativePINN
from .bayesian_pinn import BayesianPINN
from .gradient_enhanced_pinn import GradientEnhancedPINN
from .adaptive_pinn import AdaptivePINN
from .causal_pinn import CausalPINN

__all__ = [
    'BasePINN',
    'MLP',
    'VanillaPINN',
    'VariationalPINN',
    'ConservativePINN',
    'BayesianPINN',
    'GradientEnhancedPINN',
    'AdaptivePINN',
    'CausalPINN'
]
