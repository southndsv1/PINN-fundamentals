"""
Benchmark Problems Package

Contains 10 classic mechanical engineering PDE problems:
- Heat Conduction (1D parabolic)
- Wave Propagation (1D hyperbolic)
- Burgers Equation (1D nonlinear advection-diffusion)
- Navier-Stokes (2D fluid dynamics)
- Euler-Bernoulli Beam (1D structural mechanics)
- Plate Vibration (2D/3D structural dynamics)
- Linear Elasticity (2D solid mechanics)
- Advection-Diffusion (1D transport)
- Allen-Cahn (1D phase field)
- Kuramoto-Sivashinsky (1D chaotic dynamics)
"""

from .heat_conduction import HeatConduction1D
from .wave_propagation import WavePropagation1D
from .burgers_equation import BurgersEquation
from .navier_stokes import NavierStokes2D
from .euler_bernoulli_beam import EulerBernoulliBeam
from .plate_vibration import PlateVibration
from .elasticity import LinearElasticity2D
from .advection_diffusion import AdvectionDiffusion1D
from .allen_cahn import AllenCahn1D
from .kuramoto_sivashinsky import KuramotoSivashinsky1D

__all__ = [
    'HeatConduction1D',
    'WavePropagation1D',
    'BurgersEquation',
    'NavierStokes2D',
    'EulerBernoulliBeam',
    'PlateVibration',
    'LinearElasticity2D',
    'AdvectionDiffusion1D',
    'AllenCahn1D',
    'KuramotoSivashinsky1D'
]
