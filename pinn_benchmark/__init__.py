"""
PINN Benchmark Framework

A comprehensive framework for benchmarking different Physics-Informed Neural Network
variants on mechanical engineering problems.
"""

__version__ = '1.0.0'
__author__ = 'PINN Benchmark Team'

from . import pinn_variants
from . import benchmark_problems
from . import utils

__all__ = ['pinn_variants', 'benchmark_problems', 'utils']
