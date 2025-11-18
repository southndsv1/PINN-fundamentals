"""
Metrics for evaluating PINN performance.
"""

import numpy as np
from typing import Dict, Tuple


def l2_relative_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """
    Compute L2 relative error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        L2 relative error
    """
    numerator = np.linalg.norm(u_pred - u_exact)
    denominator = np.linalg.norm(u_exact)

    if denominator < 1e-12:
        return numerator

    return numerator / denominator


def max_absolute_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """
    Compute maximum absolute error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Maximum absolute error
    """
    return np.max(np.abs(u_pred - u_exact))


def mean_absolute_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Mean absolute error
    """
    return np.mean(np.abs(u_pred - u_exact))


def root_mean_square_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """
    Compute root mean square error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        RMSE
    """
    return np.sqrt(np.mean((u_pred - u_exact)**2))


def coefficient_of_determination(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """
    Compute R² score.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        R² score
    """
    ss_res = np.sum((u_exact - u_pred)**2)
    ss_tot = np.sum((u_exact - np.mean(u_exact))**2)

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0

    return 1.0 - ss_res / ss_tot


def compute_all_metrics(u_pred: np.ndarray, u_exact: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'l2_relative_error': l2_relative_error(u_pred, u_exact),
        'max_absolute_error': max_absolute_error(u_pred, u_exact),
        'mean_absolute_error': mean_absolute_error(u_pred, u_exact),
        'rmse': root_mean_square_error(u_pred, u_exact),
        'r2_score': coefficient_of_determination(u_pred, u_exact)
    }

    return metrics


def compute_convergence_rate(errors: np.ndarray, epochs: np.ndarray) -> float:
    """
    Compute convergence rate from error history.

    Args:
        errors: Array of errors over epochs
        epochs: Array of epoch numbers

    Returns:
        Convergence rate (negative slope in log space)
    """
    if len(errors) < 2:
        return 0.0

    # Filter out zero/negative errors
    valid_mask = errors > 1e-12
    if np.sum(valid_mask) < 2:
        return 0.0

    errors_valid = errors[valid_mask]
    epochs_valid = epochs[valid_mask]

    # Fit line in log space
    log_errors = np.log10(errors_valid)
    coeffs = np.polyfit(epochs_valid, log_errors, 1)

    return -coeffs[0]  # Negative slope (positive for convergence)


def analyze_training_convergence(history: Dict) -> Dict[str, float]:
    """
    Analyze training convergence from history.

    Args:
        history: Training history dictionary

    Returns:
        Convergence analysis metrics
    """
    total_loss = np.array(history['total_loss'])
    physics_loss = np.array(history['physics_loss'])
    epochs = np.array(history['epochs'])

    analysis = {
        'final_total_loss': total_loss[-1] if len(total_loss) > 0 else float('inf'),
        'final_physics_loss': physics_loss[-1] if len(physics_loss) > 0 else float('inf'),
        'min_total_loss': np.min(total_loss) if len(total_loss) > 0 else float('inf'),
        'convergence_rate': compute_convergence_rate(total_loss, epochs),
        'epochs_to_convergence': len(epochs),
        'converged': total_loss[-1] < 1e-3 if len(total_loss) > 0 else False
    }

    return analysis


def check_boundary_conditions(
    u_pred: np.ndarray,
    u_bc: np.ndarray,
    tolerance: float = 1e-3
) -> Tuple[bool, float]:
    """
    Check if boundary conditions are satisfied.

    Args:
        u_pred: Predicted values at boundary
        u_bc: Expected boundary values
        tolerance: Tolerance for satisfaction

    Returns:
        (satisfied, max_violation)
    """
    violation = np.max(np.abs(u_pred - u_bc))
    satisfied = violation < tolerance

    return satisfied, violation


def check_conservation(
    u: np.ndarray,
    initial_integral: float,
    tolerance: float = 0.01
) -> Tuple[bool, float]:
    """
    Check conservation property (e.g., mass conservation).

    Args:
        u: Current solution field
        initial_integral: Initial integral value
        tolerance: Relative tolerance

    Returns:
        (conserved, relative_change)
    """
    current_integral = np.sum(u)
    relative_change = np.abs(current_integral - initial_integral) / (np.abs(initial_integral) + 1e-12)
    conserved = relative_change < tolerance

    return conserved, relative_change


def compute_spectral_accuracy(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
    """
    Compute spectral accuracy using Fourier analysis.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Spectral accuracy metric
    """
    # Compute FFT
    fft_pred = np.fft.fft(u_pred.flatten())
    fft_exact = np.fft.fft(u_exact.flatten())

    # Compute spectral error
    spectral_error = np.linalg.norm(fft_pred - fft_exact) / (np.linalg.norm(fft_exact) + 1e-12)

    return spectral_error
