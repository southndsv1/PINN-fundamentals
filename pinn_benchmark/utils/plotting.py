"""
Plotting utilities for PINN benchmarking.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os


def setup_plot_style():
    """Setup consistent plot style."""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_convergence_history(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Training Convergence"
):
    """
    Plot training convergence history.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = np.array(history['epochs'])

    # Total loss
    axes[0, 0].semilogy(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss (log scale)')
    axes[0, 0].set_title('Total Loss Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Physics vs Data loss
    axes[0, 1].semilogy(epochs, history['physics_loss'], 'r-', linewidth=2, label='Physics Loss')
    if 'data_loss' in history and max(history['data_loss']) > 0:
        axes[0, 1].semilogy(epochs, history['data_loss'], 'g-', linewidth=2, label='Data Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Physics vs Data Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Boundary and Initial losses
    if 'boundary_loss' in history:
        axes[1, 0].semilogy(epochs, history['boundary_loss'], 'm-', linewidth=2, label='Boundary Loss')
    if 'initial_loss' in history:
        axes[1, 0].semilogy(epochs, history['initial_loss'], 'c-', linewidth=2, label='Initial Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Boundary & Initial Condition Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Learning rate
    if 'learning_rates' in history:
        axes[1, 1].plot(epochs, history['learning_rates'], 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_solution_comparison_1d(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_exact: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Solution Comparison"
):
    """Plot 1D solution comparison."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Solution comparison
    axes[0].plot(x, u_exact, 'b-', linewidth=2, label='Exact', alpha=0.7)
    axes[0].plot(x, u_pred, 'r--', linewidth=2, label='Predicted')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].set_title('Solution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Point-wise error
    error = np.abs(u_pred - u_exact)
    axes[1].plot(x, error, 'g-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title('Absolute Error')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_solution_comparison_2d(
    X: np.ndarray,
    Y: np.ndarray,
    U_pred: np.ndarray,
    U_exact: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "2D Solution Comparison"
):
    """Plot 2D solution comparison."""
    setup_plot_style()

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig)

    # Exact solution
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, U_exact, levels=50, cmap='viridis')
    ax1.set_title('Exact Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)

    # Predicted solution
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, Y, U_pred, levels=50, cmap='viridis')
    ax2.set_title('Predicted Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)

    # Error
    ax3 = fig.add_subplot(gs[0, 2])
    error = np.abs(U_pred - U_exact)
    im3 = ax3.contourf(X, Y, error, levels=50, cmap='hot')
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multiple_convergence(
    histories: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Convergence Comparison"
):
    """
    Plot convergence comparison for multiple PINN variants.

    Args:
        histories: Dictionary mapping variant name to history
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (variant_name, history), color in zip(histories.items(), colors):
        epochs = np.array(history['epochs'])
        total_loss = np.array(history['total_loss'])
        ax.semilogy(epochs, total_loss, linewidth=2, label=variant_name, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_performance_heatmap(
    results_matrix: np.ndarray,
    pinn_variants: List[str],
    problems: List[str],
    metric_name: str = "L2 Error",
    save_path: Optional[str] = None,
    title: str = "Performance Heatmap"
):
    """
    Plot heatmap of PINN performance across problems.

    Args:
        results_matrix: Matrix of shape (n_variants, n_problems) with performance metrics
        pinn_variants: List of PINN variant names
        problems: List of problem names
        metric_name: Name of metric being displayed
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Use log scale for better visualization
    results_log = np.log10(results_matrix + 1e-10)

    im = ax.imshow(results_log, cmap='RdYlGn_r', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(problems)))
    ax.set_yticks(np.arange(len(pinn_variants)))
    ax.set_xticklabels(problems, rotation=45, ha='right')
    ax.set_yticklabels(pinn_variants)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'log10({metric_name})', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(pinn_variants)):
        for j in range(len(problems)):
            text = ax.text(j, i, f'{results_matrix[i, j]:.2e}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Problem', fontsize=12)
    ax.set_ylabel('PINN Variant', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_boxplot_comparison(
    results_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Performance Comparison",
    ylabel: str = "L2 Relative Error"
):
    """
    Plot boxplot comparison of PINN variants.

    Args:
        results_dict: Dictionary mapping variant name to list of errors
        save_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    data = [results_dict[key] for key in results_dict.keys()]
    labels = list(results_dict.keys())

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_accuracy_vs_time(
    variants: List[str],
    accuracies: List[float],
    times: List[float],
    save_path: Optional[str] = None,
    title: str = "Accuracy vs Training Time"
):
    """
    Plot accuracy vs training time scatter plot.

    Args:
        variants: List of variant names
        accuracies: List of accuracy metrics (lower is better)
        times: List of training times
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))

    for i, (variant, acc, t, color) in enumerate(zip(variants, accuracies, times, colors)):
        ax.scatter(t, acc, s=200, alpha=0.6, color=color, label=variant, edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('L2 Relative Error', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_heatmap_spatial(
    X: np.ndarray,
    Y: np.ndarray,
    error: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Spatial Error Distribution"
):
    """
    Plot spatial error distribution as heatmap.

    Args:
        X: X coordinates
        Y: Y coordinates
        error: Error values
        save_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.contourf(X, Y, error, levels=50, cmap='hot')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Error', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
