"""
Utilities Package

Contains utilities for metrics computation and visualization.
"""

from .metrics import (
    l2_relative_error,
    max_absolute_error,
    mean_absolute_error,
    root_mean_square_error,
    coefficient_of_determination,
    compute_all_metrics,
    compute_convergence_rate,
    analyze_training_convergence,
    check_boundary_conditions,
    check_conservation,
    compute_spectral_accuracy
)

from .plotting import (
    setup_plot_style,
    plot_convergence_history,
    plot_solution_comparison_1d,
    plot_solution_comparison_2d,
    plot_multiple_convergence,
    plot_performance_heatmap,
    plot_boxplot_comparison,
    plot_accuracy_vs_time,
    plot_error_heatmap_spatial
)

__all__ = [
    # Metrics
    'l2_relative_error',
    'max_absolute_error',
    'mean_absolute_error',
    'root_mean_square_error',
    'coefficient_of_determination',
    'compute_all_metrics',
    'compute_convergence_rate',
    'analyze_training_convergence',
    'check_boundary_conditions',
    'check_conservation',
    'compute_spectral_accuracy',
    # Plotting
    'setup_plot_style',
    'plot_convergence_history',
    'plot_solution_comparison_1d',
    'plot_solution_comparison_2d',
    'plot_multiple_convergence',
    'plot_performance_heatmap',
    'plot_boxplot_comparison',
    'plot_accuracy_vs_time',
    'plot_error_heatmap_spatial'
]
