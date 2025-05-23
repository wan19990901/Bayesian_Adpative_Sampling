"""
BEACON (Best Early-stopping And COmparison of N-samples) Module

This module contains utilities for dynamic stopping and sample comparison, including:
- Early stopping analysis
- Sampling comparison
- Dynamic response selection
- Performance evaluation
"""

from .stopping_analysis import analyze_stopping
from .sampling_comparison import compare_samples

__all__ = [
    'analyze_stopping',
    'compare_samples'
] 