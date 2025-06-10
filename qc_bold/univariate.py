# qc_bold/univariate.py
"""
Funciones para la detección de outliers univariantes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import zscore, median_abs_deviation
import warnings

def robust_zscore(data: np.ndarray) -> np.ndarray:
    """
    Calcula el Z-score robusto usando la mediana y la Desviación Absoluta Mediana (MAD).
    El factor de escala 1.4826 se usa para hacer la MAD comparable a la desviación estándar
    para datos normalmente distribuidos.

    Args:
        data (np.ndarray): Matriz de datos (timepoints x rois).

    Returns:
        np.ndarray: Matriz de Z-scores robustos.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(data, axis=0)
        # MAD se calcula con nan_policy="omit" para ignorar NaNs
        mad = median_abs_deviation(data, axis=0, nan_policy="omit") * 1.4826
    
    # Prevenir división por cero si una columna es constante
    mad[mad == 0] = 1.0 
    
    return (data - median) / mad

def detect_univariate_outliers(
    bold_data: np.ndarray,
    z_threshold: float = 3.5,
    method: str = 'robust'
) -> tuple[np.ndarray, float]:
    """
    Detecta outliers univariantes en una matriz BOLD.

    Args:
        bold_data (np.ndarray): Matriz de datos (timepoints x rois).
        z_threshold (float): Umbral de Z-score para definir un outlier.
        method (str): 'robust' para usar mediana/MAD, 'standard' para media/std.

    Returns:
        tuple[np.ndarray, float]: 
            - máscara booleana (timepoints x rois) de outliers.
            - porcentaje de outliers detectados.
    """
    if bold_data.size == 0:
        return np.array([[]]), 0.0

    if method == 'robust':
        z_scores = robust_zscore(bold_data)
    else: # method == 'standard'
        z_scores = zscore(bold_data, axis=0, nan_policy='omit')
    
    outlier_mask = np.abs(z_scores) > z_threshold
    
    num_valid_points = np.count_nonzero(~np.isnan(bold_data))
    if num_valid_points == 0:
        return outlier_mask, 0.0
        
    outlier_count = np.nansum(outlier_mask)
    outlier_pct = 100 * outlier_count / num_valid_points
    
    return outlier_mask, outlier_pct