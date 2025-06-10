# qc_bold/multivariate.py
"""
Funciones para la detección de outliers multivariantes.
"""
from __future__ import annotations
import numpy as np
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn.impute import SimpleImputer
from scipy.stats import chi2, zscore
import warnings
from numpy.linalg import LinAlgError
import logging

log = logging.getLogger(__name__)

def mahalanobis_pvals(X: np.ndarray, cov_estimator: MinCovDet | LedoitWolf) -> np.ndarray:
    """Calcula los p-valores de las distancias de Mahalanobis."""
    n_features = X.shape[1]
    if n_features == 0:
        return np.array([])

    if isinstance(cov_estimator, MinCovDet):
        # La distancia de Mahalanobis ya está calculada en el objeto MCD
        d2 = cov_estimator.mahalanobis(X)
    elif isinstance(cov_estimator, LedoitWolf):
        # Se calcula manualmente para Ledoit-Wolf
        diff = X - cov_estimator.location_
        try:
            # Usar pseudo-inversa para mayor estabilidad
            inv_cov = np.linalg.pinv(cov_estimator.covariance_)
            d2 = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        except LinAlgError as e:
            log.error(f"Error de álgebra lineal al calcular Mahalanobis con Ledoit-Wolf: {e}")
            return np.full(X.shape[0], np.nan)
    else:
        raise TypeError("El estimador de covarianza no es soportado.")

    # Los p-valores se derivan de la distribución Chi-cuadrado
    p_values = 1 - chi2.cdf(d2, df=n_features)
    return p_values


def detect_multivariate_outliers(
    bold_data: np.ndarray,
    config: dict
) -> dict:
    """
    Pipeline para detectar outliers multivariantes.
    Imputa NaNs, estandariza, y luego aplica MCD o Ledoit-Wolf.
    """
    mv_cfg = config['outliers']['multivariate']
    results = {
        'mv_outliers_count': np.nan,
        'mv_outliers_pct': np.nan,
        'mv_analysis_method': None,
        'mv_skipped_reason': None
    }
    
    n_samples, n_features = bold_data.shape

    # 1. Chequeos preliminares
    if n_features == 0:
        results['mv_skipped_reason'] = "No_ROIs_left_for_analysis"
        return results
    if n_samples < mv_cfg['min_timepoints_for_any_mv']:
        results['mv_skipped_reason'] = f"Insufficient_TPs(is_{n_samples})_for_any_MV(min_{mv_cfg['min_timepoints_for_any_mv']})"
        return results

    # 2. Preprocesamiento: Imputar y Estandarizar
    data_imputed = SimpleImputer(strategy='median').fit_transform(bold_data)
    data_zscored = zscore(data_imputed, axis=0)

    # 3. Selección del estimador de covarianza
    can_use_mcd = (n_samples > n_features * mv_cfg['min_tp_for_mcd_strict_factor'])
    can_use_lw = (n_samples >= mv_cfg['min_tp_for_ledoitwolf'])
    
    estimator = None
    estimator_name = None

    if mv_cfg['covariance_estimator'] == 'MCD' and can_use_mcd:
        estimator_name = "MCD"
    elif mv_cfg['covariance_estimator'] == 'LedoitWolf' and can_use_lw:
        estimator_name = "LedoitWolf"
    elif mv_cfg['covariance_estimator'] == 'auto':
        if can_use_mcd:
            estimator_name = "MCD"
        elif can_use_lw:
            estimator_name = "LedoitWolf"
            log.info(f"TPs ({n_samples}) insuficientes para MCD estricto. Usando Ledoit-Wolf.")

    results['mv_analysis_method'] = estimator_name

    # 4. Ajustar el estimador y calcular outliers
    if estimator_name == "MCD":
        estimator = MinCovDet(support_fraction=None, random_state=42).fit(data_zscored)
    elif estimator_name == "LedoitWolf":
        estimator = LedoitWolf().fit(data_zscored)
    else:
        results['mv_skipped_reason'] = f"Insufficient_TPs_for_MV(TPs={n_samples},ROIs={n_features})"
        return results
        
    try:
        p_values = mahalanobis_pvals(data_zscored, estimator)
        
        mv_outliers_count = np.sum(p_values < mv_cfg['alpha_mahalanobis'])
        mv_outliers_pct = 100 * mv_outliers_count / n_samples
        
        results.update({
            'mv_outliers_count': mv_outliers_count,
            'mv_outliers_pct': mv_outliers_pct,
        })
    except Exception as e:
        log.error(f"Fallo el cálculo de outliers para {estimator_name}: {e}")
        results['mv_skipped_reason'] = f"{estimator_name}_Error"
        results['mv_analysis_method'] = f"{estimator_name}_Failed"

    return results