# fmri_features/connectome_generator.py
"""
Calcula diversas métricas de conectividad funcional (canales).
Versión robusta, compatible con diferentes TR y longitudes de señal.
"""
from __future__ import annotations
import numpy as np
import logging
from sklearn.covariance import GraphicalLassoCV
from nilearn.connectome import ConnectivityMeasure
from typing import Dict, Optional
from numpy.linalg import LinAlgError

# Carga opcional de librerías avanzadas
try:
    from dyconnmap.graphs import threshold_omst_global_cost_efficiency
    OMST_AVAILABLE = True
except ImportError:
    OMST_AVAILABLE = False
try:
    from mne_connectivity.spectral import spectral_connectivity_time
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

log = logging.getLogger(__name__)

def _fisher_r_to_z(matrix: np.ndarray) -> np.ndarray:
    """Transformada de Fisher r-a-z para estabilizar la varianza."""
    return np.arctanh(np.clip(matrix, -0.99999, 0.99999))

# --- Funciones de Cálculo de Canales ---

def pearson_full(ts: np.ndarray, **kwargs) -> np.ndarray:
    """Canal 1: Correlación de Pearson con transformada de Fisher."""
    corr_matrix = np.corrcoef(ts, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix)
    return _fisher_r_to_z(corr_matrix)

def pearson_omst(ts: np.ndarray, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal 2: Grafo de expansión mínima ortogonal (OMST) sobre Pearson."""
    if not OMST_AVAILABLE:
        log.warning(f"Sujeto {subject_id}: Librería 'dyconnmap' no encontrada. Omitiendo canal OMST.")
        return np.zeros((ts.shape[1], ts.shape[1]), dtype=np.float32)

    corr_matrix = np.corrcoef(ts, rowvar=False)
    z_corr = _fisher_r_to_z(np.nan_to_num(corr_matrix))
    
    if not np.any(z_corr):
        log.warning(f"Sujeto {subject_id}: Matriz de correlación es cero. Devolviendo matriz de ceros para OMST.")
        return np.zeros_like(z_corr)

    try:
        epsilon = 1e-9
        input_matrix = np.abs(z_corr) + epsilon
        omst_mask = threshold_omst_global_cost_efficiency(input_matrix)[1] > 0
        return (z_corr * omst_mask).astype(np.float32)
    except (ValueError, IndexError) as e:
        log.error(f"Fallo en dyconnmap para sujeto {subject_id}: {e}. Devolviendo None.")
        return None

def graphical_lasso(ts: np.ndarray, cfg: Dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal 3: Correlación parcial (Precisión) estimada con GraphicalLassoCV."""
    try:
        ts_cleaned = np.nan_to_num(ts)
        if np.any(np.std(ts_cleaned, axis=0) < 1e-9):
            log.warning(f"Sujeto {subject_id}: ROI con varianza cero detectada. GraphicalLasso puede ser inestable.")
        
        params = cfg.get('parameters', {}).get('graphical_lasso', {'cv_folds': 5})
        estimator = GraphicalLassoCV(cv=params['cv_folds'], n_jobs=1, verbose=False).fit(ts_cleaned)
        log.info(f"GraphicalLassoCV para sujeto {subject_id} finalizado. Alpha seleccionado: {estimator.alpha_:.4f}")
        return estimator.precision_.astype(np.float32)
    except LinAlgError:
        log.error(f"Fallo GraphicalLassoCV para sujeto {subject_id}: Matriz singular. Se devolverá None.")
        return None
    except Exception as e:
        log.error(f"Fallo inesperado en GraphicalLassoCV para sujeto {subject_id}: {e}", exc_info=True)
        return None

def tangent_space(ts: np.ndarray, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal 4: Conectividad en Espacio de Covarianza (base para Espacio Tangente)."""
    try:
        conn_measure = ConnectivityMeasure(kind='covariance', vectorize=False)
        covariance_matrix = conn_measure.fit_transform([ts])[0]
        return covariance_matrix.astype(np.float32)
    except Exception as e:
        log.error(f"Fallo en cálculo de Covariance para sujeto {subject_id}: {e}", exc_info=True)
        return None


def wavelet_coherence(ts: np.ndarray, cfg: Dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal 5: Coherencia media por Wavelets, optimizada para señales cortas."""
    if not MNE_AVAILABLE:
        log.warning(f"Sujeto {subject_id}: Librería 'mne_connectivity' no encontrada. Omitiendo canal Wavelet.")
        return np.zeros((ts.shape[1], ts.shape[1]), dtype=np.float32)

    try:
        pp_cfg = cfg['preprocessing']
        spec_cfg = cfg.get('parameters', {}).get('wavelet', {})
        
        # --- CAMBIO CLAVE: AJUSTAR n_cycles ---
        # Un valor más bajo (entre 3 y 5) es más adecuado para fMRI con señales cortas.
        # Esto hace la wavelet más corta en tiempo, evitando el error.
        n_cycles = spec_cfg.get('cwt_n_cycles', 4) # Usamos 4 como un default seguro y balanceado
        num_freqs = spec_cfg.get('num_freqs', 20)
        
        sfreq = 1.0 / pp_cfg.get('tr_seconds', 3.0)
        
        # Mantenemos tu rango de frecuencia original, ya que ahora n_cycles lo permite
        low_freq = pp_cfg['low_cut_hz']
        high_freq = pp_cfg['high_cut_hz']
        
        freqs = np.linspace(low_freq, high_freq, num=num_freqs)
        ts_mne = ts.T[np.newaxis, :, :] 
        
        log.debug(f"Sujeto {subject_id}: Calculando Wavelet Coherence con n_cycles={n_cycles} para evitar error de longitud.")

        # Intentamos el cálculo con los parámetros optimizados
        con = spectral_connectivity_time(
            ts_mne, freqs=freqs, method='coh', mode='cwt_morlet',
            sfreq=sfreq, n_cycles=n_cycles, n_jobs=1, verbose=False
        )
        mean_coh = con.get_data(output='dense').mean(axis=(2, 3))
        return mean_coh.astype(np.float32)
        
    except ValueError as e:
        # Este bloque ahora es un seguro, pero es menos probable que se active
        if "longer than the signal" in str(e):
             log.error(f"Sujeto {subject_id}: Fallo de Wavelet incluso con n_cycles={n_cycles}. La señal puede ser extremadamente corta. {e}")
        else:
            log.error(f"Sujeto {subject_id}: Fallo inesperado (ValueError) en Wavelet Coherence. {e}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo inesperado en Wavelet Coherence. {e}", exc_info=True)
        return None

CONNECTIVITY_METHODS = {
    'pearson_full': pearson_full,
    'pearson_omst': pearson_omst,
    'graphical_lasso': graphical_lasso,
    'tangent_space': tangent_space,
    'wavelet_coherence': wavelet_coherence,
}

def generate_connectivity_tensor(ts_data: np.ndarray, cfg: Dict, subject_id: str) -> Optional[np.ndarray]:
    """Genera un tensor 3D (canales x ROIs x ROIs) para un sujeto."""
    log.info(f"Sujeto {subject_id}: Generando tensor de conectividad.")
    matrices = []
    
    # Asegurarse de que los canales a procesar estén definidos
    channels_to_process = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    if not channels_to_process:
        log.error(f"Sujeto {subject_id}: No hay canales habilitados en la configuración. Abortando.")
        return None

    for channel_name in channels_to_process:
        if channel_name in CONNECTIVITY_METHODS:
            log.debug(f"Sujeto {subject_id}: Calculando canal '{channel_name}'...")
            func = CONNECTIVITY_METHODS[channel_name]
            matrix = func(ts=ts_data, cfg=cfg, subject_id=subject_id)
            
            if matrix is not None:
                matrices.append(np.nan_to_num(matrix))
            else:
                log.error(f"Sujeto {subject_id}: El canal '{channel_name}' no se pudo calcular. Abortando tensor para este sujeto.")
                return None
        else:
            log.warning(f"No se encontró una función para el canal '{channel_name}' en la configuración.")

    if not matrices:
        log.error(f"Sujeto {subject_id}: No se generó ninguna matriz. El tensor estaría vacío.")
        return None
        
    return np.stack(matrices, axis=0).astype(np.float32)