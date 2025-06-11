# fmri_features/connectome_generator.py
"""
Calcula un conjunto extendido de métricas de conectividad funcional.
Versión robusta que incluye:
- Conectividad estática (Pearson, Parcial)
- Conectividad dinámica (Sliding Window STD)
- Conectividad no lineal (Mutual Information, Distance Correlation)
- Conectividad dirigida (Granger)
- Conectividad en frecuencia (Wavelet por bandas)
"""
from __future__ import annotations
import numpy as np
import logging
from sklearn.covariance import GraphicalLassoCV
from nilearn.connectome import ConnectivityMeasure
from typing import Dict, Optional, Tuple
from numpy.linalg import LinAlgError
from functools import partial

# --- Carga de nuevas librerías (asegúrate de instalarlas) ---
# pip install scikit-learn dcor statsmodels mne_connectivity dyconnmap
try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
try:
    import dcor
    DCOR_AVAILABLE = True
except ImportError:
    DCOR_AVAILABLE = False
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# --- Carga de librerías existentes ---
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

# --- Funciones de Cálculo de Canales (Existentes y Nuevas) ---

# --- CANALES BASE ---
def pearson_full(ts: np.ndarray, **kwargs) -> np.ndarray:
    corr_matrix = np.corrcoef(ts, rowvar=False)
    return _fisher_r_to_z(np.nan_to_num(corr_matrix))

def pearson_omst(ts: np.ndarray, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    if not OMST_AVAILABLE: return np.zeros((ts.shape[1], ts.shape[1]))
    corr_matrix = np.corrcoef(ts, rowvar=False)
    z_corr = _fisher_r_to_z(np.nan_to_num(corr_matrix))
    if not np.any(z_corr): return np.zeros_like(z_corr)
    try:
        omst_mask = threshold_omst_global_cost_efficiency(np.abs(z_corr) + 1e-9)[1] > 0
        return (z_corr * omst_mask).astype(np.float32)
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo OMST - {e}")
        return None

def graphical_lasso(ts: np.ndarray, cfg: Dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    try:
        params = cfg.get('parameters', {}).get('graphical_lasso', {'cv_folds': 3})
        estimator = GraphicalLassoCV(cv=params['cv_folds'], n_jobs=1).fit(np.nan_to_num(ts))
        return estimator.precision_.astype(np.float32)
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo GraphicalLasso - {e}")
        return None

def tangent_space(ts: np.ndarray, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    try:
        return ConnectivityMeasure(kind='covariance').fit_transform([ts])[0]
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo TangentSpace - {e}")
        return None
    
# --- NUEVOS CANALES ---

def mutual_info_full(ts: np.ndarray, cfg: dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal de Información Mutua (simetrizada)."""
    if not SKLEARN_AVAILABLE: return np.zeros((ts.shape[1], ts.shape[1]))
    log.debug(f"Sujeto {subject_id}: Calculando Información Mutua...")
    try:
        params = cfg.get('parameters', {}).get('mutual_info', {'n_bins': 8})
        ranks = np.argsort(np.argsort(ts, axis=0), axis=0)
        disc = np.floor((ranks / float(ts.shape[0])) * params['n_bins']).astype(int)
        n_rois = ts.shape[1]
        mi_mat = np.zeros((n_rois, n_rois), dtype=np.float32)
        for i in range(n_rois):
            mi_mat[i, :] = mutual_info_regression(disc, disc[:, i], discrete_features=True)
        return (mi_mat + mi_mat.T) / 2.0
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo Mutual Information - {e}")
        return None

def distance_corr_full(ts: np.ndarray, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal de Distance Correlation."""
    if not DCOR_AVAILABLE: return np.zeros((ts.shape[1], ts.shape[1]))
    log.debug(f"Sujeto {subject_id}: Calculando Distance Correlation...")
    try:
        return dcor.distance_correlation_matrix(ts).astype(np.float32)
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo Distance Correlation - {e}")
        return None

def granger_pairwise(ts: np.ndarray, cfg: dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal de Causalidad de Granger (no simétrica)."""
    if not STATSMODELS_AVAILABLE: return np.zeros((ts.shape[1], ts.shape[1]))
    log.debug(f"Sujeto {subject_id}: Calculando Causalidad de Granger...")
    try:
        n = ts.shape[1]
        maxlag = cfg.get('parameters', {}).get('granger', {'max_lag': 1})['max_lag']
        g_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                try:
                    res = grangercausalitytests(ts[:, [j, i]], maxlag=maxlag, verbose=False)
                    g_mat[i, j] = res[maxlag][0]['ssr_ftest'][0]
                except:
                    g_mat[i, j] = 0.0
        return g_mat
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo Granger - {e}")
        return None

def sliding_window_variability(ts: np.ndarray, cfg: dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal de variabilidad de dFC (Std Dev en ventanas)."""
    log.debug(f"Sujeto {subject_id}: Calculando dFC (Std Dev)...")
    try:
        params = cfg.get('parameters', {}).get('sliding_window', {'win_len': 30, 'step': 5})
        win_len, step = params['win_len'], params['step']
        n_tp, n_roi = ts.shape
        starts = range(0, n_tp - win_len + 1, step)
        if len(starts) < 2: return np.zeros((n_roi, n_roi))
        
        window_corrs = [np.nan_to_num(np.corrcoef(ts[s:s+win_len], rowvar=False)) for s in starts]
        return np.std(np.stack(window_corrs, axis=0), axis=0).astype(np.float32)
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo Sliding Window - {e}")
        return None

def wavelet_band(ts: np.ndarray, cfg: Dict, subject_id: str, band: Tuple[float, float], **kwargs) -> Optional[np.ndarray]:
    """Función genérica para calcular coherencia por wavelet en una banda específica."""
    if not MNE_AVAILABLE: return np.zeros((ts.shape[1], ts.shape[1]))
    low, high = band
    log.debug(f"Sujeto {subject_id}: Calculando Wavelet Coherence para banda [{low}, {high}] Hz...")
    try:
        sfreq = 1.0 / cfg['preprocessing'].get('tr_seconds', 3.0)
        params = cfg.get('parameters', {}).get('wavelet', {})
        freqs = np.linspace(low, high, params.get('num_freqs_per_band', 10))
        n_cycles = params.get('cwt_n_cycles', 4)

        ts_mne = ts.T[np.newaxis, :, :]
        con = spectral_connectivity_time(
            ts_mne, freqs=freqs, method='coh', mode='cwt_morlet',
            sfreq=sfreq, n_cycles=n_cycles, n_jobs=1, verbose=False
        )
        return con.get_data(output='dense').mean(axis=(2, 3)).astype(np.float32)
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo Wavelet en banda {band} - {e}")
        return None


def wavelet_coherence(ts: np.ndarray, cfg: Dict, subject_id: str, **kwargs) -> Optional[np.ndarray]:
    """Canal 5: Coherencia media por Wavelets, optimizada para señales cortas."""
    if not MNE_AVAILABLE:
        log.warning(f"Sujeto {subject_id}: Librería 'mne_connectivity' no encontrada. Omitiendo canal Wavelet.")
        return np.zeros((ts.shape[1], ts.shape[1]), dtype=np.float32)

    try:
        pp_cfg = cfg['preprocessing']
        spec_cfg = cfg.get('parameters', {}).get('wavelet', {})
        
        # --- AJUSTE CLAVE PARA SEÑALES CORTAS ---
        n_cycles = spec_cfg.get('cwt_n_cycles', 4) # Un valor bajo (3-5) es ideal
        num_freqs = spec_cfg.get('num_freqs', 20)
        
        sfreq = 1.0 / pp_cfg.get('tr_seconds', 3.0)
        
        low_freq = pp_cfg['low_cut_hz']
        high_freq = pp_cfg['high_cut_hz']
        
        freqs = np.linspace(low_freq, high_freq, num=num_freqs)
        ts_mne = ts.T[np.newaxis, :, :] 
        
        log.debug(f"Sujeto {subject_id}: Calculando Wavelet Coherence con n_cycles={n_cycles}.")

        con = spectral_connectivity_time(
            ts_mne, freqs=freqs, method='coh', mode='cwt_morlet',
            sfreq=sfreq, n_cycles=n_cycles, n_jobs=1, verbose=False
        )
        mean_coh = con.get_data(output='dense').mean(axis=(2, 3))
        return mean_coh.astype(np.float32)
        
    except ValueError as e:
        if "longer than the signal" in str(e):
             log.error(f"Sujeto {subject_id}: Fallo de Wavelet incluso con n_cycles={n_cycles}. La señal puede ser extremadamente corta. {e}")
        else:
            log.error(f"Sujeto {subject_id}: Fallo inesperado (ValueError) en Wavelet Coherence. {e}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo inesperado en Wavelet Coherence. {e}", exc_info=True)
        return None

# --- Diccionario final de Métodos de Conectividad ---
CONNECTIVITY_METHODS = {
    # Originales
    'pearson_full': pearson_full,
    'pearson_omst': pearson_omst,
    'graphical_lasso': graphical_lasso,
    'tangent_space': tangent_space,
    # Nuevos
    'mutual_info_full': mutual_info_full,
    'distance_corr_full': distance_corr_full,
    'granger_pairwise': granger_pairwise,
    'sliding_std_corr': sliding_window_variability,
    # Nuevos por banda de frecuencia
    'wavelet_coh_slow': partial(wavelet_band, band=(0.015, 0.05)),
    'wavelet_coh_fast': partial(wavelet_band, band=(0.05, 0.1)),
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