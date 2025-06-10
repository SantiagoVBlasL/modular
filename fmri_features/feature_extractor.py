# fmri_features/feature_extractor.py
"""
Extrae características derivadas (no matriciales) como dinámica HMM o topología de grafos.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

# Carga opcional de librerías
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
try:
    import bct
    BCT_AVAILABLE = True
except ImportError:
    BCT_AVAILABLE = False

log = logging.getLogger(__name__)

def extract_hmm_features(ts: np.ndarray, cfg: dict, subject_id: str) -> Dict[str, Any] | None:
    """Ajusta un HMM a las series temporales y extrae métricas de dinámica."""
    if not HMM_AVAILABLE:
        log.warning("Librería 'hmmlearn' no encontrada. Omitiendo características HMM.")
        return None
    
    log.debug(f"Sujeto {subject_id}: Extrayendo características HMM...")
    try:
        params = cfg['parameters']['hmm']
        model = hmm.GaussianHMM(
            n_components=params['n_states'],
            covariance_type=params['covariance_type'],
            n_iter=params['n_iter']
        )
        model.fit(ts)
        
        if not model.monitor_.converged:
            log.warning(f"Sujeto {subject_id}: El modelo HMM no convergió.")

        hidden_states = model.predict(ts)
        
        # Ocupación fraccional de cada estado
        frac_occupancy = np.bincount(hidden_states, minlength=params['n_states']) / len(hidden_states)
        
        # Transiciones entre estados
        n_transitions = np.sum(hidden_states[:-1] != hidden_states[1:])
        
        return {
            'hmm_frac_occupancy': frac_occupancy,
            'hmm_n_transitions': n_transitions,
            'hmm_converged': model.monitor_.converged
        }
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo la extracción de características HMM - {e}")
        return None

def extract_graph_features(matrix: np.ndarray, subject_id: str) -> Dict[str, Any] | None:
    """Calcula métricas de topología de grafos sobre una matriz de conectividad."""
    if not BCT_AVAILABLE:
        log.warning("Librería 'bctpy' no encontrada. Omitiendo características de topología.")
        return None

    log.debug(f"Sujeto {subject_id}: Extrayendo características de topología de grafos...")
    try:
        matrix_norm = matrix.copy()
        if np.any(matrix_norm < 0):
             matrix_norm = (matrix_norm - np.min(matrix_norm)) / (np.max(matrix_norm) - np.min(matrix_norm))
        np.fill_diagonal(matrix_norm, 0)

        # --- CORRECCIÓN ---
        charpath_results = bct.charpath(matrix_norm)
        char_path = charpath_results[0]
        efficiency = charpath_results[1]

        modularity_louvain, _ = bct.modularity_und(matrix_norm)
        mean_clustering = np.mean(bct.clustering_coef_wu(matrix_norm))

        return {
            'topo_global_efficiency': efficiency,
            'topo_modularity': modularity_louvain,
            'topo_char_path_length': char_path,
            'topo_mean_clustering_coef': mean_clustering,
        }
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo la extracción de topología - {e}", exc_info=True)
        return None