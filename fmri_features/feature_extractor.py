# fmri_features/feature_extractor.py
"""
Extrae características derivadas (no matriciales) como dinámica HMM 
y un conjunto extendido de métricas de topología de grafos.
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
        
        frac_occupancy = np.bincount(hidden_states, minlength=params['n_states']) / len(hidden_states)
        n_transitions = np.sum(hidden_states[:-1] != hidden_states[1:])
        
        return {
            'hmm_frac_occupancy': frac_occupancy,
            'hmm_n_transitions': n_transitions,
            'hmm_converged': model.monitor_.converged
        }
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo la extracción de características HMM - {e}", exc_info=True)
        return None

def extract_graph_features(matrix: np.ndarray, subject_id: str) -> Dict[str, Any] | None:
    """
    Calcula un conjunto extendido de métricas de topología de grafos 
    sobre una matriz de conectividad de forma numéricamente estable.
    """
    if not BCT_AVAILABLE:
        log.warning("Librería 'bctpy' no encontrada. Omitiendo características de topología.")
        return None

    log.debug(f"Sujeto {subject_id}: Extrayendo características de topología extendidas...")
    try:
        # Preprocesamiento de la matriz para asegurar pesos positivos (0-1)
        # y diagonal de ceros.
        matrix_norm = matrix.copy()
        # Manejar posibles NaNs o Infs que vengan de cálculos previos
        matrix_norm[~np.isfinite(matrix_norm)] = 0
        
        if np.any(matrix_norm < 0):
            min_val, max_val = np.min(matrix_norm), np.max(matrix_norm)
            if (max_val - min_val) > 0:
                matrix_norm = (matrix_norm - min_val) / (max_val - min_val)
            else: # Matriz constante
                matrix_norm.fill(0)
                
        np.fill_diagonal(matrix_norm, 0)

        # --- INICIO DE LA CORRECCIÓN PARA ESTABILIDAD NUMÉRICA ---
        # `charpath` y `betweenness_wei` (a través de `weight_conversion`) calculan inversas
        # de los pesos (1/W). Para evitar divisiones por cero si un peso W es 0,
        # creamos una matriz de distancia segura (donde distancia = 1/peso).
        
        # En lugar de añadir un epsilon, el método robusto es manejar los infinitos después.
        with np.errstate(divide='ignore'): # Suprimir la advertencia de división por cero aquí
            dist_matrix = bct.weight_conversion(matrix_norm, 'lengths')
        
        # Reemplazar los infinitos (resultado de 1/0) con un valor muy grande pero finito.
        # Esto preserva la topología (un camino de peso cero es una distancia infinita)
        # sin causar problemas numéricos en los algoritmos subsiguientes.
        if np.isinf(dist_matrix).any():
            max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)]) if np.any(np.isfinite(dist_matrix)) else 1
            dist_matrix[np.isinf(dist_matrix)] = max_dist * 10 # Un valor representativo de "muy lejos"
        # --- FIN DE LA CORRECCIÓN ---


        # --- Cálculo de Métricas ---

        # Métricas que usan la matriz de DISTANCIA segura
        charpath_results = bct.charpath(dist_matrix)
        char_path = charpath_results[0]
        global_efficiency = charpath_results[1]
        betweenness_centrality = bct.betweenness_wei(dist_matrix)

        # Métricas que usan la matriz de PESO original (normalizada)
        modularity_louvain, _ = bct.modularity_und(matrix_norm)
        assortativity = bct.assortativity_wei(matrix_norm, flag=0)
        strengths = bct.strengths_und(matrix_norm)
        clustering_coeffs = bct.clustering_coef_wu(matrix_norm)

        # Devolvemos el diccionario con todas las métricas
        return {
            'topo_global_efficiency': global_efficiency,
            'topo_modularity': modularity_louvain,
            'topo_char_path_length': char_path,
            'topo_assortativity': assortativity,
            'topo_mean_clustering_coef': np.mean(clustering_coeffs),
            'topo_std_clustering_coef': np.std(clustering_coeffs),
            'topo_mean_strength': np.mean(strengths),
            'topo_std_strength': np.std(strengths),
            'topo_mean_betweenness': np.mean(betweenness_centrality),
            'topo_std_betweenness': np.std(betweenness_centrality),
        }
    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo la extracción de topología extendida - {e}", exc_info=True)
        return None