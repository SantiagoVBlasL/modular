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
    sobre una matriz de conectividad.
    """
    if not BCT_AVAILABLE:
        log.warning("Librería 'bctpy' no encontrada. Omitiendo características de topología.")
        return None

    log.debug(f"Sujeto {subject_id}: Extrayendo características de topología extendidas...")
    try:
        # Preprocesamiento de la matriz para asegurar pesos positivos (0-1)
        # y diagonal de ceros, requerido por muchas funciones de bctpy.
        matrix_norm = matrix.copy()
        if np.any(matrix_norm < 0):
             matrix_norm = (matrix_norm - np.min(matrix_norm)) / (np.max(matrix_norm) - np.min(matrix_norm))
        np.fill_diagonal(matrix_norm, 0)

        # --- Métricas Globales Existentes ---
        charpath_results = bct.charpath(matrix_norm)
        char_path = charpath_results[0]
        global_efficiency = charpath_results[1]

        modularity_louvain, _ = bct.modularity_und(matrix_norm)
        
        # --- NUEVAS MÉTRICAS ---

        # 1. Asortatividad: Mide la preferencia de los nodos a conectarse con otros de grado similar.
        assortativity = bct.assortativity_wei(matrix_norm, flag=0)

        # 2. Eficiencia Local: Promedio de la eficiencia de las subredes de cada nodo.
        #    Indica qué tan tolerante es la red a fallos a nivel local.
        # --- 2. Eficiencia local ------------------------------------------
        try:
            # API 0.6: devuelve solo array de Eloc
            local_efficiencies = bct.efficiency_wei(matrix_norm, local=True)
        except TypeError:
            # API 0.5: devuelve (Eglob, Eloc) incluso con local=True
            _, local_efficiencies = bct.efficiency_wei(matrix_norm, local=True)
        except ValueError:
            # API 0.5 sin keyword local → devuelve (Eglob, Eloc)
            _, local_efficiencies = bct.efficiency_wei(matrix_norm)

        mean_local_efficiency = float(np.mean(local_efficiencies))



        # 3. Medidas Nodales (resumidas con media y desviación estándar)
        
        # Grado/Fuerza (Degree/Strength): Suma de los pesos de las conexiones de cada nodo.
        strengths = bct.strengths_und(matrix_norm)
        
        # Centralidad de Intermediación (Betweenness Centrality): Mide la influencia de un nodo sobre el flujo
        # de información entre otros nodos en la red.
        # NOTA: bct.betweenness_wei espera una matriz de longitud (distancia), no de peso.
        dist_matrix = bct.weight_conversion(matrix_norm, 'lengths')
        betweenness_centrality = bct.betweenness_wei(dist_matrix)

        # Coeficiente de Clustering: Mide la tendencia de los nodos a formar clústeres.
        clustering_coeffs = bct.clustering_coef_wu(matrix_norm)

        # Devolvemos el diccionario con todas las métricas
        return {
            # Originales
            'topo_global_efficiency': global_efficiency,
            'topo_modularity': modularity_louvain,
            'topo_char_path_length': char_path,
            
            # Nuevas
            'topo_assortativity': assortativity,
            'topo_mean_local_efficiency': mean_local_efficiency,
            
            # Resúmenes de Métricas Nodales
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