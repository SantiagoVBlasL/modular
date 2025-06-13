# training/data_handling.py
"""
Funciones para cargar, pre-procesar y preparar los datasets para el entrenamiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
import torch

log = logging.getLogger(__name__)

def load_full_dataset(data_dir: Path, channel_indices: list[int] | None) -> dict | None:
    """Carga el dataset completo (tensores, características, claves) preparado por el script anterior."""
    log.info(f"Cargando dataset completo para CV desde: {data_dir}")
    try:
        data = {
            'key_df': pd.read_csv(data_dir / "cv_subjects_key.csv"),
            'tensors': np.load(data_dir / "cv_all_tensors.npy"),
            'features': np.load(data_dir / "cv_all_features_unscaled.npy"),
        }
        if channel_indices:
            log.info(f"Seleccionando canales en los índices: {channel_indices}")
            data['tensors'] = data['tensors'][:, channel_indices, :, :]
        return data
    except FileNotFoundError as e:
        log.error(f"Error: Archivo no encontrado en {data_dir}. Ejecuta 'prepare_and_analyze_data.py' primero. Detalle: {e}")
        return None

def preprocess_tensors_robustly(tensors: np.ndarray) -> np.ndarray:
    """Aplica un pre-procesado robusto a todo el conjunto de tensores para alinear con la activación 'tanh'."""
    log.info("Iniciando pre-procesado robusto de tensores...")
    tensors_processed = np.copy(tensors)

    # 1. Poner a cero la diagonal
    log.info("Paso 1: Poniendo a cero la diagonal de las matrices.")
    for i in range(tensors_processed.shape[0]):
        for j in range(tensors_processed.shape[1]):
            np.fill_diagonal(tensors_processed[i, j], 0)

    # 2. Escalar robustamente a [-1, 1] por canal
    log.info("Paso 2: Aplicando escalado robusto por canal para ajustar a rango [-1, 1].")
    for i in range(tensors_processed.shape[1]):
        p_max = np.percentile(np.abs(tensors_processed[:, i, :, :]), 99.5)
        if p_max < 1e-6: 
            log.warning(f"  - Canal {i}: Percentil 99.5 es casi cero ({p_max:.2e}). Se usará 1.0 para evitar división por cero.")
            p_max = 1.0
        
        tensors_processed[:, i, :, :] = np.clip(tensors_processed[:, i, :, :] / p_max, -1.0, 1.0)
        log.info(f"  - Canal {i}: Escalado con p99.5 = {p_max:.4f}. Rango final: [{np.min(tensors_processed[:, i, :, :]):.2f}, {np.max(tensors_processed[:, i, :, :]):.2f}]")

    log.info("Pre-procesado de tensores completado.")
    return tensors_processed

def create_final_feature_vector(vae_model, tensors, scalar_features, age_sex_features, device):
    """Genera el vector de características híbrido final para el clasificador."""
    vae_model.eval()
    with torch.no_grad():
        mu, _ = vae_model.encode(torch.from_numpy(tensors).float().to(device))
    latent_vec = mu.cpu().numpy()
    
    log.info(f"Composición del vector de características final: Latente(VAE)={latent_vec.shape}, Escalares(Topo/HMM)={scalar_features.shape}, Metadatos(Edad/Sexo)={age_sex_features.shape}")
    return np.concatenate([latent_vec, scalar_features, age_sex_features], axis=1)
