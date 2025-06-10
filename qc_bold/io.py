# qc_bold/io.py
"""
Funciones para manejar la entrada/salida de datos y configuración.
"""
from __future__ import annotations
import yaml
from pathlib import Path
import pandas as pd
import scipy.io as sio
import numpy as np
import logging

# Configura un logger para este módulo
log = logging.getLogger(__name__)

def load_config(config_path: str | Path) -> dict:
    """Carga la configuración desde un archivo YAML."""
    log.info(f"Cargando configuración desde: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def find_subjects(roi_dir: Path, subjects_csv_path: Path) -> pd.DataFrame:
    """
    Carga el CSV de sujetos, encuentra sus archivos .mat y verifica su existencia.
    """
    log.info(f"Buscando archivos .mat en: {roi_dir}")
    try:
        meta_df = pd.read_csv(subjects_csv_path)
        subject_id_col = 'SubjectID'
        meta_df[subject_id_col] = meta_df[subject_id_col].astype(str).str.strip()
        
        meta_df['mat_path'] = meta_df[subject_id_col].apply(
                        lambda sid: Path(roi_dir) / f'ROISignals_{sid}.mat')
        meta_df['mat_exists'] = meta_df['mat_path'].apply(lambda p: p.exists())
        
        n_found = meta_df['mat_exists'].sum()
        log.info(f"Encontrados {n_found} de {len(meta_df)} archivos .mat de sujetos.")
        if n_found == 0:
            raise FileNotFoundError("No se encontraron archivos .mat para ningún sujeto.")
            
        return meta_df[meta_df['mat_exists']].copy()

    except FileNotFoundError as e:
        log.error(f"No se pudo encontrar el archivo CSV de sujetos en: {subjects_csv_path}")
        raise e
    except Exception as e:
        log.error(f"Ocurrió un error al procesar los datos de los sujetos: {e}")
        raise e

def load_mat_data(mat_path: Path) -> np.ndarray | None:
    """Carga la matriz de señales desde un archivo .mat."""
    try:
        data = sio.loadmat(str(mat_path))
        # Busca la clave de la matriz de señales de forma más flexible
        signals_key = 'signals'
        if signals_key not in data:
            potential_keys = [k for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 2]
            if not potential_keys:
                raise KeyError("No se encontró ninguna matriz 2D en el archivo .mat")
            signals_key = potential_keys[0]
            log.warning(f"Clave 'signals' no encontrada. Usando la clave '{signals_key}' de {mat_path.name}")
        
        return np.asarray(data[signals_key], dtype=float)
    
    except Exception as e:
        log.error(f"Error cargando el archivo .mat '{mat_path.name}': {e}")
        return None