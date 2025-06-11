#!/usr/bin/env python
# scripts/process_fmri.py
"""
Script principal para procesar datos de fMRI.
Carga configuración, procesa sujetos y extrae características.
"""
import os
import sys
import logging
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

# Añadir el directorio principal al path
sys.path.append(str(Path(__file__).parent.parent))

# Importar funciones del módulo fmri_features
from fmri_features.data_loader import (
    get_subjects_to_process,
    _get_rois_to_remove,
    load_and_preprocess_ts
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Carga el archivo de configuración YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Función principal para procesar datos de fMRI."""
    # Cargar configuración
    config_path = Path(__file__).parent.parent / 'config' / 'config.yml'
    cfg = load_config(str(config_path))
    
    # Obtener sujetos que pasaron el QC
    qc_path = Path(cfg['paths']['qc_report_path'])
    subjects_df = get_subjects_to_process(qc_path)
    
    # Obtener ROIs a eliminar
    rois_to_remove = _get_rois_to_remove(cfg)
    
    # Crear directorio de salida si no existe
    output_dir = Path(cfg['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada sujeto
    processed_data = {}
    for idx, row in subjects_df.iterrows():
        subject_id = row['subject_id']
        log.info(f"Procesando sujeto {subject_id}...")
        
        # Cargar y preprocesar datos
        ts_data = load_and_preprocess_ts(subject_id, cfg, rois_to_remove)
        
        if ts_data is not None:
            processed_data[subject_id] = ts_data
            log.info(f"Sujeto {subject_id} procesado exitosamente. Shape: {ts_data.shape}")
        else:
            log.warning(f"No se pudo procesar el sujeto {subject_id}.")
    
    log.info(f"Procesamiento completo. {len(processed_data)} sujetos procesados exitosamente.")
    
    # Guardar resultados procesados
    # (Aquí puedes añadir el código para guardar los resultados)
    
if __name__ == "__main__":
    main()
