# scripts/salvage_and_analyze.py
"""
Script de rescate y análisis para consolidar los resultados de una corrida
del pipeline de conectividad que falló al final.

Este script:
1.  Encuentra todos los tensores .npy de una ejecución específica.
2.  Recalcula de forma segura las características HMM y de topología.
3.  Maneja correctamente los casos en que las características no se pueden calcular.
4.  Genera el archivo CSV de resumen final.
5.  Crea las mismas visualizaciones que el script de análisis original.
"""
import sys
import yaml
import logging
import argparse
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

# --- Añadir el directorio raíz del proyecto al path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fmri_features import feature_extractor, data_loader # Importamos ambos

# --- Configuración del Estilo y Logging ---
sns.set_theme(style='whitegrid', context='notebook')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def find_and_load_data(run_dir_name: str, subjects_csv_path: str) -> Optional[Tuple[pd.DataFrame, Dict, Path]]:
    """Escanea la carpeta de ejecución, encuentra tensores .npy y los une con metadatos."""
    base_output_dir = project_root / 'connectivity_features'
    run_path = base_output_dir / run_dir_name
    config_path = run_path / 'config_used.yaml'
    
    if not run_path.exists() or not config_path.exists():
        log.error(f"La carpeta de ejecución '{run_dir_name}' o su config.yaml no existen en '{base_output_dir}'.")
        return None

    log.info(f"Escaneando resultados desde: {run_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    tensor_files = list(run_path.glob('tensor_*.npy'))
    if not tensor_files:
        log.warning("No se encontraron archivos de tensor .npy.")
        return None

    log.info(f"Se encontraron {len(tensor_files)} tensores de sujetos procesados.")
    
    processed_subjects = [{'subject_id': f.stem.split('tensor_')[-1], 'tensor_path': str(f)} for f in tensor_files]
    processed_df = pd.DataFrame(processed_subjects)

    meta_path = project_root / subjects_csv_path
    if not meta_path.exists():
        log.error(f"No se encontró el archivo de metadatos: {meta_path}")
        return None
    
    meta_df = pd.read_csv(meta_path)
    # Asegurar que los IDs de sujeto sean strings para un merge robusto
    processed_df['subject_id'] = processed_df['subject_id'].astype(str)
    meta_df['SubjectID'] = meta_df['SubjectID'].astype(str)
    
    # Unir con metadatos y QC
    qc_report_path = project_root / cfg['paths']['qc_output_dir'] / cfg['paths']['qc_report_filename']
    qc_df = data_loader.get_subjects_to_process(qc_report_path)
    qc_df['subject_id'] = qc_df['subject_id'].astype(str)

    full_df = pd.merge(processed_df, qc_df, on='subject_id', how='left')
    full_df = pd.merge(full_df, meta_df.rename(columns={'SubjectID': 'subject_id'}), on='subject_id', how='left')

    return full_df, cfg, run_path


def recalculate_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recalcula las características escalares a partir de los tensores .npy guardados."""
    all_features = []
    
    # Para la extracción de HMM, necesitamos las series temporales originales
    rois_to_remove = data_loader._get_rois_to_remove(cfg)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Recalculando características"):
        features = {'subject_id': row['subject_id']}
        tensor = np.load(row['tensor_path'], mmap_mode='r')
        
        # Recalcular topología
        if cfg.get('features', {}).get('graph_topology'):
            base_matrix = tensor[0, :, :]
            topo_features = feature_extractor.extract_graph_features(base_matrix, row['subject_id'])
            if topo_features:
                features.update(topo_features)
        
        # Recalcular HMM
        if cfg.get('features', {}).get('hmm_dynamics'):
            # Necesitamos recargar la serie temporal para el HMM
            ts_data = data_loader.load_and_preprocess_ts(row['subject_id'], cfg, rois_to_remove)
            if ts_data is not None:
                hmm_features = feature_extractor.extract_hmm_features(ts_data, cfg, row['subject_id'])
                if hmm_features:
                    features.update(hmm_features)
            del ts_data # Liberar memoria
        
        all_features.append(features)
        del tensor
        gc.collect()

    features_df = pd.DataFrame(all_features)
    
    # SOLUCIÓN AL TYPEERROR: aplanar las características de forma segura
    
    # 1. Tratar con la ocupación HMM
    if 'hmm_frac_occupancy' in features_df.columns:
        # Filtrar filas donde la ocupación es NaN (porque el HMM falló)
        valid_occupancy = features_df['hmm_frac_occupancy'].dropna()
        
        n_states = cfg.get('parameters', {}).get('hmm', {}).get('n_states', 5)
        occupancy_cols = [f'hmm_occupancy_{i}' for i in range(n_states)]
        
        # Crear el DataFrame de ocupación SÓLO para las filas válidas
        occupancy_df = pd.DataFrame(
            valid_occupancy.to_list(),
            columns=occupancy_cols,
            index=valid_occupancy.index
        )
        # Unir de nuevo, los sujetos con HMM fallido tendrán NaN en estas columnas
        features_df = features_df.join(occupancy_df)
        features_df = features_df.drop(columns=['hmm_frac_occupancy'])

    # 2. Aplanar otras características anidadas si las hubiera
    # (En este caso, no parece haber más, pero es buena práctica)

    # Unir con el DataFrame principal
    final_df = pd.merge(df, features_df, on='subject_id', how='left')
    return final_df


def main():
    """Orquesta el análisis y la visualización."""
    parser = argparse.ArgumentParser(description="Script de rescate para resultados de conectividad.")
    parser.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a analizar.")
    parser.add_argument('--meta', default='SubjectsData_Schaefer400.csv', help="CSV con metadatos de sujetos.")
    args = parser.parse_args()
    
    run_data = find_and_load_data(args.run, args.meta)
    if run_data is None: 
        log.error("No se pudieron cargar los datos. Abortando.")
        return
        
    df_processed, cfg, run_path = run_data

    # El directorio de salida para los nuevos artefactos
    salvage_dir = run_path / "salvaged_analysis"
    salvage_dir.mkdir(exist_ok=True)
    
    log.info("Recalculando y consolidando características...")
    df_final_report = recalculate_features(df_processed, cfg)
    
    # Guardar el CSV que falló originalmente
    final_csv_path = salvage_dir / "final_report_with_features.csv"
    df_final_report.to_csv(final_csv_path, index=False, float_format='%.6f')
    log.info(f"¡ÉXITO! Reporte final consolidado y guardado en: {final_csv_path}")
    
    # (Opcional) Generar las mismas visualizaciones que el script de análisis
    # (Se puede añadir aquí el código de los plots de analyze_and_visualize_results.py si se desea)

if __name__ == '__main__':
    main()