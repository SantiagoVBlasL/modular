# scripts/patch_mutual_info.py
"""
Script de reparación final y robusto para recalcular y reemplazar un canal de
conectividad específico (ej. 'mutual_info_full') en los tensores ya generados.
Esta versión utiliza un método de cálculo manual para la Información Mutua,
evitando los errores de la librería y asegurando la ejecución.
"""
import sys
import yaml
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional

# --- Añadir el directorio raíz del proyecto al path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fmri_features import data_loader

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)


def calculate_mi_manually(ts: np.ndarray, cfg: dict) -> Optional[np.ndarray]:
    """
    Calcula la matriz de Información Mutua de forma manual y robusta,
    iterando sobre pares de ROIs usando mutual_info_score. Este método
    es más estable que usar mutual_info_regression.
    """
    try:
        from sklearn.metrics import mutual_info_score
    except ImportError:
        log.error("Scikit-learn no está instalado. No se puede calcular la Información Mutua.")
        return None

    n_rois = ts.shape[1]
    
    # La discretización es clave para MI. Usamos ranks para un método no paramétrico.
    ranks = ts.argsort(axis=0).argsort(axis=0)
    
    # Obtener n_bins de la configuración de forma segura
    n_bins_config = cfg.get('parameters', {}).get('mutual_info', {})
    n_bins = 10
    if n_bins_config is not None:
        n_bins = n_bins_config.get('n_bins', 10)
    
    discrete_ts = np.floor(ranks / len(ts) * n_bins).astype(int)
    
    mi_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
    
    # Iterar sobre cada par único de ROIs para calcular su MI
    for i in range(n_rois):
        for j in range(i, n_rois):
            if i == j:
                # La diagonal de una matriz de conectividad es 0.
                mi_matrix[i, j] = 0
                continue
            
            # Calcular MI para el par (i, j)
            mi = mutual_info_score(discrete_ts[:, i], discrete_ts[:, j])
            
            # Asignar el valor a la matriz simétrica
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def patch_connectivity_channel(run_dir_name: str, channel_to_patch: str):
    """
    Recalcula un canal de conectividad y actualiza los archivos .npy.
    """
    run_path = project_root / 'connectivity_features' / run_dir_name
    config_path = run_path / 'config_used.yaml'
    
    if not run_path.exists() or not config_path.exists():
        log.error(f"La carpeta de ejecución '{run_dir_name}' o su config.yaml no existen.")
        return

    log.info(f"Iniciando el parche para el canal '{channel_to_patch}' en la corrida '{run_dir_name}'.")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    active_channels = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    try:
        channel_index = active_channels.index(channel_to_patch)
        log.info(f"El canal '{channel_to_patch}' se reemplazará en el índice {channel_index} del tensor.")
    except ValueError:
        log.error(f"El canal '{channel_to_patch}' no se encontró en la configuración: {active_channels}")
        return

    report_path = run_path / "salvaged_analysis" / "final_report_with_features.csv"
    if not report_path.exists():
        log.error(f"No se encontró el archivo de reporte 'final_report_with_features.csv' en {report_path.parent}.")
        return

    log.info(f"Cargando lista de sujetos desde: {report_path}")
    df_subjects = pd.read_csv(report_path)
    
    if 'tensor_path' not in df_subjects.columns:
        log.error(f"La columna 'tensor_path' no se encontró en {report_path}. No se puede continuar.")
        return

    rois_to_remove = data_loader._get_rois_to_remove(cfg)
    
    subjects_processed, subjects_failed = 0, 0
    
    for _, row in tqdm(df_subjects.iterrows(), total=len(df_subjects), desc="Reparando tensores"):
        subject_id = str(row['subject_id'])
        tensor_path = row['tensor_path']
        
        try:
            ts_data = data_loader.load_and_preprocess_ts(subject_id, cfg, rois_to_remove)
            if ts_data is None:
                log.warning(f"No se pudo cargar la serie temporal para {subject_id}. Omitiendo.")
                subjects_failed += 1
                continue
            
            # Llamar a la nueva función de cálculo manual
            new_matrix = calculate_mi_manually(ts_data, cfg)

            if new_matrix is None:
                log.error(f"Fallo el cálculo de MI para {subject_id}.")
                subjects_failed += 1
                continue
            
            # Cargar el tensor, reemplazar el canal y guardar
            tensor = np.load(tensor_path)
            tensor[channel_index, :, :] = new_matrix
            np.save(tensor_path, tensor)
            subjects_processed += 1

        except Exception as e:
            log.error(f"Fallo irrecuperable procesando al sujeto {subject_id}: {e}", exc_info=True)
            subjects_failed += 1
            continue

    log.info("--- PROCESO DE REPARACIÓN FINALIZADO ---")
    log.info(f"Sujetos procesados exitosamente: {subjects_processed}")
    log.info(f"Sujetos fallidos: {subjects_failed}")


def main():
    parser = argparse.ArgumentParser(description="Recalcula y parchea un canal de conectividad en tensores existentes.")
    parser.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a analizar.")
    parser.add_argument('--channel', default='mutual_info_full', help="Nombre exacto del canal a recalcular.")
    args = parser.parse_args()
    
    patch_connectivity_channel(args.run, args.channel)

if __name__ == '__main__':
    main()