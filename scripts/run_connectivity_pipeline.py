# scripts/run_connectivity_pipeline.py
"""
Script principal para ejecutar el pipeline de extracción de características de conectividad.
"""
import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from datetime import datetime
from typing import List, Dict

# Añadir el directorio raíz del proyecto al path para poder importar los paquetes
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fmri_features import data_loader, connectome_generator, feature_extractor

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

def process_subject(subject_id: str, cfg: Dict, rois_to_remove: List[int]) -> Dict:
    """
    Orquesta el pipeline completo para un único sujeto:
    1. Carga y preprocesa la serie temporal (con limpieza de ROIs).
    2. Genera el tensor de conectividad.
    3. Extrae características adicionales.
    4. Devuelve un diccionario con los resultados y rutas.
    """
    result = {'subject_id': subject_id, 'status': 'FAILURE', 'tensor_path': None, 'features': {}}
    
    # 1. Cargar y Preprocesar TS
    ts_data = data_loader.load_and_preprocess_ts(subject_id, cfg, rois_to_remove)
    if ts_data is None:
        result['error'] = 'Fallo en la carga o preprocesamiento de la serie temporal.'
        return result
        
    # 2. Generar Tensor de Conectividad
    tensor = connectome_generator.generate_connectivity_tensor(ts_data, cfg, subject_id)
    if tensor is None:
        result['error'] = 'Fallo en la generación del tensor de conectividad.'
        return result

    # 3. Extraer Características Derivadas
    if cfg.get('features', {}).get('hmm_dynamics'):
        result['features']['hmm'] = feature_extractor.extract_hmm_features(ts_data, cfg, subject_id)
        
    if cfg.get('features', {}).get('graph_topology'):
        # Usar la primera capa del tensor (ej. Pearson) como base para la topología
        base_matrix_for_topo = tensor[0, :, :] 
        result['features']['topology'] = feature_extractor.extract_graph_features(base_matrix_for_topo, subject_id)

    result['status'] = 'SUCCESS'
    result['tensor'] = tensor # Devolvemos el tensor para guardarlo en el proceso principal
    return result


def main():
    """Función principal que ejecuta el pipeline completo."""
    script_start_time = time.time()
    log.info("--- INICIANDO PIPELINE DE EXTRACCIÓN DE CONECTIVIDAD fMRI ---")
    
    # Cargar configuración
    config_path = project_root / 'config_connectivity.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Crear directorio de salida
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    active_channels = [k for k, v in cfg.get('channels', {}).items() if v]
    output_dir_name = f"connectivity_{len(active_channels)}ch_{timestamp}"
    output_dir = Path(cfg['paths']['base_output_dir']) / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Los resultados se guardarán en: {output_dir}")
    
    # Guardar la configuración usada en esta ejecución
    with open(output_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(cfg, f)

    # Obtener lista de sujetos que pasaron el QC
    qc_report_path = Path(cfg['paths']['qc_output_dir']) / cfg['paths']['qc_report_filename']
    subjects_df = data_loader.get_subjects_to_process(qc_report_path)
    subject_ids = subjects_df['subject_id'].tolist()
    
    # --- PASO CLAVE: Determinar qué ROIs eliminar ANTES de empezar el bucle ---
    rois_to_remove = data_loader._get_rois_to_remove(cfg)

    # Procesamiento en paralelo
    max_workers = cfg.get('max_workers', 'auto')
    if max_workers == 'auto':
        max_workers = max(1, multiprocessing.cpu_count() // 2)
    
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Crear un mapa de futuros para poder pasar la configuración
        futures = {executor.submit(process_subject, sid, cfg, rois_to_remove): sid for sid in subject_ids}
        
        for future in tqdm(as_completed(futures), total=len(subject_ids), desc="Procesando Sujetos"):
            try:
                res = future.result()
                # Guardar el tensor desde el proceso principal
                if res['status'] == 'SUCCESS' and 'tensor' in res:
                    tensor_path = output_dir / f"tensor_{res['subject_id']}.npy"
                    np.save(tensor_path, res['tensor'])
                    res['tensor_path'] = str(tensor_path)
                    del res['tensor']
                all_results.append(res)
            except Exception as e:
                log.error(f"Error procesando al sujeto {futures[future]}: {e}", exc_info=True)
                all_results.append({'subject_id': futures[future], 'status': 'CRITICAL_FAILURE', 'error': str(e)})
    
    # Consolidar y guardar resultados/logs
    results_summary_df = pd.DataFrame(all_results)
    
    # Aplanar las características para un fácil guardado en CSV
    if not results_summary_df.empty and 'features' in results_summary_df.columns:
        # Usar json_normalize para aplanar el diccionario de características
        features_flat_df = pd.json_normalize(results_summary_df['features'])
        
        # El array de ocupación HMM necesita un tratamiento especial
        if 'hmm.hmm_frac_occupancy' in features_flat_df.columns:
            # Obtener el número de estados desde la configuración para crear las columnas
            n_states = cfg.get('parameters', {}).get('hmm', {}).get('n_states', 5)
            occupancy_cols = [f'hmm_occupancy_{i}' for i in range(n_states)]
            
            # Crear un DataFrame con las columnas de ocupación
            occupancy_df = pd.DataFrame(
                features_flat_df['hmm.hmm_frac_occupancy'].to_list(), 
                columns=occupancy_cols,
                index=features_flat_df.index
            )
            
            # Unir las nuevas columnas y eliminar la original
            features_flat_df = features_flat_df.drop(columns=['hmm.hmm_frac_occupancy']).join(occupancy_df)

        # Unir el DataFrame aplanado con el de resultados principal
        results_summary_df = results_summary_df.drop(columns=['features']).join(features_flat_df)

    summary_path = output_dir / "processing_summary_log.csv"
    results_summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    
    total_time_min = (time.time() - script_start_time) / 60
    log.info(f"--- Pipeline de Conectividad Finalizado en {total_time_min:.2f} minutos ---")
    log.info(f"Resumen del procesamiento guardado en: {summary_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main()