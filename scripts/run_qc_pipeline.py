# scripts/run_qc_pipeline.py
"""
Script principal para ejecutar el pipeline de control de calidad (QC) de BOLD fMRI.
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Añadir el directorio raíz del proyecto al path para poder importar qc_bold
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from qc_bold import io, univariate, multivariate, report

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


def initial_qc(signals: np.ndarray, expected_rois: int) -> dict:
    """Realiza un QC rápido inicial sobre la matriz de datos cruda."""
    if signals is None or signals.size == 0:
        return {'timepoints': 0, 'raw_rois': 0, 'nan_pct': 100.0, 'null_channels_pct': 100.0}

    tp, raw_rois_count = signals.shape
    nan_pct = 100 * np.isnan(signals).sum() / signals.size if signals.size > 0 else 0
    null_channels_count = np.sum(np.all(signals == 0, axis=0))
    null_channels_pct = 100 * null_channels_count / raw_rois_count if raw_rois_count > 0 else 0
    
    return {
        'timepoints': tp,
        'raw_rois': raw_rois_count,
        'raw_rois_match_expected': raw_rois_count == expected_rois,
        'nan_pct': nan_pct,
        'null_channels_pct': null_channels_pct
    }

def process_subject(subject_info: pd.Series, config: dict) -> dict:
    """Procesa un único sujeto a través de todo el pipeline de QC."""
    subject_id = subject_info['SubjectID']
    log.info(f"Procesando sujeto: {subject_id}")
    
    # 1. Cargar datos
    raw_signals = io.load_mat_data(subject_info['mat_path'])
    if raw_signals is None:
        log.warning(f"No se pudo cargar el archivo .mat para el sujeto {subject_id}. Saltando.")
        return {'subject_id': subject_id, 'error': 'Failed to load .mat file'}

    # 2. QC Inicial
    qc_initial_results = initial_qc(raw_signals, config['atlas']['raw_expected_rois'])

    # 3. Detección de Outliers Univariantes (sobre la matriz cruda)
    #    Este paso es importante y faltaba en la versión anterior.
    _, univ_pct = univariate.detect_univariate_outliers(
        raw_signals,
        config['outliers']['univariate']['z_threshold'],
        config['outliers']['univariate']['method']
    )

    # 4. Detección de Outliers Multivariantes (con filtrado previo)
    log.debug(f"Sujeto {subject_id}: Filtrando canales inválidos para análisis multivariante.")
    
    # Los índices en el config son 1-based, se convierten a 0-based para Python.
    aal3_indices_to_remove = [idx - 1 for idx in config['atlas']['aal3_missing_indices_1based']]

    # Se usa np.delete para eliminar las columnas (axis=1) de los canales inválidos.
    filtered_signals = np.delete(raw_signals, aal3_indices_to_remove, axis=1)

    # Verificación para asegurar que el filtrado resultó en el número esperado de ROIs.
    # Esta variable se añade al config.yaml.
    expected_rois_after_filter = 166 # Calculado de 170 - 4
    if filtered_signals.shape[1] != expected_rois_after_filter:
        log.warning(f"Sujeto {subject_id}: Tras filtrar, se esperaban {expected_rois_after_filter} ROIs pero hay {filtered_signals.shape[1]}.")
    
    # Llamada ÚNICA al análisis multivariante con los datos ya filtrados.
    mv_results = multivariate.detect_multivariate_outliers(filtered_signals, config)

    # 5. Consolidar todos los resultados
    final_results = {
        'subject_id': subject_id,
        **qc_initial_results,
        'univ_outliers_pct': univ_pct, # Se usa el resultado del paso 3.
        **mv_results # Se usa el resultado del paso 4.
    }
    return final_results


def main():
    """Función principal que ejecuta el pipeline completo."""
    log.info("--- INICIANDO PIPELINE DE QC DE BOLD fMRI (REFACTORIZADO Y CORREGIDO) ---")
    
    # Cargar configuración
    config_path = project_root / 'config.yaml'
    cfg = io.load_config(config_path)
    
    # Crear directorio de salida
    export_dir = Path(cfg['paths']['export_dir'])
    export_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Los resultados se guardarán en: {export_dir}")

    # Encontrar sujetos
    subjects_df = io.find_subjects(Path(cfg['paths']['roi_dir']), Path(cfg['paths']['subjects_csv']))

    # Procesar cada sujeto
    all_results = []
    # Usar tqdm para una barra de progreso
    for _, subject_row in tqdm(subjects_df.iterrows(), total=len(subjects_df), desc="Procesando Sujetos"):
        subject_result = process_subject(subject_row, cfg)
        all_results.append(subject_result)
        
    # Crear DataFrame final
    results_df = pd.DataFrame(all_results)
    
    # Aplicar criterios de exclusión
    excl_cfg = cfg['exclusion_criteria']
    results_df['discard_low_tp'] = results_df['timepoints'] < excl_cfg['min_timepoints']
    results_df['discard_high_nan'] = results_df['nan_pct'] > excl_cfg['max_nan_pct']
    results_df['discard_high_univ'] = results_df['univ_outliers_pct'] > excl_cfg['max_univ_outliers_pct']
    results_df['discard_high_mv'] = results_df['mv_outliers_pct'] > excl_cfg['max_mv_outliers_pct']
    results_df['discard_mv_skipped'] = results_df['mv_skipped_reason'].notna() & excl_cfg['exclude_if_mv_skipped']
    
    discard_cols = [col for col in results_df.columns if col.startswith('discard_')]
    results_df['to_discard_overall'] = results_df[discard_cols].any(axis=1)

    log.info("\n=== Resumen de Descarte ===")
    for col in discard_cols:
        log.info(f"Sujetos descartados por '{col.replace('discard_', '')}': {results_df[col].sum()}")
    log.info(f"Total a descartar: {results_df['to_discard_overall'].sum()} / {len(results_df)}")
    log.info(f"Total retenidos: {len(results_df) - results_df['to_discard_overall'].sum()}")

    # Guardar reportes
    report.save_dataframe(results_df, export_dir, 'report_qc_final.csv')
    
    # Guardar metadatos de la ejecución en el DataFrame para los gráficos
    results_df.attrs = {
        'z_thresh': cfg['outliers']['univariate']['z_threshold'],
        'alpha_mahal': cfg['outliers']['multivariate']['alpha_mahalanobis'],
        'max_univ_outliers_pct': excl_cfg['max_univ_outliers_pct'],
        'max_mv_outliers_pct': excl_cfg['max_mv_outliers_pct'],
    }
    report.generate_summary_plots(results_df, export_dir)
    # Pasar el diccionario de configuración completo al generador de reportes HTML
    report.generate_html_report(results_df, cfg, export_dir)

    log.info("--- PIPELINE COMPLETADO ---")

if __name__ == '__main__':
    main()