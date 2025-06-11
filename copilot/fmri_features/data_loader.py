# fmri_features/data_loader.py
"""
Carga y preprocesa datos de sujetos que pasaron el QC,
incluyendo la eliminación de ROIs no válidos.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict

log = logging.getLogger(__name__)

# (Las funciones get_subjects_to_process y _get_rois_to_remove no cambian)
def get_subjects_to_process(qc_report_path: Path) -> pd.DataFrame:
    """Carga el reporte de QC y devuelve un DataFrame con los sujetos que pasaron."""
    log.info(f"Cargando reporte de QC desde: {qc_report_path}")
    if not qc_report_path.exists():
        log.error(f"El archivo de reporte de QC no se encontró: {qc_report_path}")
        raise FileNotFoundError(f"QC report not found at {qc_report_path}")

    qc_df = pd.read_csv(qc_report_path)
    
    if 'to_discard_overall' not in qc_df.columns:
        log.error("La columna 'to_discard_overall' no se encuentra en el reporte de QC.")
        raise ValueError("Missing 'to_discard_overall' column in QC report.")

    subjects_passed = qc_df[qc_df['to_discard_overall'] == False].copy()
    log.info(f"{len(subjects_passed)} de {len(qc_df)} sujetos pasaron el QC y serán procesados.")
    return subjects_passed

def _get_rois_to_remove(cfg: dict) -> list[int]:
    """
    Identifica los índices de los ROIs a eliminar basados en la configuración del atlas.
    """
    atlas_cfg = cfg.get('atlas', {})
    if not atlas_cfg:
        log.warning("Sección 'atlas' no encontrada en la configuración. No se eliminarán ROIs.")
        return []
        
    meta_path = Path(atlas_cfg['aal3_meta_path'])
    if not meta_path.exists():
        log.warning(f"Archivo de metadatos del atlas no encontrado en {meta_path}. No se eliminarán ROIs.")
        return []

    meta_df = pd.read_csv(meta_path, sep='\t')
    meta_df['color'] = pd.to_numeric(meta_df['color'], errors='coerce').dropna().astype(int)
    
    missing_colors = set(atlas_cfg.get('aal3_missing_indices_1based', []))
    small_rois_mask = meta_df['vol_vox'] < atlas_cfg.get('small_roi_voxel_threshold', 0)
    small_colors = set(meta_df[small_rois_mask]['color'])
    
    colors_to_remove = missing_colors.union(small_colors)
    
    # Asumimos que la columna de datos 'i' corresponde al ROI con ID de color 'i+1'.
    indices_to_remove = [color - 1 for color in colors_to_remove]
    
    indices_to_remove = sorted(list(set(indices_to_remove)))
    log.info(f"Se ha identificado una lista de {len(indices_to_remove)} ROIs a eliminar por sus índices de columna.")
    return indices_to_remove


def _bandpass_filter(ts: np.ndarray, low: float, high: float, fs: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, ts, axis=0)

def _homogenize_length(ts: np.ndarray, target_len: int) -> np.ndarray:
    current_len = ts.shape[0]
    if current_len == target_len:
        return ts
    
    log.debug(f"Homogeneizando longitud de {current_len} a {target_len}.")
    if current_len > target_len:
        return ts[:target_len, :]
    
    x_old = np.linspace(0, 1, current_len)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, ts, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

def load_and_preprocess_ts(subject_id: str, cfg: Dict, rois_to_remove: List[int]) -> np.ndarray | None:
    """Carga, limpia, preprocesa y devuelve la serie temporal de un sujeto."""
    roi_dir = Path(cfg['paths']['roi_signals_dir'])
    mat_path = roi_dir / f'ROISignals_{subject_id}.mat'
    
    if not mat_path.exists():
        log.warning(f"Sujeto {subject_id}: No se encontró el archivo .mat en {mat_path}.")
        return None

    try:
        data = sio.loadmat(mat_path)
        key = next((k for k in ['signals', 'ROISignals'] if k in data), None)
        if key is None:
            log.warning(f"Sujeto {subject_id}: No se encontró una clave de señal válida en el .mat.")
            return None
        
        ts_raw = np.asarray(data[key], dtype=float)
        
        # Orientación de la matriz
        if ts_raw.shape[0] == 170 and ts_raw.shape[1] != 170:
            ts_raw = ts_raw.T
        elif ts_raw.shape[1] != 170:
            log.warning(f"Sujeto {subject_id}: No se pudo identificar la dimensión de 170 ROIs. Shape: {ts_raw.shape}. Se continuará, pero la limpieza de ROIs puede ser incorrecta.")

        # Limpieza inicial de ROIs
        ts_cleaned = np.delete(ts_raw, rois_to_remove, axis=1)
        log.debug(f"Sujeto {subject_id}: Shape crudo {ts_raw.shape} -> Shape limpio {ts_cleaned.shape}")

        pp_cfg = cfg['preprocessing']
        fs = 1.0 / pp_cfg['tr_seconds']
        ts_filtered = _bandpass_filter(ts_cleaned, pp_cfg['low_cut_hz'], pp_cfg['high_cut_hz'], fs, pp_cfg['filter_order'])
        
        # Detectar y ADVERTIR sobre ROIs con varianza cero, PERO NO ELIMINARLOS
        std_devs = np.std(ts_filtered, axis=0)
        zero_var_mask = std_devs < 1e-8
        
        if np.any(zero_var_mask):
            log.warning(f"Sujeto {subject_id}: Se encontraron {np.sum(zero_var_mask)} ROIs con varianza cero. NO se eliminarán para mantener la dimensión.")
        
        if ts_filtered.shape[1] == 0:
             log.error(f"Sujeto {subject_id}: No quedaron ROIs después de la limpieza inicial.")
             return None

        # StandardScaler manejará correctamente las columnas con varianza cero (las dejará como ceros)
        ts_scaled = StandardScaler().fit_transform(ts_filtered)
        
        ts_final = _homogenize_length(ts_scaled, pp_cfg['target_length_tps'])
        
        return ts_final.astype(np.float32)

    except Exception as e:
        log.error(f"Sujeto {subject_id}: Fallo el preprocesamiento - {e}", exc_info=True)
        return None
