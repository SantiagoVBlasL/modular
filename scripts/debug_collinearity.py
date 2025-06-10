# scripts/debug_collinearity.py
"""
Script de depuración avanzado para visualizar la multicolinealidad
en los datos de un único sujeto mediante heatmaps de correlación.
"""
import sys
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import List, Dict

# Asegúrate de tener plotly instalado: pip install plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fmri_features import data_loader

# --- Configuración ---
SUBJECT_ID_TO_DEBUG = '002_S_0413' # Cambia esto por cualquier sujeto que esté fallando
CONFIG_PATH = project_root / 'config_connectivity.yaml'
OUTPUT_DIR = project_root / 'debug_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

def find_perfectly_correlated_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.9999):
    """Encuentra y reporta pares de ROIs con correlación (casi) perfecta."""
    # Evitar duplicados y auto-correlaciones
    corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Encontrar pares que superen el umbral
    to_report = corr_matrix_upper[corr_matrix_upper.abs() > threshold].stack().reset_index()
    to_report.columns = ['ROI_1', 'ROI_2', 'Correlation']
    return to_report

def main():
    log.info(f"--- INICIANDO DEPURACIÓN DE COLINEALIDAD PARA: {SUBJECT_ID_TO_DEBUG} ---")

    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    rois_to_remove = data_loader._get_rois_to_remove(cfg)
    ts_final = data_loader.load_and_preprocess_ts(SUBJECT_ID_TO_DEBUG, cfg, rois_to_remove)

    if ts_final is None:
        log.error("El preprocesamiento falló. Abortando.")
        return

    log.info(f"Datos preprocesados cargados. Shape: {ts_final.shape}")

    log.info("Calculando matriz de correlación de Pearson...")
    corr_matrix = pd.DataFrame(ts_final).corr()

    problematic_pairs = find_perfectly_correlated_pairs(corr_matrix)
    print("-" * 70)
    if not problematic_pairs.empty:
        log.warning(f"¡EVIDENCIA ENCONTRADA! Se detectaron {len(problematic_pairs)} pares de ROIs con multicolinealidad perfecta:")
        print(problematic_pairs.to_string(index=False))
    else:
        log.info("No se encontraron pares de ROIs con correlación perfecta. La singularidad puede deberse a otras razones más complejas de multicolinealidad.")
    print("-" * 70)

    # --- VISUALIZACIÓN ---
    log.info("Generando heatmap estático con Seaborn...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='viridis', vmin=-1, vmax=1, square=True)
    plt.title(f'Heatmap de Correlación para Sujeto {SUBJECT_ID_TO_DEBUG}', fontsize=16)
    plt.xlabel('Índice de ROI (después de limpieza)')
    plt.ylabel('Índice de ROI (después de limpieza)')
    static_plot_path = OUTPUT_DIR / f"heatmap_static_{SUBJECT_ID_TO_DEBUG}.png"
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Heatmap estático guardado en: {static_plot_path}")

    if PLOTLY_AVAILABLE:
        log.info("Generando heatmap interactivo con Plotly...")
        fig = px.imshow(corr_matrix,
                        color_continuous_scale='Viridis',
                        zmin=-1, zmax=1,
                        title=f'Heatmap Interactivo para Sujeto {SUBJECT_ID_TO_DEBUG}',
                        labels=dict(x="Índice de ROI (después de limpieza)", y="Índice de ROI (después de limpieza)", color="Correlación"))
        fig.update_layout(height=700, width=700)
        interactive_plot_path = OUTPUT_DIR / f"heatmap_interactive_{SUBJECT_ID_TO_DEBUG}.html"
        fig.write_html(interactive_plot_path)
        log.info(f"Heatmap interactivo guardado en: {interactive_plot_path}")
    else:
        log.warning("Plotly no está instalado, se omite el heatmap interactivo. Instálalo con: 'pip install plotly'")

if __name__ == '__main__':
    main()