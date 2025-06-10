# scripts/debug_single_subject.py
"""
Script de depuración para ejecutar el pipeline de conectividad completo
para un único sujeto y visualizar las matrices resultantes.
"""
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fmri_features import data_loader, connectome_generator

# --- CONFIGURACIÓN ---
# Elige el sujeto que quieres probar. Asegúrate de que pasó el QC.
SUBJECT_ID_TO_DEBUG = '002_S_0295' 
CONFIG_PATH = project_root / 'config_connectivity.yaml'
OUTPUT_DIR = project_root / 'debug_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


def main():
    log.info(f"--- INICIANDO PRUEBA DE PIPELINE PARA SUJETO: {SUBJECT_ID_TO_DEBUG} ---")

    # 1. Cargar configuración
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    log.info(f"Configuración cargada desde: {CONFIG_PATH}")

    # 2. Cargar y preprocesar la serie temporal del sujeto
    log.info("Paso 1: Cargando y preprocesando la serie temporal...")
    rois_to_remove = data_loader._get_rois_to_remove(cfg)
    ts_data = data_loader.load_and_preprocess_ts(SUBJECT_ID_TO_DEBUG, cfg, rois_to_remove)

    if ts_data is None:
        log.error("Fallo en la carga de datos. El script no puede continuar.")
        return
    log.info(f"Serie temporal preprocesada. Shape final: {ts_data.shape}")

    # 3. Generar el tensor de conectividad
    log.info("Paso 2: Generando el tensor de conectividad multi-canal...")
    tensor = connectome_generator.generate_connectivity_tensor(ts_data, cfg, SUBJECT_ID_TO_DEBUG)

    if tensor is None:
        log.error("Fallo en la generación del tensor. El script no puede continuar.")
        return
    log.info(f"Tensor de conectividad generado con éxito. Shape final: {tensor.shape}")

    # 4. Visualizar las matrices de conectividad
    log.info("Paso 3: Visualizando las matrices de conectividad...")
    
    channel_names = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    n_channels = tensor.shape[0]

    fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 5, 5.5), constrained_layout=True)
    if n_channels == 1:
        axes = [axes]
        
    fig.suptitle(f'Matrices de Conectividad para Sujeto: {SUBJECT_ID_TO_DEBUG}', fontsize=20)

    for i in range(n_channels):
        ax = axes[i]
        matrix = tensor[i, :, :]
        
        # Ajustar el rango de colores para una mejor visualización
        if 'pearson' in channel_names[i] or 'tangent' in channel_names[i] or 'lasso' in channel_names[i]:
            vmax = np.percentile(np.abs(matrix), 99) # Usar percentil para evitar que outliers dominen la escala
            vmin = -vmax
            cmap = 'viridis'
        else: # Para wavelet coherence, que es >= 0
            vmax = np.percentile(matrix, 99)
            vmin = 0
            cmap = 'inferno'

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(channel_names[i], fontsize=14)
        ax.set_xlabel("Índice de ROI")
        if i == 0:
            ax.set_ylabel("Índice de ROI")
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Guardar la figura
    output_path = OUTPUT_DIR / f"CONNECTIVITY_CHECK_{SUBJECT_ID_TO_DEBUG}.png"
    plt.savefig(output_path, dpi=200)
    log.info(f"Gráfico de visualización guardado en: {output_path}")
    
    # Mostrar la figura
    plt.show()

    log.info(f"--- PRUEBA COMPLETADA PARA SUJETO: {SUBJECT_ID_TO_DEBUG} ---")


if __name__ == '__main__':
    main()