# scripts/verify_exclusion.py
"""
Script de utilidad para verificar que una lista de sujetos problemáticos
conocidos ha sido correctamente excluida del conjunto de datos final
preparado para la validación cruzada.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

# --- Añadir el directorio raíz del proyecto al path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# --- Lista de Sujetos Problemáticos a Verificar ---
# Esta es la lista que proporcionaste. Puedes añadir o quitar IDs aquí.
PROBLEM_SUBJECT_IDS = [
    '114_S_6039',
    '035_S_6953',
    '128_S_2002',
    '031_S_4021',
    '130_S_5231',
    '130_S_6647',
]

def main():
    parser = argparse.ArgumentParser(description="Verifica la exclusión de sujetos problemáticos del dataset final.")
    parser.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a verificar.")
    args = parser.parse_args()

    run_path = project_root / 'connectivity_features' / args.run
    final_key_file = run_path / "data_for_cv" / "cv_subjects_key.csv"

    if not final_key_file.exists():
        log.error(f"No se encontró el archivo clave final en: {final_key_file}")
        log.error("Asegúrate de haber ejecutado 'prepare_and_analyze_data.py' primero.")
        return

    log.info(f"Cargando la lista de sujetos finales desde: {final_key_file}")
    try:
        final_subjects_df = pd.read_csv(final_key_file)
        final_subject_list = set(final_subjects_df['subject_id'].astype(str))
    except Exception as e:
        log.error(f"No se pudo leer el archivo CSV. Error: {e}")
        return
        
    log.info(f"Se encontraron {len(final_subject_list)} sujetos en el conjunto de datos final.")
    log.info("Verificando la exclusión de la lista de sujetos problemáticos...")

    found_subjects = []
    for subject_id in PROBLEM_SUBJECT_IDS:
        if subject_id in final_subject_list:
            found_subjects.append(subject_id)
            
    print("\n--- REPORTE DE VERIFICACIÓN DE EXCLUSIÓN ---\n")
    if not found_subjects:
        log.info("✅ ÉXITO: Todos los sujetos problemáticos de la lista fueron excluidos correctamente del dataset final.")
    else:
        log.error(f"❌ FALLO: Se encontraron {len(found_subjects)} sujetos problemáticos en el dataset final:")
        for subject_id in found_subjects:
            log.error(f"  - {subject_id}")
        log.error("\nEstos sujetos deberían ser investigados. Es posible que hayan superado el QC inicial por error.")
    
    print("\n------------------------------------------\n")

if __name__ == "__main__":
    main()