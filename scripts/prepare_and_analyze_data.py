#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import yaml
import logging
import argparse
import gc
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
import umap
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from joblib import dump
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from training import data_handling

sns.set_theme(style='whitegrid', context='notebook', palette='viridis')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

def load_and_consolidate_data(run_dir_name: str, meta_file: str) -> Optional[Tuple[pd.DataFrame, Path, Dict]]:
    run_path = project_root / 'connectivity_features' / run_dir_name
    report_path = run_path / "salvaged_analysis" / "final_report_with_features.csv"
    if not report_path.exists():
        log.error(f"No se encontró el reporte final en {report_path.parent}.")
        return None, None, None
        
    log.info(f"Cargando reporte consolidado desde: {report_path}")
    df = pd.read_csv(report_path)
    
    with open(run_path / 'config_used.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    meta_df = pd.read_csv(project_root / meta_file)
    df['subject_id'] = df['subject_id'].astype(str)
    meta_df['SubjectID'] = meta_df['SubjectID'].astype(str)
    
    merged_df = pd.merge(df, meta_df.rename(columns={'SubjectID': 'subject_id'}), on='subject_id', how='left')
    cols_x = [c for c in merged_df.columns if c.endswith('_x')]
    for col_x in cols_x:
        base_name, col_y = col_x[:-2], f"{col_x[:-2]}_y"
        if col_y in merged_df.columns:
            merged_df[base_name] = merged_df[col_x].fillna(merged_df[col_y])
            merged_df.drop(columns=[col_x, col_y], inplace=True)

    log.info(f"Datos cargados y consolidados para {len(merged_df)} sujetos.")
    
    mci_labels_to_consolidate = ['EMCI', 'LMCI', 'MCI']
    merged_df['ResearchGroup'] = merged_df['ResearchGroup'].replace(mci_labels_to_consolidate, 'MCI')
    log.info(f"Distribución de grupos consolidada: \n{merged_df['ResearchGroup'].value_counts()}")

    return merged_df, run_path, cfg

def plot_channel_data_distribution(df: pd.DataFrame, cfg: dict, save_dir: Path):
    pass

def rank_channels_by_information_gain(df: pd.DataFrame, cfg: dict, save_dir: Path, all_tensors: np.ndarray):
    pass

def prepare_and_serialize_artifacts(df: pd.DataFrame, run_path: Path, cfg: Dict):
    log.info("--- Iniciando Preparación y Serialización de Artefactos para CV ---")
    
    feature_cols = [c for c in df.columns if c.startswith(('topo_', 'hmm_')) and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in feature_cols if df[c].std(skipna=True) > 1e-6]

    cols_for_modeling = ['subject_id', 'tensor_path', 'ResearchGroup', 'Sex', 'Age'] + feature_cols
    df_model = df[cols_for_modeling].copy().dropna().reset_index(drop=True)
    
    log.info(f"Se prepararán y guardarán {len(df_model)} sujetos con datos completos.")
    
    save_dir = run_path / "data_for_cv"
    save_dir.mkdir(exist_ok=True)
    
    all_tensors_raw = np.stack([np.load(p) for p in tqdm(df_model['tensor_path'], desc="Apilando tensores crudos")])
    all_scalar_features_unscaled = df_model[feature_cols].values
    
    tensors_processed, p99_5_values = data_handling.preprocess_tensors_robustly(all_tensors_raw, return_scalers=True)
    
    np.save(save_dir / "tensors_preprocessed.npy", tensors_processed)
    dump(p99_5_values, save_dir / "p99_5_per_channel.pkl")
    
    tensor_hash = hashlib.sha256(tensors_processed.tobytes()).hexdigest()

    log.info("Ajustando y serializando transformadores (scalers/encoders)...")
    
    scalar_scaler = StandardScaler().fit(all_scalar_features_unscaled)
    dump(scalar_scaler, save_dir / "scalar_features_scaler.pkl")

    age_scaler = StandardScaler().fit(df_model[['Age']])
    dump(age_scaler, save_dir / "age_scaler.pkl")
    
    sex_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df_model[['Sex']])
    dump(sex_encoder, save_dir / "sex_encoder.pkl")
    
    np.save(save_dir / "features_unscaled.npy", all_scalar_features_unscaled)
    df_model[['subject_id', 'ResearchGroup', 'Sex', 'Age']].to_csv(save_dir / "cv_subjects_key.csv", index=False)
    
    channels_to_use_config = cfg.get('channels_to_use', 'all')
    meta_info = {
        'run_id': run_path.name,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'n_subjects': int(tensors_processed.shape[0]),
        'tensor_shape': list(tensors_processed.shape),
        'scalar_features_shape': list(all_scalar_features_unscaled.shape),
        'tensor_sha256': tensor_hash,
        'channels_used_from_config': channels_to_use_config,
        'recommended_final_activation': 'tanh'
    }
    
    with open(save_dir / "meta.yaml", 'w') as f:
        yaml.dump(meta_info, f, default_flow_style=False)
        
    log.info(f"Artefactos de datos y transformadores guardados exitosamente en: {save_dir}")
    
    with open(save_dir / "README.md", "w") as f:
        f.write("# Artefacto de Datos para Entrenamiento\n\n")
        f.write(f"Generado en: `{meta_info['timestamp_utc']}`\n\n")
        f.write("Este directorio contiene todos los datos pre-procesados y artefactos necesarios para ejecutar la Etapa 4 (entrenamiento) de forma reproducible.\n\n")
        f.write("## Contenido:\n\n")
        f.write(f"- **Sujetos:** {meta_info['n_subjects']}\n")
        f.write(f"- **Shape de Tensores:** `{meta_info['tensor_shape']}`\n")
        f.write(f"- **Hash de Tensores (SHA-256):** `{meta_info['tensor_sha256']}`\n")
        f.write(f"- **Activación recomendada:** `{meta_info['recommended_final_activation']}`\n\n")
        f.write("Los archivos `.pkl` contienen los objetos `scikit-learn` ajustados (fitted) para la transformación de datos.\n")

    return all_tensors_raw

def main():
    parser = argparse.ArgumentParser(description="Análisis Exploratorio y Preparación de Artefactos de Datos.")
    parser.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a analizar.")
    parser.add_argument('--meta', default='SubjectsData_Schaefer400.csv', help="CSV con metadatos de los sujetos.")
    parser.add_argument('--skip_plots', action='store_true', help="Omitir la generación de gráficos exploratorios.")
    args = parser.parse_args()

    df, run_path, cfg = load_and_consolidate_data(args.run, args.meta)
    if df is None: return

    log.info("--- Iniciando Etapa 3.1: Preparación y Serialización de Artefactos ---")
    all_tensors_raw = prepare_and_serialize_artifacts(df, run_path, cfg)
    
    if not args.skip_plots:
        log.info("--- Iniciando Etapa 3.2: Análisis Exploratorio (sobre datos crudos) ---")
        analysis_dir = run_path / "analisis_tesis_exploratorio"
        analysis_dir.mkdir(exist_ok=True)
        
        rank_channels_by_information_gain(df, cfg, analysis_dir, all_tensors_raw)
        plot_channel_data_distribution(df, cfg, analysis_dir)

    log.info("Proceso completado.")
    gc.collect()

if __name__ == '__main__':
    main()
