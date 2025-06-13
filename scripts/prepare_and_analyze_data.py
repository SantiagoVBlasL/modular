#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_and_analyze_data.py

Script de doble propósito para la fase de pre-modelado de la tesis.
1.  **Análisis Exploratorio:** Realiza un análisis completo sobre el 100% de los
    datos para entender sus propiedades, generar figuras y tablas.
2.  **Preparación de Datos para Cross-Validation:** Toma el conjunto de datos completo,
    consolida las etiquetas de MCI, lo limpia y guarda un único set de artefactos
    (tensores, características sin escalar, y etiquetas) listo para ser usado en
    el script de entrenamiento con validación cruzada.

Este script es el paso PREVIO y NECESARIO antes de ejecutar 'train_vae_classifier.py'.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif # <--- Importación añadida
import umap
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import joblib

# --- Añadir el directorio raíz del proyecto al path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Configuración del Estilo y Logging ---
sns.set_theme(style='whitegrid', context='notebook', palette='viridis')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

# --- 1. Funciones de Carga y Limpieza de Datos ---
def load_and_consolidate_data(run_dir_name: str, meta_file: str) -> Optional[Tuple[pd.DataFrame, Path, Dict]]:
    """Carga el reporte final, lo une con metadatos, maneja columnas duplicadas y consolida etiquetas MCI."""
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
    original_counts = merged_df['ResearchGroup'].value_counts()
    log.info(f"Distribución de grupos original: \n{original_counts}")
    merged_df['ResearchGroup'] = merged_df['ResearchGroup'].replace(mci_labels_to_consolidate, 'MCI')
    consolidated_counts = merged_df['ResearchGroup'].value_counts()
    log.info(f"Distribución de grupos consolidada: \n{consolidated_counts}")

    return merged_df, run_path, cfg

# --- Análisis de Canales por Teoría de la Información ---

def rank_channels_by_information_gain(df: pd.DataFrame, cfg: dict, save_dir: Path):
    """
    Calcula la Información Mutua entre cada canal de conectividad y las etiquetas de grupo
    para rankear los canales por su relevancia para la clasificación.
    """
    log.info("Iniciando ranking de canales por ganancia de información (Mutual Information)...")
    
    channel_names = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    
    # Cargar todos los tensores y prepararlos
    df_filtered = df.dropna(subset=['tensor_path', 'ResearchGroup']).reset_index(drop=True)
    all_tensors = np.stack([np.load(p) for p in tqdm(df_filtered['tensor_path'], desc="Cargando tensores para análisis de IM")])
    
    # Preparar las etiquetas de clase (y)
    le = LabelEncoder()
    y = le.fit_transform(df_filtered['ResearchGroup'])
    
    channel_information = []

    for i, ch_name in enumerate(tqdm(channel_names, desc="Analizando Canales")):
        # Extraer las matrices aplanadas para el canal actual
        n_rois = all_tensors.shape[2]
        iu_indices = np.triu_indices(n_rois, k=1)
        X_channel = all_tensors[:, i, :, :][:, iu_indices[0], iu_indices[1]]
        
        mi_scores = mutual_info_classif(X_channel, y, random_state=42)
        
        total_information_gain = np.sum(mi_scores)
        channel_information.append({'channel': ch_name, 'total_information_gain': total_information_gain})

    ranking_df = pd.DataFrame(channel_information).sort_values('total_information_gain', ascending=False)
    ranking_path = save_dir / "ranking_canales_por_informacion_mutua.csv"
    ranking_df.to_csv(ranking_path, index=False)
    log.info(f"Ranking de canales guardado en: {ranking_path}")

    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_information_gain', y='channel', data=ranking_df, palette='plasma')
    plt.title('Ranking de Relevancia de Canales de Conectividad', fontsize=16)
    plt.xlabel('Ganancia de Información Total (Información Mutua con Etiquetas de Grupo)')
    plt.ylabel('Canal de Conectividad')
    plt.tight_layout()
    plt.savefig(save_dir / 'ranking_canales_por_informacion_mutua.png', dpi=200)
    plt.close()
    
    del all_tensors; gc.collect()

# --- 2. Funciones de Análisis Exploratorio (Se ejecutan sobre TODOS los datos) ---

def plot_channel_data_distribution(df: pd.DataFrame, cfg: dict, save_dir: Path):
    """
    Analiza y grafica la distribución de los valores de conectividad para cada canal,
    separado por los grupos 'CN' y 'AD', para informar la elección de la función
    de activación del decoder.
    """
    log.info("Iniciando análisis de distribución de datos por canal para CN y AD...")
    
    df_filtered = df[df['ResearchGroup'].isin(['CN', 'AD'])].copy()
    if df_filtered.empty:
        log.warning("No se encontraron sujetos CN o AD para el análisis de distribución. Omitiendo.")
        return

    channel_names = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    n_channels = len(channel_names)
    
    tensors_cache = {
        row['subject_id']: np.load(row['tensor_path']) 
        for _, row in df_filtered.iterrows() if pd.notna(row['tensor_path'])
    }

    n_cols = min(3, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False, constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle('Distribución de Valores de Conectividad (Fuera de la Diagonal) por Canal y Grupo', fontsize=18, y=1.03)

    for i, ch_name in enumerate(tqdm(channel_names, desc="Analizando distribución de canales")):
        ax = axes[i]
        cn_values, ad_values = [], []

        for subject_id, group in df_filtered[['subject_id', 'ResearchGroup']].values:
            if subject_id in tensors_cache:
                tensor = tensors_cache[subject_id]
                matrix = tensor[i, :, :]
                off_diagonal_values = matrix[~np.eye(matrix.shape[0], dtype=bool)]
                
                if group == 'CN':
                    cn_values.extend(off_diagonal_values)
                elif group == 'AD':
                    ad_values.extend(off_diagonal_values)
        
        sns.histplot(cn_values, color='blue', label='CN (Sanos)', ax=ax, stat='density', bins=50, kde=True)
        sns.histplot(ad_values, color='red', label='AD (Alzheimer)', ax=ax, stat='density', bins=50, kde=True)
        
        ax.set_title(ch_name.replace("_", " ").title())
        ax.set_xlabel("Valor de Conectividad")
        ax.set_ylabel("Densidad")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.savefig(save_dir / 'distribucion_valores_por_canal_grupo.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Gráfico de distribución de datos por canal guardado en: {save_dir / 'distribucion_valores_por_canal_grupo.png'}")
    
    del tensors_cache; gc.collect()


def plot_group_connectome_analysis(df: pd.DataFrame, cfg: dict, save_dir: Path):
    """Calcula y grafica las matrices promedio y de diferencia entre grupos."""
    log.info("Iniciando análisis de conectomas por grupo (promedios y diferencias)...")
    groups = ['CN', 'MCI', 'AD']
    channel_names = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    n_channels = len(channel_names)
    
    tensors_cache = {row['subject_id']: np.load(row['tensor_path']) for _, row in df.iterrows() if pd.notna(row['tensor_path'])}
    
    group_means = {}
    for group in groups:
        subject_ids = df[df['ResearchGroup'] == group]['subject_id']
        group_tensors = [tensors_cache[sid] for sid in subject_ids if sid in tensors_cache]
        if group_tensors:
            group_means[group] = np.mean(np.stack(group_tensors), axis=0)

    fig, axes = plt.subplots(n_channels, len(groups), figsize=(len(groups) * 5, n_channels * 4.5), squeeze=False, constrained_layout=True)
    fig.suptitle('Matrices de Conectividad Promedio por Grupo', fontsize=20, y=1.03)
    for i, ch_name in enumerate(channel_names):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            if group in group_means:
                matrix = group_means[group][i, :, :]
                im = ax.imshow(matrix, cmap='viridis', interpolation='none')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'{group} - {ch_name.replace("_", " ").title()}')
    plt.savefig(save_dir / 'connectomas_promedio_por_grupo.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    if 'AD' in group_means and 'CN' in group_means:
        fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 5, 4.5), squeeze=False, constrained_layout=True)
        fig.suptitle('Diferencia de Conectividad (AD - CN)', fontsize=20, y=1.03)
        diff_tensor = group_means['AD'] - group_means['CN']
        
        for i, ch_name in enumerate(channel_names):
            ax = axes[0, i]
            diff_matrix = diff_tensor[i, :, :]
            vmax = np.percentile(np.abs(diff_matrix), 99)
            if vmax > 0:
                im = ax.imshow(diff_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='none')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(diff_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(ch_name.replace("_", " ").title())
        plt.savefig(save_dir / 'connectomas_diferencia_AD-CN.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    del tensors_cache, group_means; gc.collect()

def plot_feature_distributions(df: pd.DataFrame, feature_cols: list, save_dir: Path):
    log.info("Generando gráficos de distribución de características (Violin Plots)...")
    n_features = len(feature_cols)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), constrained_layout=True)
    axes = axes.flatten()
    order = ['CN', 'MCI', 'AD']
    for i, col in enumerate(feature_cols):
        sns.violinplot(data=df, x='ResearchGroup', y=col, ax=axes[i], order=order, inner='quartile', cut=0)
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_xlabel(None); axes[i].set_ylabel("Valor")
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    plt.savefig(save_dir / 'distribucion_caracteristicas_violin.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_feature_clustermap(df: pd.DataFrame, feature_cols: list, save_dir: Path):
    log.info("Generando clustermap de correlación de características...")
    corr_matrix = df[feature_cols].corr(method='spearman')
    g = sns.clustermap(corr_matrix, cmap='viridis', annot=False, figsize=(16, 16))
    g.fig.suptitle('Clustermap de Correlación (Spearman) entre Características', fontsize=20, y=1.02)
    plt.savefig(save_dir / 'clustermap_correlacion_caracteristicas.png', dpi=200, bbox_inches='tight')
    plt.close()

def perform_statistical_tests(df: pd.DataFrame, feature_cols: List[str], save_dir: Path):
    log.info("Realizando pruebas estadísticas (Kruskal-Wallis) para comparar grupos...")
    results = []
    order = ['CN', 'MCI', 'AD']
    for feature in tqdm(feature_cols, desc="Probando características"):
        groups_data = [df[df['ResearchGroup'] == g][feature].dropna() for g in order]
        if any(len(g) < 3 for g in groups_data): continue
        stat, p_value = kruskal(*groups_data)
        dunn_results = posthoc_dunn(df, val_col=feature, group_col='ResearchGroup', p_adjust='holm') if p_value < 0.05 else None
        results.append({
            'feature': feature,
            'p_value': p_value,
            'dunn_CN_vs_MCI': dunn_results.loc['CN', 'MCI'] if dunn_results is not None and 'MCI' in dunn_results.columns and 'CN' in dunn_results.index else 'N/A',
            'dunn_CN_vs_AD': dunn_results.loc['CN', 'AD'] if dunn_results is not None else 'N/A',
            'dunn_MCI_vs_AD': dunn_results.loc['MCI', 'AD'] if dunn_results is not None and 'MCI' in dunn_results.index and 'AD' in dunn_results.columns else 'N/A',
        })
    pd.DataFrame(results).sort_values('p_value').to_csv(save_dir / 'analisis_estadistico_caracteristicas.csv', index=False)
    log.info(f"Resultados estadísticos guardados en {save_dir / 'analisis_estadistico_caracteristicas.csv'}")

def plot_exploratory_feature_importance(df: pd.DataFrame, feature_cols: list, save_dir: Path):
    log.info("Estimando importancia de características (Exploratorio, sobre todos los datos)...")
    df_clean = df[['ResearchGroup'] + feature_cols].dropna()
    X = df_clean[feature_cols]
    y = df_clean['ResearchGroup']
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced').fit(X_scaled, y)
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20), palette='viridis')
    plt.title('Top 20 Características más Importantes (Exploratorio)')
    plt.xlabel("Importancia (Reducción de Impureza Gini)")
    plt.ylabel("Característica")
    plt.tight_layout()
    plt.savefig(save_dir / "importancia_caracteristicas_exploratorio.png", dpi=200)
    plt.close()
    importance_df.to_csv(save_dir / "importancia_caracteristicas_exploratorio.csv", index=False)

def plot_latent_space_projections(df: pd.DataFrame, feature_cols: List[str], save_dir: Path):
    log.info("Generando proyecciones en espacio latente (PCA y UMAP)...")
    df_clean = df[['ResearchGroup'] + feature_cols].dropna()
    X = df_clean[feature_cols]
    y = df_clean['ResearchGroup']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    order = ['CN', 'MCI', 'AD']
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=ax[0], style=y, s=80, hue_order=order)
    ax[0].set_title(f'PCA (Varianza Explicada: {pca.explained_variance_ratio_.sum():.2%})')
    ax[0].set_xlabel('Componente Principal 1')
    ax[0].set_ylabel('Componente Principal 2')
    ax[0].legend(title='Grupo')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, ax=ax[1], style=y, s=80, hue_order=order)
    ax[1].set_title('UMAP (n_neighbors=15, min_dist=0.1)')
    ax[1].set_xlabel('Dimensión UMAP 1')
    ax[1].set_ylabel('Dimensión UMAP 2')
    ax[1].legend(title='Grupo')
    plt.tight_layout()
    plt.savefig(save_dir / 'proyecciones_latentes_pca_umap.png', dpi=200)
    plt.close()

# --- 3. Función de Preparación de Datos para Cross-Validation ---
def export_dataset_for_cv(df: pd.DataFrame, feature_cols: list, run_path: Path):
    """
    Limpia, consolida y guarda el dataset completo para ser usado en una
    validación cruzada en el script de entrenamiento.
    """
    log.info("--- Iniciando Preparación de Datos para Cross-Validation ---")
    
    cols_for_modeling = ['subject_id', 'tensor_path', 'ResearchGroup', 'Sex', 'Age'] + feature_cols
    df_model = df[cols_for_modeling].copy()
    
    df_model.dropna(inplace=True)
    df_model.reset_index(drop=True, inplace=True)
    log.info(f"Se exportarán {len(df_model)} sujetos con datos completos para la CV.")
    
    save_dir = run_path / "data_for_cv"
    save_dir.mkdir(exist_ok=True)
    
    key_df_path = save_dir / "cv_subjects_key.csv"
    df_model.to_csv(key_df_path, index=False)
    log.info(f"Archivo clave de sujetos para CV guardado en: {key_df_path}")
    
    all_tensors = np.stack([np.load(p) for p in tqdm(df_model['tensor_path'], desc="Apilando tensores")])
    all_features_unscaled = df_model[feature_cols].values
    
    np.save(save_dir / "cv_all_tensors.npy", all_tensors)
    np.save(save_dir / "cv_all_features_unscaled.npy", all_features_unscaled)
    
    log.info(f"Artefactos Numpy guardados en: {save_dir}")
    log.info(f"Shape de tensores: {all_tensors.shape}")
    log.info(f"Shape de características: {all_features_unscaled.shape}")
    log.info("--- Finalizada Preparación de Datos ---")

def main():
    parser = argparse.ArgumentParser(description="Análisis Exploratorio y Preparación de Datos para CV.")
    parser.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a analizar.")
    parser.add_argument('--meta', default='SubjectsData_Schaefer400.csv', help="CSV con metadatos de los sujetos.")
    args = parser.parse_args()

    df, run_path, cfg = load_and_consolidate_data(args.run, args.meta)
    if df is None: return

    analysis_dir = run_path / "analisis_tesis_exploratorio"
    analysis_dir.mkdir(exist_ok=True)
    
    feature_cols = [c for c in df.columns if c.startswith(('topo_', 'hmm_')) and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in feature_cols if df[c].std(skipna=True) > 1e-6]

    # --- Ejecutar los dos pasos principales ---
    
    # 1. Análisis Exploratorio
    log.info("--- Paso 1: Iniciando Análisis Exploratorio ---")
    plot_group_connectome_analysis(df, cfg, analysis_dir)
    plot_feature_distributions(df, feature_cols, analysis_dir)
    plot_feature_clustermap(df, feature_cols, analysis_dir)
    perform_statistical_tests(df, feature_cols, analysis_dir)
    plot_exploratory_feature_importance(df, feature_cols, analysis_dir)
    plot_latent_space_projections(df, feature_cols, analysis_dir)
    
    # --- LLAMADAS A LAS NUEVAS FUNCIONES ---
    rank_channels_by_information_gain(df, cfg, analysis_dir)
    plot_channel_data_distribution(df, cfg, analysis_dir) # <--- ANÁLISIS AÑADIDO
    
    # 2. Preparación y Exportación de Datos para Cross-Validation
    log.info("--- Paso 2: Iniciando Preparación de Datos para CV ---")
    export_dataset_for_cv(df, feature_cols, run_path)
    
    log.info("Proceso completado.")

if __name__ == '__main__':
    main()