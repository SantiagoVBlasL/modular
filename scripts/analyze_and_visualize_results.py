# scripts/analyze_and_visualize_results.py
"""
Script de Análisis Exploratorio Exhaustivo para Tesis Doctoral

Este script realiza un análisis exploratorio completo de los resultados del pipeline
de conectividad. Su objetivo es la inspección de datos, la generación de
visualizaciones para publicación y la creación de un conjunto de datos consolidado
y limpio, listo para futuras etapas de modelado.

Funcionalidades Clave:
1.  Análisis de Conectomas por Grupo:
    - Calcula y grafica las matrices de conectividad PROMEDIO por grupo (CN, MCI, AD).
    - Calcula y grafica las matrices de DIFERENCIA (ej. AD - CN) con escalas de
      color optimizadas por canal para resaltar alteraciones.
2.  Análisis Estadístico y de Características:
    - Realiza tests de Kruskal-Wallis para comparar características entre grupos.
    - Estima la importancia de características de forma exploratoria con RandomForest.
3.  Análisis Exploratorio y Visualización:
    - Genera un clustermap de correlación para entender la estructura de las características.
    - Muestra la distribución de cada característica por grupo mediante violin plots.
    - Proyecta los datos en 2D usando PCA y UMAP para visualizar la separabilidad de los grupos.
4.  Exportación de Datos Consolidados:
    - Guarda un único archivo CSV con todos los sujetos, sus características procesadas y
      metadatos relevantes, sirviendo como un "datasheet" final para la etapa de modelado.
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
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# --- Añadir el directorio raíz del proyecto al path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Configuración del Estilo y Logging ---
sns.set_theme(style='whitegrid', context='notebook', palette='viridis')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)


# --- 1. CARGA Y CONSOLIDACIÓN DE DATOS ---

def load_and_consolidate_data(run_dir_name: str, meta_file: str) -> Optional[Tuple[pd.DataFrame, Dict, Path]]:
    """Carga el reporte final, lo une con metadatos y maneja columnas duplicadas."""
    run_path = project_root / 'connectivity_features' / run_dir_name
    report_path = run_path / "salvaged_analysis" / "final_report_with_features.csv"
    config_path = run_path / 'config_used.yaml'

    if not report_path.exists():
        log.error(f"No se encontró el reporte final 'final_report_with_features.csv' en {report_path.parent}.")
        log.error("Por favor, ejecuta primero el script 'salvage_and_analyze.py' para generarlo.")
        return None

    log.info(f"Cargando reporte consolidado desde: {report_path}")
    df = pd.read_csv(report_path)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    meta_df = pd.read_csv(project_root / meta_file)
    df['subject_id'] = df['subject_id'].astype(str)
    meta_df['SubjectID'] = meta_df['SubjectID'].astype(str)
    
    merged_df = pd.merge(df, meta_df.rename(columns={'SubjectID': 'subject_id'}), on='subject_id', how='left')

    cols_x = [c for c in merged_df.columns if c.endswith('_x')]
    for col_x in cols_x:
        base_name = col_x[:-2]
        col_y = f"{base_name}_y"
        if col_y in merged_df.columns:
            merged_df[base_name] = merged_df[col_x].fillna(merged_df[col_y])
            merged_df.drop(columns=[col_x, col_y], inplace=True)

    log.info(f"Datos cargados y consolidados para {len(merged_df)} sujetos.")
    return merged_df, cfg, run_path


# --- 2. ANÁLISIS EXPLORATORIO Y VISUALIZACIÓN ---

def plot_group_connectome_analysis(df: pd.DataFrame, cfg: dict, save_dir: Path):
    """Calcula y grafica las matrices promedio y de diferencia entre grupos."""
    log.info("Análisis de conectomas por grupo (promedios y diferencias)...")
    groups = ['CN', 'MCI', 'AD']
    channel_names = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    n_channels = len(channel_names)
    
    tensors = {row['subject_id']: np.load(row['tensor_path']) for _, row in df.iterrows()}
    
    group_means = {}
    for group in groups:
        subject_ids = df[df['ResearchGroup'] == group]['subject_id']
        group_tensors = [tensors[sid] for sid in subject_ids if sid in tensors]
        if group_tensors:
            group_means[group] = np.mean(np.stack(group_tensors), axis=0)

    # Graficar matrices promedio
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

    # --- CORRECCIÓN DE VISUALIZACIÓN ---
    # Graficar matrices de diferencia con escala de color individual
    if 'AD' in group_means and 'CN' in group_means:
        fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 5, 4.5), squeeze=False, constrained_layout=True)
        fig.suptitle('Diferencia de Conectividad (AD - CN)', fontsize=20, y=1.03)
        diff_tensor = group_means['AD'] - group_means['CN']
        
        for i, ch_name in enumerate(channel_names):
            ax = axes[0, i]
            diff_matrix = diff_tensor[i, :, :]
            # Calcular límite de color simétrico para CADA canal
            vmax = np.percentile(np.abs(diff_matrix), 99) # Usar percentil 99 para robustez a outliers
            if vmax > 0:
                im = ax.imshow(diff_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='none')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else: # Si no hay diferencia, mostrar matriz gris
                ax.imshow(diff_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(ch_name.replace("_", " ").title())

        plt.savefig(save_dir / 'connectomas_diferencia_AD-CN.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    del tensors; gc.collect()

def plot_feature_distributions(df: pd.DataFrame, feature_cols: list, save_dir: Path):
    log.info("Generando gráficos de distribución de características (Violin Plots)...")
    n_features = len(feature_cols)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), constrained_layout=True)
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.violinplot(data=df, x='ResearchGroup', y=col, ax=axes[i], order=['CN', 'MCI', 'AD'], inner='quartile', cut=0)
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


# --- 3. ANÁLISIS ESTADÍSTICO Y DE CARACTERÍSTICAS ---
def perform_statistical_tests(df: pd.DataFrame, feature_cols: List[str], save_dir: Path):
    log.info("Realizando pruebas estadísticas (Kruskal-Wallis) para comparar grupos...")
    results = []
    for feature in tqdm(feature_cols, desc="Probando características"):
        groups_data = [df[df['ResearchGroup'] == g][feature].dropna() for g in ['CN', 'MCI', 'AD']]
        if any(len(g) < 3 for g in groups_data): continue
        
        stat, p_value = kruskal(*groups_data)
        dunn_results = posthoc_dunn(df, val_col=feature, group_col='ResearchGroup', p_adjust='holm') if p_value < 0.05 else None
        
        results.append({
            'feature': feature,
            'p_value': p_value,
            'dunn_CN_vs_MCI': dunn_results.loc['CN', 'MCI'] if dunn_results is not None else 'N/A',
            'dunn_CN_vs_AD': dunn_results.loc['CN', 'AD'] if dunn_results is not None else 'N/A',
            'dunn_MCI_vs_AD': dunn_results.loc['MCI', 'AD'] if dunn_results is not None else 'N/A',
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
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y)
    
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': clf.feature_importances_}) \
        .sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20), palette='viridis')
    plt.title('Top 20 Características más Importantes (Exploratorio)')
    plt.xlabel("Importancia (Reducción de Impureza Gini)")
    plt.ylabel("Característica")
    plt.tight_layout()
    plt.savefig(save_dir / "importancia_caracteristicas_exploratorio.png", dpi=200)
    plt.close()
    importance_df.to_csv(save_dir / "importancia_caracteristicas_exploratorio.csv", index=False)


# --- 4. VISUALIZACIÓN EN ESPACIO LATENTE (EXPLORATORIO) ---
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
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=ax[0], style=y, s=80, hue_order=['CN', 'MCI', 'AD'])
    ax[0].set_title(f'PCA (Varianza Explicada: {pca.explained_variance_ratio_.sum():.2%})')
    ax[0].set_xlabel('Componente Principal 1')
    ax[0].set_ylabel('Componente Principal 2')
    ax[0].legend(title='Grupo')

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, ax=ax[1], style=y, s=80, hue_order=['CN', 'MCI', 'AD'])
    ax[1].set_title('UMAP (n_neighbors=15, min_dist=0.1)')
    ax[1].set_xlabel('Dimensión UMAP 1')
    ax[1].set_ylabel('Dimensión UMAP 2')
    ax[1].legend(title='Grupo')

    plt.tight_layout()
    plt.savefig(save_dir / 'proyecciones_latentes_pca_umap.png', dpi=200)
    plt.close()


# --- 5. EXPORTACIÓN DE DATOS CONSOLIDADOS ---
def export_full_dataset(df: pd.DataFrame, feature_cols: list, run_path: Path):
    """Guarda el DataFrame final y limpio para uso futuro."""
    log.info("Exportando el conjunto de datos completo y limpio...")
    
    cols_to_keep = ['subject_id', 'ResearchGroup', 'Age', 'Sex', 'PTEDUCAT', 'MMSE'] + feature_cols
    df_export = df[cols_to_keep].copy()
    
    save_path = run_path / "analisis_tesis_exploratorio" / "final_dataset_for_modeling.csv"
    df_export.to_csv(save_path, index=False)
    log.info(f"Conjunto de datos consolidado guardado en: {save_path}")


# --- FUNCIÓN PRINCIPAL DE ORQUESTACIÓN ---
def main():
    parser = argparse.ArgumentParser(description="Análisis Exploratorio de Datos para Tesis.")
    parser.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a analizar.")
    parser.add_argument('--meta', default='SubjectsData_Schaefer400.csv', help="CSV con metadatos.")
    args = parser.parse_args()

    run_data = load_and_consolidate_data(args.run, args.meta)
    if run_data is None: return
    df, cfg, run_path = run_data

    analysis_dir = run_path / "analisis_tesis_exploratorio"
    analysis_dir.mkdir(exist_ok=True)
    
    feature_cols = [c for c in df.columns if c.startswith(('topo_', 'hmm_')) and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in feature_cols if df[c].std(skipna=True) > 1e-6]

    log.info("--- INICIANDO PIPELINE DE ANÁLISIS EXPLORATORIO ---")
    
    plot_group_connectome_analysis(df, cfg, analysis_dir)
    plot_feature_distributions(df, feature_cols, analysis_dir)
    plot_feature_clustermap(df, feature_cols, analysis_dir)
    perform_statistical_tests(df, feature_cols, analysis_dir)
    plot_exploratory_feature_importance(df, feature_cols, analysis_dir)
    plot_latent_space_projections(df, feature_cols, analysis_dir)
    export_full_dataset(df, feature_cols, run_path)
    
    log.info(f"--- ANÁLISIS EXPLORATORIO COMPLETO ---")
    log.info(f"Todos los resultados se han guardado en: {analysis_dir}")

if __name__ == '__main__':
    main()