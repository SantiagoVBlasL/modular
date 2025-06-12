# scripts/analyze_and_visualize_results.py
"""
Script avanzado para el análisis exploratorio y visualización de los resultados
del pipeline de conectividad, diseñado para funcionar mientras el pipeline principal
aún está en ejecución.

Funcionalidades:
1.  Usa argparse para flexibilidad en la selección de la ejecución y metadatos.
2.  Escanea una carpeta de ejecución en busca de tensores .npy ya procesados.
3.  Crea una "galería" de matrices de conectividad para varios sujetos por grupo.
4.  Recalcula características escalares (Topología) para los sujetos encontrados.
5.  Genera gráficos de distribución mejorados (violin plots).
6.  Crea un "clustermap" de correlación para identificar la estructura de las características.
7.  Añade un análisis de PCA (scree plot) como paso previo al VAE.
8.  Mejora la gestión de memoria y la estética de los gráficos.
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
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm # <-- CORRECCIÓN: Importación correcta de tqdm

# --- Añadir el directorio raíz del proyecto al path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fmri_features import feature_extractor

# --- Configuración del Estilo y Logging ---
sns.set_theme(style='whitegrid', context='notebook')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# --- Funciones de Carga de Datos ---
def find_and_load_interim_data(run_dir_name: str, meta_file: str) -> Optional[Tuple[pd.DataFrame, Dict, Path]]:
    """Escanea la carpeta de ejecución, encuentra tensores .npy y los une con metadatos."""
    run_path = project_root / 'connectivity_features' / run_dir_name
    config_path = run_path / 'config_used.yaml'
    
    if not run_path.exists() or not config_path.exists():
        log.error(f"La carpeta de ejecución '{run_dir_name}' o su config.yaml no existen.")
        return None

    log.info(f"Escaneando resultados intermedios desde: {run_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    tensor_files = list(run_path.glob('tensor_*.npy'))
    if not tensor_files:
        log.warning("No se encontraron archivos de tensor .npy. ¿Seguro que el pipeline ya ha procesado algunos sujetos?")
        return None

    log.info(f"Se encontraron {len(tensor_files)} tensores de sujetos ya procesados.")
    
    processed_subjects = [{'subject_id': f.stem.split('tensor_')[-1], 'tensor_path': str(f)} for f in tensor_files]
    processed_df = pd.DataFrame(processed_subjects)

    meta_path = project_root / meta_file
    if not meta_path.exists():
        log.error(f"No se encontró el archivo de metadatos: {meta_path}")
        return None
    meta_df = pd.read_csv(meta_path)
    
    processed_df['subject_id'] = processed_df['subject_id'].astype(str)
    meta_df['SubjectID'] = meta_df['SubjectID'].astype(str)
    meta_df = meta_df[['SubjectID', 'ResearchGroup']].rename(columns={'SubjectID': 'subject_id'})
    full_df = pd.merge(processed_df, meta_df, on='subject_id', how='left')

    return full_df, cfg, run_path

# --- Funciones de Visualización y Análisis ---

def plot_connectivity_gallery(df: pd.DataFrame, cfg: dict, save_dir: Path, n_samples: int = 3):
    """Crea una "galería" de matrices de conectividad para varios sujetos por grupo."""
    log.info(f"Generando galería de conectomas para {n_samples} sujetos por grupo...")
    groups = ['CN', 'MCI', 'AD']
    channel_names = [name for name, enabled in cfg.get('channels', {}).items() if enabled]
    
    for group in groups:
        group_df = df[df['ResearchGroup'] == group]
        if group_df.empty:
            continue
        
        # Usar una semilla para reproducibilidad en el muestreo
        sample_df = group_df.sample(min(n_samples, len(group_df)), random_state=42)
        
        for _, subject_row in sample_df.iterrows():
            subject_id, tensor_path = subject_row['subject_id'], subject_row['tensor_path']
            log.debug(f"Graficando para {subject_id} ({group})")
            
            # Usar mmap_mode para no cargar todo el tensor en RAM de golpe
            tensor = np.load(tensor_path, mmap_mode='r')
            
            n_channels = tensor.shape[0]
            fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 4.5, 5), constrained_layout=True)
            if n_channels == 1: axes = [axes]
            fig.suptitle(f'Conectomas ({group}) - Sujeto: {subject_id}', fontsize=18)

            for i in range(n_channels):
                ax = axes[i]
                matrix = tensor[i, :, :]
                # Escala de color individual y robusta por canal
                vmin, vmax = np.percentile(matrix, [2, 98])
                
                im = axes[i].imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
                axes[i].set_title(channel_names[i].replace('_', ' ').title(), fontsize=12)
                axes[i].set_xticks([]); axes[i].set_yticks([]) # Limpiar ejes
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            plt.savefig(save_dir / f'gallery_matrices_{subject_id}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            # Liberar memoria explícitamente
            del tensor
            gc.collect()

def plot_feature_distributions(df: pd.DataFrame, save_dir: Path):
    """Crea violin plots de las características escalares por grupo de diagnóstico."""
    feature_cols = [col for col in df.columns
                    if col.startswith(('topo_', 'hmm_'))
                    and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols: return
    log.info("Generando gráficos de distribución de características (Violin Plots)...")
    
    n_features = len(feature_cols)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.violinplot(data=df, x='ResearchGroup', y=col, ax=axes[i], order=['CN', 'MCI', 'AD'], inner='quartile', cut=0)
        sns.stripplot(data=df, x='ResearchGroup', y=col, ax=axes[i], color=".25", size=3, jitter=0.2, order=['CN', 'MCI', 'AD'])
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_xlabel(None); axes[i].set_ylabel("Valor")

    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    plt.savefig(save_dir / 'distribucion_caracteristicas_violin.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_feature_clustermap(df: pd.DataFrame, save_dir: Path):
    """Crea un clustermap de la correlación de Spearman para ver la estructura de las características."""
    # 1. solo numéricas y con varianza > 0
    num_cols = [c for c in df.columns
                if c.startswith(('topo_', 'hmm_'))
                and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in num_cols if df[c].std(skipna=True) > 0]
    if len(feature_cols) < 2: return
    log.info("Generando clustermap de correlación de características...")
    
    corr_matrix = df[feature_cols].corr(method='spearman')

    # 2. quitar valores no finitos
    corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan)
    corr_matrix = corr_matrix.dropna(axis=0, how='any').dropna(axis=1, how='any')
    if corr_matrix.shape[0] < 2:
        log.warning("No quedan columnas suficientes (todas constantes o con NaNs).")
        return
    
    g = sns.clustermap(corr_matrix, cmap='viridis', annot=True, fmt='.2f', figsize=(18, 18), annot_kws={"size": 8})
    g.fig.suptitle('Clustermap de Correlación de Spearman entre Características', fontsize=20)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    
    plt.savefig(save_dir / 'clustermap_correlacion_caracteristicas.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_pca_scree(df: pd.DataFrame, save_dir: Path):
    """Realiza un PCA sobre las características y muestra un scree plot."""
    feature_cols = [col for col in df.columns
                    if col.startswith(('topo_', 'hmm_'))
                    and pd.api.types.is_numeric_dtype(df[col])]
    if len(feature_cols) < 2:
        return
    log.info("Generando scree plot de PCA para análisis Pre-VAE...")
    
    features = df[feature_cols].dropna()
    
    # Eliminar Inf/-Inf explícitamente antes de escalar
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    if features.empty or features.shape[1] < 2:
        log.warning("No hay suficientes datos después de eliminar valores infinitos.")
        return

    features_scaled = StandardScaler().fit_transform(features)
    
    pca = PCA().fit(features_scaled)
    
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('Varianza Acumulada Explicada por Componentes Principales')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada Explicada')
    plt.grid(True)
    plt.axhline(y=0.9, color='r', linestyle=':', label='90% Varianza')
    plt.ylim(0, 1.05)
    plt.legend()
    
    plt.savefig(save_dir / 'pca_scree_plot_caracteristicas.png', dpi=200, bbox_inches='tight')
    plt.close()


# --- Función Principal ---

def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    p = argparse.ArgumentParser(description="Análisis exploratorio de resultados del pipeline de conectividad.")
    p.add_argument('--run', required=True, help="Nombre de la carpeta de la corrida a analizar (ej: connectivity_8ch_...).")
    p.add_argument('--meta', default='SubjectsData_Schaefer400.csv', help="Nombre del archivo CSV con metadatos de los sujetos.")
    p.add_argument('--samples', type=int, default=3, help="Número de sujetos por grupo para la galería de matrices.")
    return p.parse_args()

def main(args):
    """Orquesta el análisis y la visualización."""
    
    run_data = find_and_load_interim_data(args.run, args.meta)
    if run_data is None: return
    df_processed, cfg, run_path = run_data

    figs_dir = run_path / "figs_analisis_exploratorio"
    figs_dir.mkdir(exist_ok=True)
    
    log.info("Recalculando características escalares (Topología) para los sujetos procesados...")
    all_features = []
    # CORRECCIÓN: Usar la función tqdm directamente
    for _, row in tqdm(df_processed.iterrows(), total=len(df_processed), desc="Recalculando features"):
        features = {'subject_id': row['subject_id']}
        tensor = np.load(row['tensor_path'], mmap_mode='r')
        
        if cfg.get('features', {}).get('graph_topology'):
            base_matrix = tensor[0, :, :]
            topo_features = feature_extractor.extract_graph_features(base_matrix, row['subject_id'])
            if topo_features:
                features.update(topo_features)
        
        all_features.append(features)
        del tensor
        gc.collect()

    features_df = pd.DataFrame(all_features)
    # Se mantienen las columnas originales y se unen las nuevas características
    df_with_features = pd.merge(df_processed, features_df, on='subject_id', how='left')

    # --- Ejecutar todos los análisis ---
    plot_connectivity_gallery(df_with_features, cfg, figs_dir, args.samples)
    plot_feature_distributions(df_with_features, figs_dir)
    plot_feature_clustermap(df_with_features, figs_dir)
    plot_pca_scree(df_with_features, figs_dir)
    
    log.info(f"¡Análisis completado! Revisa la nueva subcarpeta '{figs_dir.name}' dentro de tu directorio de ejecución.")

if __name__ == '__main__':
    args = parse_args()
    main(args)