# qc_bold/report.py
"""
Funciones para generar reportes y visualizaciones, incluyendo un reporte HTML detallado.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import base64
from datetime import datetime

# Se necesita plotly para los gráficos interactivos. Instalar con: pip install plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

log = logging.getLogger(__name__)

def save_dataframe(df: pd.DataFrame, path: Path, filename: str):
    """Guarda un DataFrame a un archivo CSV."""
    filepath = path / filename
    log.info(f"Guardando reporte CSV en: {filepath}")
    df.to_csv(filepath, index=False, float_format='%.3f')

def generate_summary_plots(df: pd.DataFrame, export_dir: Path):
    """
    Genera y guarda un gráfico estático de diagnóstico.
    """
    log.info("Generando gráfico de resumen estático (PNG)...")
    
    plt.figure(figsize=(10, 7))
    plot_data = df.dropna(subset=['univ_outliers_pct', 'mv_outliers_pct'])
    
    if not plot_data.empty:
        sns.scatterplot(
            x='univ_outliers_pct', 
            y='mv_outliers_pct', 
            data=plot_data, 
            alpha=0.7,
            hue='mv_analysis_method',
            size='timepoints',
            sizes=(50, 250),
            legend='auto'
        )
        
    plt.title('% Outliers Univariantes vs. % Outliers Multivariantes')
    plt.xlabel(f"% Outliers Univariantes (Z > {df.attrs.get('z_thresh', 'N/A')})")
    plt.ylabel(f"% Outliers Multivariantes (p < {df.attrs.get('alpha_mahal', 'N/A')})")
    
    if 'max_mv_outliers_pct' in df.attrs:
        plt.axhline(df.attrs['max_mv_outliers_pct'], color='red', ls='--', label=f"Umbral MV ({df.attrs['max_mv_outliers_pct']}%)")
    if 'max_univ_outliers_pct' in df.attrs:
        plt.axvline(df.attrs['max_univ_outliers_pct'], color='blue', ls='--', label=f"Umbral Univ. ({df.attrs['max_univ_outliers_pct']}%)")

    plt.legend(title='Método MV / TPs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plot_path = export_dir / 'plot_scatter_univ_vs_mv_outliers.png'
    plt.savefig(plot_path)
    log.info(f"Gráfico de dispersión guardado en: {plot_path}")
    plt.close()

def generate_html_report(df: pd.DataFrame, config: dict, export_dir: Path):
    """
    Genera un reporte HTML de nivel doctoral, auto-contenido e interactivo.
    """
    if not PLOTLY_AVAILABLE:
        log.warning("Plotly no está instalado. No se puede generar el reporte HTML interactivo.")
        log.warning("Por favor, instálalo con: pip install plotly")
        return

    log.info("Generando reporte HTML interactivo...")
    
    # --- 1. Preparación de Datos y Contenido ---
    
    # Incrustar la imagen estática en Base64 para que el HTML sea auto-contenido
    static_plot_path = export_dir / 'plot_scatter_univ_vs_mv_outliers.png'
    try:
        with open(static_plot_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        static_plot_html = f'<img src="data:image/png;base64,{encoded_string}" alt="Static Scatter Plot" style="width:100%; max-width:800px; margin: auto; display: block;">'
    except FileNotFoundError:
        static_plot_html = "<p>Gráfico estático no encontrado.</p>"

    # Métricas generales
    n_total = len(df)
    n_retained = df['to_discard_overall'].eq(False).sum()
    n_discarded = df['to_discard_overall'].eq(True).sum()
    
    # Desglose de descarte
    discard_cols = [col for col in df.columns if col.startswith('discard_')]
    discard_counts = df[discard_cols].sum().sort_values(ascending=False)
    discard_counts = discard_counts[discard_counts > 0] # Mostrar solo razones con >0 descartes
    
    # --- 2. Creación de Gráficos Interactivos con Plotly ---
    
    # Histograma de distribuciones clave
    hist_cols = ['timepoints', 'nan_pct', 'univ_outliers_pct', 'mv_outliers_pct']
    fig_hist = px.histogram(df.melt(value_vars=hist_cols), x='value', facet_col='variable', 
                            facet_col_wrap=2, histnorm='percent',
                            title="Distribución de Métricas Clave de QC",
                            labels={'value': 'Valor', 'variable': 'Métrica'})
    fig_hist.update_yaxes(title_text='Porcentaje de Sujetos')
    fig_hist.update_xaxes(matches=None) # Permitir ejes X independientes
    hist_html = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')

    # Gráfico de barras de razones de descarte
    fig_discard = px.bar(x=discard_counts.index, y=discard_counts.values,
                         title="Conteo de Sujetos por Razón de Descarte",
                         labels={'x': 'Razón de Descarte', 'y': 'Número de Sujetos'})
    discard_plot_html = fig_discard.to_html(full_html=False, include_plotlyjs='cdn')

    # --- 3. Estilizado de la Tabla de Datos ---
    
    def highlight_discarded(s):
        return ['background-color: #FFCDD2' if s.to_discard_overall else '' for _ in s]

    df_styled = df.style.apply(highlight_discarded, axis=1)\
        .background_gradient(cmap='Reds', subset=['nan_pct', 'null_channels_pct', 'univ_outliers_pct', 'mv_outliers_pct'])\
        .format("{:.2f}", subset=pd.IndexSlice[:, ['nan_pct', 'null_channels_pct', 'univ_outliers_pct', 'mv_outliers_pct']])\
        .set_properties(**{'border': '1px solid #ddd', 'text-align': 'center'})
        
    table_html = df_styled.to_html(escape=False)
    
    # --- 4. Ensamblaje del Reporte HTML ---
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Control de Calidad (QC) de BOLD fMRI</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
            h1 {{ text-align: center; }}
            .summary-cards {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
            .card {{ background: #ecf0f1; border-radius: 8px; padding: 20px; text-align: center; flex-grow: 1; border-left: 5px solid #3498db; }}
            .card h3 {{ border: none; margin-top: 0; }}
            .card .value {{ font-size: 2.5em; font-weight: bold; color: #3498db; }}
            .card-retained .value {{ color: #2ecc71; }}
            .card-retained {{ border-left-color: #2ecc71; }}
            .card-discarded .value {{ color: #e74c3c; }}
            .card-discarded {{ border-left-color: #e74c3c; }}
            .section {{ margin-bottom: 40px; }}
            .plot-container {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; overflow: hidden; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
            th, td {{ padding: 8px 12px; border: 1px solid #ddd; }}
            thead {{ background-color: #34495e; color: white; }}
            footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Reporte de Control de Calidad (QC) de BOLD fMRI</h1>
                <p style="text-align:center;">Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </header>

            <div class="section">
                <h2>Parámetros de la Ejecución</h2>
                <div class="summary-cards">
                     <div class="card"><h3>Umbral Z Univ.</h3><span class="value">{df.attrs.get('z_thresh', 'N/A')}</span></div>
                     <div class="card"><h3>p-valor Mahalanobis</h3><span class="value">{df.attrs.get('alpha_mahal', 'N/A')}</span></div>
                     <div class="card"><h3>Max % Outliers Univ.</h3><span class="value">{df.attrs.get('max_univ_outliers_pct', 'N/A')}%</span></div>
                     <div class="card"><h3>Max % Outliers MV.</h3><span class="value">{df.attrs.get('max_mv_outliers_pct', 'N/A')}%</span></div>
                </div>
            </div>

            <div class="section">
                <h2>Resumen General</h2>
                <div class="summary-cards">
                    <div class="card"><h3>Sujetos Procesados</h3><span class="value">{n_total}</span></div>
                    <div class="card card-retained"><h3>Sujetos Retenidos</h3><span class="value">{n_retained}</span></div>
                    <div class="card card-discarded"><h3>Sujetos Descartados</h3><span class="value">{n_discarded}</span></div>
                </div>
            </div>
            
            <div class="section">
                <h2>Análisis de Descarte y Distribuciones</h2>
                <div class="plot-container">{discard_plot_html}</div>
                <div class="plot-container" style="margin-top: 20px;">{hist_html}</div>
            </div>

            <div class="section">
                <h2>Análisis Univariante vs. Multivariante</h2>
                <div class="plot-container">{static_plot_html}</div>
            </div>

            <div class="section">
                <h2>Tabla de Resultados Detallada</h2>
                <div style="overflow-x:auto;">{table_html}</div>
            </div>

            <footer>
                Reporte generado automáticamente por el pipeline qc_bold.
            </footer>
        </div>
    </body>
    </html>
    """

    # --- 5. Guardar el archivo HTML ---
    report_path = export_dir / 'summary_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    log.info(f"Reporte HTML guardado exitosamente en: {report_path}")