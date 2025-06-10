# Pipeline Modular y Reproducible para Control de Calidad y Análisis de Conectividad de Señales BOLD fMRI

Este repositorio alberga el pipeline desarrollado como parte de un proyecto de tesis doctoral en Inteligencia Artificial aplicada a neurociencia. El objetivo central es implementar un flujo de trabajo **modular**, **escalable** y **totalmente reproducible** para:

* **Control de Calidad (QC)** de las series temporales BOLD de fMRI
* **Extracción de Conectomas**: generación de matrices de conectividad funcional

La arquitectura de este proyecto sigue las mejores prácticas de investigación doctoral, priorizando la transparencia, la trazabilidad y la parametrización mediante archivos de configuración en YAML.

## 📂 Estructura del Repositorio

```
├── connectivity_features/         # Salida: matrices de conectividad
├── data/                          # Datos de entrada (no incluidos)
├── debug_outputs/                 # Salida de debugging
├── fmri_features/                 # Scripts para generación de conectomas
│   ├── __init__.py
│   ├── connectome_generator.py
│   └── ...
├── qc_bold/                       # Paquete de QC de fMRI
│   ├── __init__.py
│   ├── io.py
│   ├── multivariate.py
│   ├── report.py
│   └── univariate.py
├── qc_outputs/                    # Reportes de QC generados
├── scripts/                       # Ejecutables del pipeline
│   ├── run_qc_pipeline.py
│   └── run_connectivity_pipeline.py
├── config.yaml                    # Configuración del QC (YAML)
├── config_connectivity.yaml       # Configuración de conectividad (YAML)
├── SubjectsData_Schaefer400.csv   # Metadatos de sujetos
└── pyproject.toml                 # Definición de proyecto y dependencias
```

## ⚙️ Instalación

```bash
git clone <URL-del-repositorio>
cd <nombre-del-repositorio>
python -m venv venv
source venv/bin/activate    # En Windows: venv\Scripts\activate
pip install -e .
```

Esto instalará el paquete `qc_bold` en modo editable, junto con todas las dependencias necesarias.

## 🚀 Uso

### 1. Control de Calidad (QC)

Ejecuta el script dedicado al QC, que leerá los parámetros de `config.yaml`, procesará las series BOLD y generará:

* `report_qc_final.csv`: Métricas de calidad (FD, DVARS, etc.) por sujeto
* `summary_report.html`: Reporte interactivo con visualizaciones para inspección detallada

```bash
python scripts/run_qc_pipeline.py
```

> **Nota doctoral**: Revisa y ajusta umbrales en `config.yaml` para garantizar la rigurosidad en la exclusión de sujetos.

### 2. Extracción de Conectomas

Una vez seleccionado el subconjunto de sujetos que supera los criterios de QC, genera las matrices de conectividad funcional.

```bash
python scripts/run_connectivity_pipeline.py
```

Las salidas se almacenarán en `connectivity_features/<timestamp>/`, incluyendo:

* Matrices de conectividad en `.npy` o `.csv`
* Copia del archivo de configuración empleado

## 🔧 Configuración

Toda la parametrización se efectúa mediante YAML:

* `config.yaml`: Define rutas, métricas de QC (FD, DVARS...), umbrales de exclusión y opciones de reporte.
* `config_connectivity.yaml`: Especifica el atlas de parcelación, la lista de sujetos a procesar y el tipo de conectividad (`pearson`, `partial`, etc.).

Esta estrategia permite un ajuste fino sin modificar el código fuente, favoreciendo la **reproducibilidad** y **colaboración académica**.

## 📦 Dependencias

Listadas en `pyproject.toml`:

* numpy ≥ 1.21
* pandas ≥ 1.3
* scipy ≥ 1.7
* scikit-learn ≥ 1.0
* matplotlib ≥ 3.4
* seaborn ≥ 0.11
* pyyaml ≥ 6.0
* omegaconf ≥ 2.1
* pytest ≥ 7.0
* tqdm ≥ 4.62

## 👤 Autor

**Santiago V. Blas Laguzza**
Doctorando en Inteligencia Artificial y Neurociencia
Email: [santiblaas@gmail.com](mailto:santiblaas@gmail.com)

## 📝 Licencia

Este proyecto se distribuye bajo la **MIT License**. Consulta el archivo `LICENSE` para más información.
