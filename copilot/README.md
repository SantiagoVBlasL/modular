# Proyecto de Procesamiento de fMRI con AAL3

Este proyecto procesa datos de fMRI (imágenes por resonancia magnética funcional) utilizando el atlas AAL3 para extraer características y realizar análisis de conectividad cerebral.

## Estructura del Proyecto

```
AAL3_pro/
├── config/             # Archivos de configuración
│   └── config.yml      # Configuración principal
├── data/               # Datos de entrada (no incluidos en el repositorio)
├── fmri_features/      # Módulos para procesamiento de fMRI
│   └── data_loader.py  # Funciones para cargar y preprocesar datos
├── results/            # Resultados generados
└── scripts/            # Scripts de procesamiento
    └── process_fmri.py # Script principal
```

## Requisitos

- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- PyYAML

Puedes instalar los requisitos con:

```bash
pip install -r requirements.txt
```

## Uso

1. Modifica el archivo `config/config.yml` para definir las rutas y parámetros de tu proyecto.
2. Ejecuta el script principal:

```bash
python scripts/process_fmri.py
```

## Descripción de los Módulos

### data_loader.py

Este módulo contiene funciones para:
- Cargar datos de sujetos que pasaron el control de calidad (QC)
- Preprocesar series temporales de ROIs (Regiones de Interés)
- Aplicar filtros de paso de banda
- Estandarizar los datos
- Manejar ROIs con varianza cero

## Notas Importantes

- El procesamiento mantiene todas las ROIs, incluso aquellas con varianza cero, para mantener la dimensionalidad consistente entre todos los sujetos.
- Los datos crudos no están incluidos en este repositorio y deben colocarse en el directorio `data/`.
