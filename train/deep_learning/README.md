# Estructura de archivos y metodología en Deep Learning

## Estructura de archivos

```
1_modeling/                                  
├── config.json                              # Parámetros de entrenamiento
├── train.py                                 # Script principal de entrenamiento
└── data_loaders/                            # Scripts de carga de datos para entrenamiento
    ├── data_loader_for_cv_org.py
    └── data_loader_for_cv_roi.py

2_analyse_results/ 
├── 1_predict/
│   ├── predict.py                           # Generación de predicciones
│   └── z_data_loader_for_cv_for_predict.py  # Carga de datos para evaluación
├── analyse_probs/
│   └── analyse_probs.py                     # Análisis de la distribución de las predicciones
└── simple_statistical_analysis/
    ├── 1_compare_models.py                  # Comparación estadística entre métricas de los modelos
    └── 1a_gland_vs_full_diferences.py       # Comparación entre glándula e imagen completa

3_model_explicability/
├── interpretability_analysis.py             # Análisis de explicabilidad de predicciones
└── z_data_loader_for_explicability_roi.py   # Carga de datos para interpretabilidad

```

## Metodología

El proceso de análisis mediante Deep Learning se organiza en tres fases principales:

### 1. Entrenamiento y validación ([`train.py`](./1_modeling/train.py))

Este script entrena modelos convolucionales mediante validación cruzada estratificada con grupos.

1. Cargadores de datos

    Se pueden usar dos estrategias de carga de datos, según el valor del parámetro `--mode`:

    - [`data_loader_for_cv_org.py`](./1_modeling/data_loaders/data_loader_for_cv_org.py): utiliza la imagen completa como entrada.  
    - [`data_loader_for_cv_roi.py`](./1_modeling/data_loaders/data_loader_for_cv_roi.py): usa únicamente la región correspondiente a la glándula prostática.

2. Modelos y configuraciones

    - Los modelos utilizados se definen en `config.json`, donde cada clave corresponde a una configuración concreta.  
    - Es posible definir modelos con o sin transformaciones adicionales, especificadas en el campo `extra_transforms`.

3. Entrenamiento y evaluación

    - Se entrena el modelo seleccionado mediante validación cruzada, con cálculo de métricas clásicas: AUC, F1 (macro y binario), accuracy, sensibilidad, especificidad, MCC, etc.  
    - Se aplica *early stopping* basado en el AUC de validación.  
    - Por cada *split* se guarda el mejor modelo, y también se identifica y almacena el mejor modelo global.

#### Archivos generados

Los resultados de esta fase se almacenan en [`artifacts/deep_learning`](../../artifacts/deep_learning/), organizados por modalidad (`full` o `gland`) y configuración:

- Checkpoints de modelos por split y mejor modelo global (`.pth`)  
- Archivos `.csv` con métricas de entrenamiento y validación  
- Logs de entrenamiento en formato `.log`


### 2. Comparación de modelos

La comparación de los modelos se realiza a partir de las predicciones de los conjuntos de validación. Estas predicciones se generan mediante el script [`predict.py`](./2_analyse_results/1_predict/predict.py), incluyendo también las probabilidades asignadas a cada clase para cada caso.

A partir de esto, se utilizan dos estrategias complementarias para comparar el rendimiento de los distintos modelos entrenados:

1. Comparación simple ([`1_compare_models.py`](./2_analyse_results/simple_statistical_analysis/1_compare_models.py))

    - A partir de las predicciones, se calculcan métricas como AUC, F1, accuracy, sensibilidad, etc.  
    - Se identifican los modelos más prometedores mediante la visualización de dichas métricas   
    - Sobre el AUC, se aplican análisis estadísticos como el test de Friedman y comparaciones *post-hoc* (Wilcoxon con corrección de Holm).
    - Se compara tambien el AUC de cada configuración al usar la glándula prostática frente a exclusivamente la imagen completa, como entrada de los modelos ([`1a_gland_vs_full_diferences.py`](./2_analyse_results/simple_statistical_analysis/1a_gland_vs_full_diferences.py)).

2. Comparación mediante predicciones ([`analyse_probs.py`](./2_analyse_results/analyse_probs/analyse_probs.py))
 
    Las probabilidades generadas por los modelos se comparan realizando un análisis estadístico acompañado de visualizaciones (boxplots, heatmaps de *p-valores*).

#### Archivos generados

Las predicciones se guardan en la carpeta [`artifacts/deep_learning/[mode]/z_predictions`](../../artifacts/deep_learning/gland/z_predictions/).

Los resultados de los análisis se almacenan en la carpeta [`results/deep_learning/model_comparison`](../../results/deep_learning/model_comparison/), incluyendo:

- Visualizaciones por métrica (gráficos de radar, barras, boxplots).  
- Archivos `.csv` con métricas combinadas y estadísticas resumen.  
- Informes `.txt` con resultados de los análisis estadísticos.

### 3. Explicabilidad del modelo

Para comprender el comportamiento del modelo, se aplica un análisis de interpretabilidad sobre ciertas muestras. El script principal que gestiona todo este proceso es ([`interpretability_analysis.py`](./3_model_explicability/interpretability_analysis.py)).

Este análisis incluye dos enfoques complementarios:

- **GradCAM y Guided Backpropagation**. Se generan mapas de activación (GradCAM) y mapas de *guided backpropagation* (GBP) sobre las regiones de entrada utilizadas por el modelo. Posteriormente, ambos mapas se combinan para identificar las zonas más relevantes en la toma de decisiones.

- **Mapas de sensibilidad por oclusión**. Se evalúa la sensibilidad del modelo a regiones específicas del volumen mediante oclusión. Los mapas individuales se generan para cada muestra correctamente clasificada y, a partir de ellos, se construyen mapas agregados por clase (csPCa y no csPCa).

#### Archivos generados

Los resultados del análisis se almacenan en [`results/deep_learning/interpretability`](../../results/deep_learning/interpretability/), organizados por tipo de análisis y modalidad:

- Mapas individuales y combinados de GradCAM y GBP (GradCAM)

- Mapas de oclusión individuales (OcclusionSensitivity)

- Mapas de oclusión agregados (AggregatedMaps)

