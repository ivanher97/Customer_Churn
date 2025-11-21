# Telco Customer Churn Prediction Pipeline

## Descripción del Proyecto
Este proyecto implementa un **pipeline de Machine Learning end-to-end escalable** utilizando **Apache Spark (PySpark)**. El objetivo es predecir la fuga de clientes (Churn) en una empresa de telecomunicaciones.

El sistema ingesta datos crudos, los procesa y almacena en formato optimizado (Parquet), realiza ingeniería de características distribuida y entrena modelos clasificadores para identificar usuarios en riesgo, permitiendo al negocio tomar acciones preventivas de retención.

---

## 1. Problema de Negocio
**El Desafío:**
La adquisición de nuevos clientes es entre 5 y 25 veces más costosa que la retención de los existentes. La empresa necesita identificar proactivamente qué clientes tienen una alta probabilidad de cancelar sus servicios el próximo mes.

**La Solución:**
Un modelo predictivo capaz de clasificar clientes con riesgo de fuga (`Churn = Yes`).
* **KPI Principal:** Maximizar el **Recall** (Sensibilidad). Preferimos falsos positivos (ofrecer descuentos a quien no se iba a ir) que falsos negativos (perder un cliente sin haber intentado retenerlo).

---

## 2. Arquitectura y Stack Tecnológico

El proyecto sigue una arquitectura de procesamiento de datos moderna basada en el ecosistema Hadoop/Spark.

### Tech Stack
* **Motor de Procesamiento:** Apache Spark 3.5.0 (PySpark).
* **Lenguaje:** Python 3.11.
* **Almacenamiento:** Parquet (Columnar storage para alto rendimiento).
* **ML Library:** Spark MLlib (Pipelines, CrossValidator).
* **Control de Versiones:** Git.

### Pipeline de Datos

1.  **Ingesta (ETL):**
    * Descarga de datos desde fuente remota (GitHub/IBM).
    * Limpieza de tipos de datos (Casting de `TotalCharges` a Double).
    * Imputación y eliminación de nulos.
    * Generación de variable objetivo binaria (`label`).
    * **Persistencia:** Guardado en capa *Silver* como archivos **Parquet**.

2.  **Feature Engineering:**
    * Tratamiento de variables categóricas: `StringIndexer` + `OneHotEncoder`.
    * Vectorización de features: `VectorAssembler`.
    * División de datos (Train/Test).

3.  **Modelado y Tuning:**
    * Comparación de modelos: **Logistic Regression** vs. **Random Forest**.
    * Optimización de hiperparámetros con `CrossValidator` y `ParamGridBuilder`.
    * Ajuste de umbral de decisión (Threshold tuning) para priorizar el Recall.

---

## 3. Resultados e Insights

### Métricas de Evaluación (Test Set)
Se priorizó el modelo **Logistic Regression** con ajuste de umbral (0.3) por su capacidad de capturar el 81% de los fugados.

| Modelo | Accuracy | Precision | Recall | AUC | Comentarios |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Base)** | 81.4% | 68.1% | 56.8% | **0.86** | Alta precisión, bajo recall. |
| **Logistic Regression (Tuned)**| 77.3% | 55.0% | **81.2%** | 0.86 | **Modelo seleccionado.** Maximiza detección de fuga. |
| **Random Forest (CV)** | 84.9% | - | - | 0.85 | Robusto, utilizado para Feature Importance. |

### Top Factores de Fuga (Feature Importance)
Según el modelo Random Forest, las variables que más influyen en la decisión del cliente de abandonar son:

1.  **Tipo de Contrato (Mes a mes):** Los clientes sin permanencia anual son los más propensos a irse.
2.  **Seguridad Online (No):** La falta de servicios añadidos facilita la fuga.
3.  **Antigüedad (Tenure):** A mayor antigüedad, menor probabilidad de fuga.
4.  **Cargos Totales:** Facturas acumuladas altas son un factor de riesgo.
5.  **Internet Fibra Óptica:** Curiosamente, los usuarios de fibra tienen una tasa de churn más alta (posiblemente por precio o problemas técnicos).

---

## Estructura del Proyecto

```bash
├── data/
│   ├── processed/       # Datos limpios en formato Parquet
│   └── Telco-Customer-Churn.csv
├── models/              # Modelos serializados para inferencia
├── 01_hello_spark.ipynb # ETL: Ingesta, limpieza y guardado en Parquet
├── 02_feature_engineering.ipynb # Entrenamiendo Regresión Logística & Evaluación
├── 03_random_forest_pipeline.ipynb # Pipeline complejo con Random Forest & GridSearch
├── test_arquitectura.ipynb # Validación de entorno Spark
└── .gitignore
