# Telco Customer Churn Prediction Pipeline

![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange?style=flat-square&logo=apachespark)
![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)

## Project Overview
This project implements an **End-to-End Big Data Machine Learning Pipeline** to predict customer churn for a Telecommunications company. 
Unlike standard notebook scripts, this project focuses on **Scalability, Reproducibility, and Engineering Best Practices** using Apache Spark.

The goal is to identify customers at high risk of leaving (Churn) to enable proactive retention campaigns.

## Architecture & Workflow
The solution is designed as a modular pipeline:

1.  **Data Ingestion & EDA:** Analysis of raw Telco data (7k+ rows), handling schema enforcement and statistical summaries.
2.  **Feature Engineering:** * `StringIndexer` & `OneHotEncoder` for categorical variables.
    * `VectorAssembler` for feature vectorization.
    * Handling of class imbalance via Threshold Tuning.
3.  **Modeling Strategy:**
    * **Baseline:** Logistic Regression (AUC ~0.86).
    * **Champion:** Random Forest Classifier optimized via **Grid Search & Cross-Validation (3-Folds)**.
4.  **Pipeline Persistence:** The full transformation and inference pipeline is serialized for production serving.

## Key Results & Business Insights
After training and validating multiple models, the system achieved:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.85** | Robust separation between churners and non-churners. |
| **Recall** | **~0.81** | Optimized threshold (0.3) to prioritize capturing potential churners (minimizing False Negatives). |

### The "Smoking Gun"
The Random Forest Feature Importance analysis revealed that **Contract Type** is the single most critical factor:
* Customers with **Month-to-month contracts** are significantly more likely to churn (Importance: ~18.5%).
* *Business Recommendation:* Incentivize users to switch to 1-year or 2-year contracts to reduce churn rate immediately.

## Project Structure
```bash
├── 01_hello_spark.ipynb            # Environment setup & Data Ingestion
├── 02_feature_engineering.ipynb    # Baseline Model (Logistic Regression)
├── 03_random_forest_pipeline.ipynb # Advanced Pipeline (RF + CrossValidation)
├── data/                           # Raw and Processed Parquet data (GitIgnored)
├── models/                         # Serialized Spark Pipelines (GitIgnored)
└── docker-compose.yml              # Infrastructure as Code
