# 🌾 AgriTech Analytics: Arquitectura IoT y Machine Learning para la Agricultura de Precisión

Este repositorio contiene el código fuente, los modelos predictivos y el dashboard interactivo del proyecto **"Arquitectura de Gestión de Datos Masivos e IoT para la Recomendación Predictiva de Cultivos Agrícolas"**. 

Desarrollado como prueba de concepto (PoC) académica para la Facultad de Ingeniería de Sistemas e Informática (FISI) de la Universidad Nacional Mayor de San Marcos (UNMSM).

## Descripción del Proyecto
El sistema simula un entorno de Big Data e Internet de las Cosas (IoT) aplicado a la agricultura. Utiliza un algoritmo de aprendizaje supervisado (**Random Forest Classifier**) entrenado con datos edafoclimáticos (Nitrógeno, Fósforo, Potasio, Temperatura, Humedad, pH y Precipitación) para inferir y recomendar el cultivo biológicamente óptimo.

El proyecto destaca por su alto nivel de explicabilidad algorítmica (análisis del Índice de Gini para justificar la inversión en hardware IoT) y su despliegue en una interfaz interactiva de análisis en tiempo real.

## Stack Tecnológico
* **Lenguaje Core:** Python 3.x
* **Procesamiento de Datos:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `joblib`
* **Frontend / Visualización:** `streamlit`, `plotly`

## 📁 Estructura del Repositorio
```text
├── dataset/
│   └── Crop_recommendation.csv      # Dataset original balanceado (2200 registros, 8 dimensiones)
├── notebook/
│   └── modelo_agritech.ipynb        # Cuaderno Colab: EDA, Boxplots, Entrenamiento y Exportación (.pkl)
├── app/
│   ├── app.py                       # Código fuente del Dashboard interactivo
│   └── random_forest_model.pkl      # Motor de inferencia (Modelo serializado)
├── requirements.txt                 # Dependencias del proyecto
└── README.md
