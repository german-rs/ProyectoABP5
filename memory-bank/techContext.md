# techContext.md — Contexto técnico

## Entorno de desarrollo
- **IDE:** Visual Studio Code con extensión Jupyter
- **Python:** 3.10+
- **Gestor de entorno:** venv o conda

## Dependencias principales

| Librería     | Versión recomendada | Uso principal                              |
|--------------|---------------------|--------------------------------------------|
| pandas       | ≥ 2.0               | Manipulación y análisis de datos           |
| numpy        | ≥ 1.24              | Operaciones numéricas y generación de datos|
| matplotlib   | ≥ 3.7               | Visualizaciones personalizadas y exportación|
| seaborn      | ≥ 0.12              | Visualizaciones estadísticas de alto nivel |
| statsmodels  | ≥ 0.14              | Modelos de regresión lineal (OLS)          |
| scikit-learn | ≥ 1.3               | Métricas de evaluación (MSE, MAE, R²)      |
| scipy        | ≥ 1.10              | Pruebas estadísticas y cálculo de Z-score  |

## Instalación del entorno
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy jupyter
```

## Dataset
- **Origen:** Sintético (generado con NumPy/pandas o descargado de Kaggle/UCI)
- **Dominio:** E-commerce — comportamiento de clientes
- **Variables esperadas:** compras, visitas, montos, devoluciones, reseñas, región, género, etc.
- **Ubicación:** `data/raw/` (sin modificar), `data/processed/` (limpio)

## Referencias oficiales
- Seaborn: https://seaborn.pydata.org/
- Matplotlib: https://matplotlib.org/stable/users/index.html
- Statsmodels: https://www.statsmodels.org/stable/index.html
- Pandas: https://pandas.pydata.org/docs/
- Kaggle datasets: https://www.kaggle.com/learn/data-visualization
