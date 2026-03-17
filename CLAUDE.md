# CLAUDE.md — Instrucciones globales del proyecto

## Contexto de trabajo
Este proyecto es una evaluación académica del módulo de **Análisis Exploratorio de Datos (EDA)**
de Alkemy. El stack es Python puro con Jupyter/VSCode. Todas las respuestas y comentarios de
código deben estar en **español**.

## Stack tecnológico
- Python 3.10+
- pandas, numpy
- matplotlib, seaborn
- statsmodels (regresión)
- scikit-learn (solo para métricas: MSE, MAE)

## Convenciones de código
- Nombres de variables y funciones: `snake_case` en español (ej: `datos_clientes`, `calcular_outliers`)
- Cada sección del notebook debe tener una celda Markdown con título y descripción del objetivo
- Los gráficos SIEMPRE deben incluir: título, etiquetas de ejes, y exportarse en PNG a `outputs/graphs/`
- Usar `plt.tight_layout()` antes de cada `plt.savefig()`
- Comentar el "por qué" de cada decisión analítica, no solo el "qué"

## Estructura de archivos a respetar
```
src/           → scripts y notebook principal
data/raw/      → datos originales sin modificar
data/processed/→ datos limpios y transformados
outputs/graphs/→ gráficos exportados en PNG
outputs/reports/→ informe PDF final
memory-bank/   → contexto del proyecto para IA
```

## Lo que NO hacer
- No usar `inplace=True` en pandas (dificulta el debugging)
- No hardcodear rutas absolutas (usar `pathlib.Path`)
- No entregar código sin comentarios en celdas clave
- No omitir la exportación de gráficos
