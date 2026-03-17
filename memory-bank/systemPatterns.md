# systemPatterns.md — Arquitectura y patrones del proyecto

## Patrón de arquitectura elegido: Pipeline Analítico Lineal

Este patrón es el más adecuado para un proyecto EDA académico porque:
- El flujo es secuencial y predecible (datos crudos → limpios → analizados → visualizados → reportados)
- Cada etapa tiene entradas y salidas claramente definidas
- Facilita la trazabilidad para el informe técnico
- Es reproducible de principio a fin con un solo notebook

```
[data/raw/]
     │
     ▼
[01_ida.ipynb]          ← Carga, inspección inicial, tipos de variables
     │
     ▼
[data/processed/]       ← Datos limpios guardados como checkpoint
     │
     ▼
[02_estadistica.ipynb]  ← Medidas de tendencia central, dispersión, outliers
     │
     ▼
[03_correlacion.ipynb]  ← Correlaciones, scatterplots, matriz de correlación
     │
     ▼
[04_regresion.ipynb]    ← Modelos OLS, métricas R², MSE, MAE
     │
     ▼
[05_seaborn.ipynb]      ← pairplot, violinplot, heatmap, FacetGrid
     │
     ▼
[06_matplotlib.ipynb]   ← Figuras finales, subplots, exportación
     │
     ▼
[outputs/reports/]      ← Informe PDF con todos los hallazgos
```

## Patrones de código reutilizables

### Patrón: función de exportación de gráficos
```python
from pathlib import Path

def guardar_grafico(nombre: str, carpeta: str = "outputs/graphs"):
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{carpeta}/{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Patrón: función de resumen estadístico
```python
def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T.assign(
        skewness=df.skew(),
        kurtosis=df.kurt(),
        nulos=df.isnull().sum()
    )
```

### Patrón: detección de outliers (IQR)
```python
def detectar_outliers_iqr(serie: pd.Series) -> pd.Series:
    Q1, Q3 = serie.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return serie[(serie < Q1 - 1.5 * IQR) | (serie > Q3 + 1.5 * IQR)]
```

## Convención de nombres de archivos de gráficos
```
{leccion}_{tipo_grafico}_{variable}.png
# Ejemplos:
02_boxplot_monto_compra.png
03_heatmap_correlacion.png
04_regresion_visitas_vs_monto.png
```
