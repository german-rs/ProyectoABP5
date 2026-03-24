# progress.md — Estado del proyecto

## Checklist general de entregables

- [ ] Notebook VSCode completo y comentado
- [ ] Gráficos exportados en PNG (mínimo 1 por lección)
- [ ] Informe técnico PDF con visualizaciones embebidas
- [ ] Código fuente limpio y comentado
- [ ] Documento de insights y recomendaciones
- [ ] Repositorio GitHub con README profesional
- [ ] Link entregado en Moodle

---

## Progreso por lección

### Lección 1 — EDA / IDA
- [X] Dataset generado o cargado
- [X] Variables clasificadas (cuantitativas / categóricas)
- [X] Nulos detectados y documentados
- [X] Duplicados identificados
- [X] Primeros hallazgos escritos en Markdown

### Lección 2 — Estadística Descriptiva
- [X] Media, mediana, moda calculadas
- [X] Varianza y desviación estándar calculadas
- [X] Cuartiles y percentiles determinados
- [X] Histogramas generados y exportados
- [X] Boxplots generados y exportados
- [X] Outliers identificados y documentados

### Lección 3 — Correlación
- [X] Scatterplots entre variables clave
- [X] Matriz de correlación construida (heatmap)
- [X] Coeficiente de Pearson calculado
- [X] Correlaciones espurias identificadas y justificadas
- [X] Asociaciones documentadas estadística y visualmente

### Lección 4 — Regresión Lineal
- [X] Regresión simple aplicada con statsmodels
- [X] Regresión múltiple aplicada con statsmodels
- [X] R², MSE, MAE calculados e interpretados
- [X] Significancia de predictores evaluada (p-value)
- [X] Coeficientes interpretados
- [X] Regresión visualizada con Seaborn

### Lección 5 — Análisis Visual (Seaborn)
- [X] pairplot generado
- [X] violinplot generado
- [X] jointplot generado
- [X] heatmap generado
- [X] FacetGrid segmentado por categorías
- [X] Estilos, colores y tamaños ajustados
- [X] Insights visuales documentados

### Lección 6 — Visualización con Matplotlib (Dashboard Final)
- [x] Gráficos de barras de ventas por canal y categoría
- [x] Gráfico de línea con evolución temporal de ventas
- [x] Anotaciones de máximos y mínimos (idxmax/idxmin)
- [x] Exportación de gráficos individuales en PNG
- [x] Dashboard integrador (figura con múltiples subplots)

#### Visualizaciones Generadas
A continuación se muestran los resultados finales del análisis visual:

**1. Ventas por Canal y Categoría**
![Ventas por Canal](outputs/graphs/06_barras_ventas_canal_categoria.png)

**2. Evolución Temporal de Ventas**
*(Este gráfico incluye las correcciones de manejo de fechas y validación de secuencias vacías)*
![Evolución Temporal](outputs/graphs/06_linea_evolucion_mensual.png)

**3. Dashboard Final ComercioYA**
![Dashboard Final](outputs/graphs/06_dashboard_final.png)
