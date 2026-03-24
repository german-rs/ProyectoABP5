# EDA — Análisis Exploratorio de Datos para ComercioYA
### Alkemy | Módulo 5 — Evaluación Final

Proyecto de análisis exploratorio de datos sobre comportamiento de clientes de un e-commerce, desarrollado con Python, Pandas, Seaborn y Matplotlib.

---

## 📊 Informe Ejecutivo

**Proyecto:** Evaluación de Comportamiento de Clientes de ComercioYA 
**Autor:** Germán Riveros 
**Fecha:** 23 de marzo de 2026  

### 1. Resumen Ejecutivo

El presente análisis identifica patrones clave en el comportamiento de compra de los clientes de ComercioYA. Tras procesar 1.000 registros, se concluye que el negocio posee una base de clientes joven-adulta con fuerte presencia regional.

Aunque el ticket promedio es de **$717.261 CLP**, la mediana de **$450.000 CLP** refleja de mejor manera la realidad del comprador frecuente, viéndose el promedio inflado por ventas de alto valor en la categoría Electrohogar.

---

### 2. Metodología y Calidad de Datos (IDA)

Se aplicó un proceso de limpieza riguroso para garantizar la fiabilidad de los hallazgos:

- **Estandarización:** Unificación de variantes de la Región Metropolitana  
- **Integridad:** Imputación de nulos en regiones (38%) y categorías, y eliminación de duplicados (1.96%)  
- **Temporalidad:** Conversión de tipos de datos para análisis de series de tiempo  

---

### 3. Hallazgos Clave

- **Perfil de Usuario:** Edad media de 37 años, desmitificando que la App sea solo para jóvenes  
- **Correlación:** Relación fuerte ($r \approx 0.86$) entre precio unitario y monto total  
- **Distribución:** Comportamiento bimodal entre compras de bajo costo y alta inversión  

---

### 4. Dashboard Final

A continuación, la visualización consolidada de los KPIs:

![Dashboard EDA Final](https://raw.githubusercontent.com/german-rs/ProyectoABP5/main/outputs/reports/06_dashboard_eda_final.png)

> *Figura: Dashboard generado con Pandas, Matplotlib y Seaborn.*

---

### 5. Conclusiones y Recomendaciones

- **Refuerzo Regional:** Potenciar logística en Valparaíso y Biobío  
- **Incentivos en App:** Beneficios para aumentar recurrencia en tickets bajos  
- **Monitoreo Continuo:** Seguimiento mensual mediante dashboard  

---

## Estructura del proyecto

```
comercioya-eda/
│
├── CLAUDE.md                    ← Instrucciones globales para IA
│
├── memory-bank/
│   ├── projectBrief.md          ← Resumen ejecutivo y objetivos
│   ├── productContext.md        ← Contexto del negocio y variables
│   ├── techContext.md           ← Stack, dependencias, instalación
│   ├── systemPatterns.md        ← Arquitectura y patrones de código
│   ├── activeContext.md         ← Tarea actual (actualizar por sesión)
│   └── progress.md              ← Checklist de avance por lección
│
├── data/
│   ├── raw/                     ← Dataset original sin modificar
│   └── processed/               ← Dataset limpio y transformado
│
├── src/
│   ├── 01_ida.py             ← Lección 1: EDA inicial e IDA
│   ├── 02_estadistica.py     ← Lección 2: Estadística descriptiva
│   ├── 03_correlacion.py     ← Lección 3: Correlación
│   ├── 04_regresion.py       ← Lección 4: Regresión lineal
│   ├── 05_seaborn.py         ← Lección 5: Visualización con Seaborn
│   └── 06_matplotlib.py      ← Lección 6: Visualización con Matplotlib
│
└── outputs/
    ├── graphs/                  ← Gráficos exportados en PNG
    └── reports/                 ← Informe técnico final (PDF)
```

## Instalación rápida

```bash
git clone <tu-repo>
cd comercioya-eda
python -m venv venv && source venv/bin/activate
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy jupyter
```

## Cómo usar el Memory Bank con IA
Al comenzar cada sesión, indica al modelo:
> *"Lee los archivos en `memory-bank/` antes de ayudarme, en especial `activeContext.md` y `progress.md`"*

## Tecnologías
`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Statsmodels`

