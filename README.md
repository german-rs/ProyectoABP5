# EDA — Análisis Exploratorio de Datos para ComercioYA
### Alkemy | Módulo 5 — Evaluación Final

Proyecto de análisis exploratorio de datos sobre comportamiento de clientes
de un e-commerce, desarrollado con Python, Pandas, Seaborn y Matplotlib.

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
