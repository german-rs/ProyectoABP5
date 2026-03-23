# =============================================================================
# LECCIÓN 3: CORRELACIÓN
# Proyecto: ComercioYA — EDA para decisiones comerciales
# Módulo 5 — Alkemy
# =============================================================================
# OBJETIVO: Detectar y cuantificar relaciones lineales entre variables
# numéricas mediante scatterplots, la matriz de correlación de Pearson
# y la identificación de correlaciones espurias.
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
RUTA_RAIZ     = Path(__file__).resolve().parent.parent
RUTA_PROCESADO = RUTA_RAIZ / "data" / "processed"
RUTA_GRAFICOS  = RUTA_RAIZ / "outputs" / "graphs"

RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FUNCIÓN REUTILIZABLE: guardar gráfico
# =============================================================================

def guardar_grafico(nombre: str):
    """
    Exporta el gráfico activo a outputs/graphs/ en PNG a 150 dpi.
    Convención de nombre: {leccion}_{tipo}_{descripcion}.png
    """
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Gráfico exportado: outputs/graphs/{nombre}.png")


# =============================================================================
# SECCIÓN 1: CARGA DEL DATASET LIMPIO
# Se carga el archivo procesado generado en la Lección 2, que ya tiene
# duplicados eliminados y nulos imputados. Esto garantiza consistencia
# entre lecciones y evita repetir pasos de limpieza.
# =============================================================================
print("=" * 65)
print("LECCIÓN 3 — CORRELACIÓN")
print("=" * 65)

archivo = RUTA_PROCESADO / "datos_limpios.csv"
datos = pd.read_csv(archivo, sep=";")
print(f"\n✅ Dataset cargado: {datos.shape[0]} filas × {datos.shape[1]} columnas")

# =============================================================================
# SECCIÓN 2: VARIABLE DERIVADA — monto_venta
# Las variables originales no tienen entre sí correlaciones relevantes.
# Sin embargo, el negocio necesita entender qué determina el monto total
# de cada venta. Se crea monto_venta como variable derivada que captura
# el valor económico real de cada transacción.
# Fórmula: precio_unitario × cantidad × (1 − descuento_pct)
# Esta variable es el eje del análisis de correlación de esta lección.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 2: VARIABLE DERIVADA — monto_venta")
print("=" * 65)

datos = datos.assign(
    monto_venta=datos["precio_unitario"] * datos["cantidad"] * (1 - datos["descuento_pct"])
)

print(f"\n✅ Variable 'monto_venta' creada.")
print(f"   Media  : ${datos['monto_venta'].mean():,.0f} CLP")
print(f"   Mediana: ${datos['monto_venta'].median():,.0f} CLP")
print(f"   Mínimo : ${datos['monto_venta'].min():,.0f} CLP")
print(f"   Máximo : ${datos['monto_venta'].max():,.0f} CLP")

# Variables para el análisis de correlación
VARS_CORRELACION = [
    "edad_cliente",
    "precio_unitario",
    "cantidad",
    "descuento_pct",
    "antiguedad_vendedor",
    "monto_venta"
]

# =============================================================================
# SECCIÓN 3: COEFICIENTE DE PEARSON CON P-VALUE
# Se calcula r de Pearson y su p-value para cada par de variables.
# El p-value indica si la correlación es estadísticamente significativa
# (p < 0.05) o podría deberse al azar.
# IMPORTANTE: Pearson mide SOLO relaciones lineales. Un r ≈ 0 no implica
# que no haya relación, sino que no hay relación LINEAL.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 3: COEFICIENTE DE PEARSON CON SIGNIFICANCIA ESTADÍSTICA")
print("=" * 65)

print(f"\n{'Par de variables':<45} {'r de Pearson':>12} {'p-value':>12} {'Significativa':>14}")
print("-" * 85)

pares_analizados = []
for i, var1 in enumerate(VARS_CORRELACION):
    for var2 in VARS_CORRELACION[i+1:]:
        serie1 = datos[var1].dropna()
        serie2 = datos[var2].dropna()
        # Alinear índices para trabajar con el mismo conjunto de filas
        idx_comun = serie1.index.intersection(serie2.index)
        r, p = stats.pearsonr(serie1[idx_comun], serie2[idx_comun])
        significativa = "✅ Sí" if p < 0.05 else "❌ No"
        par = f"{var1} ↔ {var2}"
        print(f"{par:<45} {r:>12.4f} {p:>12.4f} {significativa:>14}")
        pares_analizados.append({
            "var1": var1, "var2": var2,
            "pearson_r": round(r, 4), "p_value": round(p, 6),
            "significativa": p < 0.05
        })

df_pearson = pd.DataFrame(pares_analizados)

# =============================================================================
# SECCIÓN 4: MATRIZ DE CORRELACIÓN — HEATMAP
# La matriz de correlación presenta todos los coeficientes r de Pearson
# en forma visual. Se usa una paleta divergente (rojo-blanco-azul) para
# que sea inmediatamente claro cuáles son correlaciones positivas,
# negativas o nulas.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 4: HEATMAP — MATRIZ DE CORRELACIÓN")
print("=" * 65)

matriz_corr = datos[VARS_CORRELACION].corr(method="pearson")

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    matriz_corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",       # Rojo = correlación negativa, Azul = positiva
    vmin=-1, vmax=1,     # Escala fija para interpretación correcta
    linewidths=0.5,
    linecolor="white",
    square=True,
    ax=ax,
    annot_kws={"size": 10, "weight": "bold"}
)

ax.set_title(
    "ComercioYA — Matriz de correlación de Pearson\n"
    "(variables numéricas + monto_venta)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

guardar_grafico("03_heatmap_correlacion")
print("✅ Heatmap generado.")

# =============================================================================
# SECCIÓN 5: SCATTERPLOTS DE PARES RELEVANTES
# Se visualizan los tres pares con mayor r absoluto para explorar la
# naturaleza de la relación (lineal, no lineal, con grupos, etc.).
# Se agrega línea de regresión para facilitar la lectura visual.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 5: SCATTERPLOTS DE PARES CLAVE")
print("=" * 65)

# Los tres pares más correlacionados (por r absoluto) según el análisis previo
pares_graficos = [
    ("precio_unitario", "monto_venta",
     "Precio unitario (CLP)", "Monto de venta (CLP)",
     "Precio unitario vs. Monto de venta\n(correlación esperada: r=0.86)"),

    ("cantidad", "monto_venta",
     "Cantidad de unidades", "Monto de venta (CLP)",
     "Cantidad vs. Monto de venta\n(correlación moderada: r=0.38)"),

    ("edad_cliente", "descuento_pct",
     "Edad del cliente (años)", "Descuento aplicado (proporción)",
     "Edad vs. Descuento\n(correlación débil: r=-0.08)"),
]

fig, ejes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "ComercioYA — Scatterplots de pares de variables",
    fontsize=13, fontweight="bold"
)

for ax, (var_x, var_y, etiq_x, etiq_y, titulo) in zip(ejes, pares_graficos):
    x = datos[var_x]
    y = datos[var_y]
    r, p = stats.pearsonr(x, y)

    ax.scatter(x, y, alpha=0.3, s=18, color="#4C72B0", edgecolors="none")

    # Línea de tendencia (regresión simple)
    m, b = np.polyfit(x, y, 1)
    x_linea = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_linea, m * x_linea + b, color="#DD4444",
            linewidth=2, label=f"r = {r:.2f} | p = {p:.4f}")

    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel(etiq_x, fontsize=9)
    ax.set_ylabel(etiq_y, fontsize=9)
    ax.legend(fontsize=9)

    # Formatear eje Y en millones si es monto_venta
    if var_y == "monto_venta":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda val, _: f"${val/1_000_000:.1f}M")
        )

guardar_grafico("03_scatterplots_pares_clave")
print("✅ Scatterplots generados.")

# =============================================================================
# SECCIÓN 6: IDENTIFICACIÓN DE CORRELACIONES ESPURIAS
# Una correlación espuria ocurre cuando dos variables muestran r elevado
# pero la relación NO es causal ni directa. El caso más importante aquí
# es precio_unitario ↔ monto_venta: la correlación es alta (r=0.86) pero
# es matemáticamente inevitable porque monto_venta SE CALCULA a partir de
# precio_unitario. No es una correlación descubierta, es una identidad
# algebraica. Documentarla evita interpretaciones erróneas en el informe.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 6: CORRELACIONES ESPURIAS — ANÁLISIS CRÍTICO")
print("=" * 65)

print("""
⚠️  CASO 1: precio_unitario ↔ monto_venta  (r = 0.86)
    Esta es la correlación más alta del análisis, pero es una
    CORRELACIÓN ESPURIA POR CONSTRUCCIÓN:
    monto_venta = precio_unitario × cantidad × (1 − descuento_pct)
    El precio unitario es un componente directo del monto de venta,
    por lo tanto es IMPOSIBLE que no estén correlacionados.
    Conclusión: esta correlación no revela ningún patrón de negocio;
    solo confirma la fórmula matemática. No debe usarse como hallazgo.

⚠️  CASO 2: cantidad ↔ monto_venta  (r = 0.38)
    Correlación moderada con la misma causa: cantidad también es un
    factor del cálculo de monto_venta. Sin embargo, como cantidad
    tiene menos varianza que precio_unitario (rango: 1-3 unidades),
    su contribución a la correlación es menor.
    Conclusión: correlación esperada por construcción, no por
    comportamiento del cliente.

✅  CORRELACIÓN GENUINA MÁS DESTACABLE:
    edad_cliente ↔ descuento_pct  (r = -0.08, p < 0.05)
    Aunque débil, es la única correlación estadísticamente
    significativa entre variables INDEPENDIENTES entre sí.
    Sugiere que clientes de mayor edad tienden a recibir
    descuentos ligeramente menores. Requiere análisis adicional
    por segmento para confirmar si es un patrón real de negocio.
""")

# =============================================================================
# SECCIÓN 7: RESUMEN DE HALLAZGOS DE CORRELACIÓN
# =============================================================================
print("=" * 65)
print("RESUMEN — LECCIÓN 3 COMPLETADA")
print("=" * 65)

significativas = df_pearson[df_pearson["significativa"] == True]
print(f"\n📊 Pares analizados          : {len(df_pearson)}")
print(f"   Correlaciones significativas (p<0.05): {len(significativas)}")
print()
print("   Correlaciones estadísticamente significativas:")
for _, fila in significativas.iterrows():
    print(f"   → {fila['var1']} ↔ {fila['var2']}: "
          f"r={fila['pearson_r']:.4f} | p={fila['p_value']:.6f}")

print("""
✅ Heatmap exportado   → outputs/graphs/03_heatmap_correlacion.png
✅ Scatterplots exportados → outputs/graphs/03_scatterplots_pares_clave.png

⚠️  HALLAZGO PRINCIPAL:
    Las variables originales del dataset prácticamente no tienen
    correlación lineal entre sí. El único par con r relevante
    involucra la variable DERIVADA monto_venta, cuya correlación
    con precio_unitario y cantidad es matemáticamente esperada
    (correlación espuria por construcción).

🔜 PRÓXIMA LECCIÓN: Regresión lineal simple y múltiple
   Variable dependiente sugerida: monto_venta
   Predictores candidatos: precio_unitario, cantidad, descuento_pct
""")




## Hallazgos clave de esta lección

#El resultado más importante (y pedagógicamente valioso) es que **ningún par de variables originales tiene correlación lineal relevante**. Esto ocurre en datasets de e-commerce reales donde el comportamiento del cliente es muy heterogéneo. Los puntos centrales son:

#Las correlaciones altas (r=0.86 y r=0.38) entre `monto_venta` y sus componentes son **espurias por construcción** — son identidades algebraicas, no descubrimientos. La única correlación estadísticamente significativa entre variables independientes es `edad_cliente` ↔ `descuento_pct` (r=-0.08), que aunque débil, abre una pregunta de negocio interesante para explorar.

#---

## Actualización de `progress.md` — Lección 3

### Lección 3 — Correlación
# - [x] Scatterplots entre variables clave
# - [x] Matriz de correlación construida (heatmap)
# - [x] Coeficiente de Pearson calculado
# - [x] Correlaciones espurias identificadas y justificadas
# - [x] Asociaciones documentadas estadística y visualmente
