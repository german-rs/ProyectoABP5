# =============================================================================
# LECCIÓN 5: ANÁLISIS VISUAL DE DATOS — SEABORN
# Proyecto: ComercioYA — EDA para decisiones comerciales
# Módulo 5 — Alkemy
# =============================================================================
# OBJETIVO: Representar relaciones complejas y distribuciones mediante
# Seaborn. Generar pairplot, violinplot, jointplot, heatmap y FacetGrid,
# ajustando estilos, colores y tamaños. Documentar insights visuales.
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
RUTA_RAIZ      = Path(__file__).resolve().parent.parent
RUTA_PROCESADO = RUTA_RAIZ / "data" / "processed"
RUTA_GRAFICOS  = RUTA_RAIZ / "outputs" / "graphs"

RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURACIÓN GLOBAL DE ESTILO SEABORN
# Se aplica un tema consistente a todos los gráficos de esta lección.
# "whitegrid" es ideal para análisis estadístico porque las líneas de
# cuadrícula facilitan la lectura de valores sin distraer visualmente.
# =============================================================================
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PALETA_CATEGORIAS = "Set2"    # Paleta cualitativa para variables categóricas
PALETA_SECUENCIAL = "Blues_d" # Paleta secuencial para variables numéricas


def guardar_grafico(nombre: str):
    """Exporta el gráfico activo a outputs/graphs/ en PNG a 150 dpi."""
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Gráfico exportado: outputs/graphs/{nombre}.png")


# =============================================================================
# SECCIÓN 1: CARGA Y PREPARACIÓN DEL DATASET
# Se carga el dataset limpio y se aplican transformaciones necesarias
# para los gráficos: unificación de región metropolitana, creación de
# monto_venta y filtrado de nulos en categóricas para visualizaciones.
# =============================================================================
print("=" * 65)
print("LECCIÓN 5 — ANÁLISIS VISUAL CON SEABORN")
print("=" * 65)

archivo = RUTA_PROCESADO / "datos_limpios.csv"
datos = pd.read_csv(archivo, sep=";")

# Variable derivada consistente con lecciones anteriores
datos = datos.assign(
    monto_venta=datos["precio_unitario"] * datos["cantidad"] * (1 - datos["descuento_pct"])
)

# Unificar variantes de Región Metropolitana (inconsistencia detectada en L1)
datos["region_cliente"] = datos["region_cliente"].replace(
    {"RM": "Región Metropolitana", "Metropolitana": "Región Metropolitana"}
)

# Dataset filtrado: solo filas con categoria_producto y region conocidas.
# Se usa para gráficos que requieren esa variable como eje o segmentación.
# Los nulos no se eliminaron del dataset principal para no perder info en
# los gráficos que no necesitan esa columna.
datos_con_categoria = datos.dropna(subset=["categoria_producto"])
datos_con_region    = datos.dropna(subset=["region_cliente"])

print(f"\n✅ Dataset completo         : {datos.shape[0]} filas")
print(f"   Con categoría conocida   : {datos_con_categoria.shape[0]} filas")
print(f"   Con región conocida      : {datos_con_region.shape[0]} filas")

VARS_NUMERICAS = ["edad_cliente", "precio_unitario", "cantidad",
                  "descuento_pct", "antiguedad_vendedor", "monto_venta"]

# =============================================================================
# SECCIÓN 2: PAIRPLOT
# El pairplot muestra la distribución de cada variable en la diagonal y
# la relación entre cada par en los ejes fuera de la diagonal.
# Se segmenta por canal_venta para ver si los canales tienen patrones
# distintos de comportamiento en las variables numéricas.
# Se excluyen variables de ID (no analíticas) y se limita a 4 variables
# para mantener el gráfico legible (más variables → grilla muy densa).
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 2: PAIRPLOT — RELACIONES ENTRE VARIABLES NUMÉRICAS")
print("=" * 65)

vars_pairplot = ["edad_cliente", "precio_unitario", "cantidad", "monto_venta"]

grilla_pair = sns.pairplot(
    datos[vars_pairplot + ["canal_venta"]],
    hue="canal_venta",
    palette=PALETA_CATEGORIAS,
    diag_kind="kde",        # Curva de densidad en la diagonal (más suave que histograma)
    plot_kws={"alpha": 0.4, "s": 20},
    diag_kws={"linewidth": 2}
)

grilla_pair.figure.suptitle(
    "ComercioYA — Pairplot de variables numéricas clave\nsegmentado por canal de venta",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
grilla_pair.savefig(
    RUTA_GRAFICOS / "05_pairplot_variables_numericas.png",
    dpi=150, bbox_inches="tight"
)
plt.close()
print("   💾 Gráfico exportado: outputs/graphs/05_pairplot_variables_numericas.png")

print("""
📝 INSIGHTS — Pairplot:
▸ precio_unitario vs monto_venta: relación lineal clara, confirmando
  los resultados de la Lección 4 (r=0.86).
▸ Los tres canales (Web, App, Tienda) se solapan en casi todas las
  variables: el canal de venta no discrimina el perfil del cliente.
▸ cantidad tiene solo 3 valores posibles (1, 2, 3), lo que genera
  bandas horizontales/verticales en sus scatterplots.
""")

# =============================================================================
# SECCIÓN 3: VIOLINPLOT
# El violinplot combina un boxplot con una estimación de densidad kernel
# (KDE), mostrando la distribución completa de los datos.
# Se compara monto_venta por categoria_producto para visualizar la
# enorme diferencia entre categorías de bajo y alto valor.
# Se usa escala logarítmica en el eje Y porque la diferencia entre
# Belleza (~$47K) y Electrohogar (~$1.4M) es de 30x, lo que aplanaría
# visualmente las categorías baratas en escala lineal.
# =============================================================================
print("=" * 65)
print("SECCIÓN 3: VIOLINPLOT — DISTRIBUCIÓN DE MONTO POR CATEGORÍA")
print("=" * 65)

orden_categorias = (
    datos_con_categoria
    .groupby("categoria_producto")["monto_venta"]
    .median()
    .sort_values()
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.violinplot(
    data=datos_con_categoria,
    x="categoria_producto",
    y="monto_venta",
    order=orden_categorias,
    palette=PALETA_CATEGORIAS,
    inner="quartile",   # Muestra Q1, mediana y Q3 dentro del violín
    linewidth=1.2,
    ax=ax
)

ax.set_yscale("log")    # Escala log para manejar diferencia de magnitud entre categorías
ax.set_title(
    "ComercioYA — Distribución del monto de venta por categoría de producto\n"
    "(escala logarítmica — eje Y)",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Categoría de producto", fontsize=11)
ax.set_ylabel("Monto de venta (CLP, escala log)", fontsize=11)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"${v/1_000:,.0f}K" if v < 1_000_000
                          else f"${v/1_000_000:.1f}M")
)
ax.tick_params(axis="x", labelsize=10)

guardar_grafico("05_violinplot_monto_por_categoria")

print("""
📝 INSIGHTS — Violinplot:
▸ Electrohogar y Tecnología tienen montos medianos ~30x superiores
  a Belleza y Vestuario. Son las categorías que generan mayor
  volumen de ingresos por transacción.
▸ Vestuario tiene la distribución más compacta (menor varianza):
  los precios de ropa son más homogéneos.
▸ Electrohogar tiene la distribución más amplia: hay productos
  desde ~$400K hasta $2.4M (lavadoras, refrigeradores, etc.).
▸ Calzado ocupa un segmento intermedio con distribución estrecha.
""")

# =============================================================================
# SECCIÓN 4: JOINTPLOT
# El jointplot muestra la distribución conjunta de dos variables y sus
# distribuciones marginales (individuales) simultáneamente.
# Se analiza precio_unitario vs monto_venta para ver con detalle la
# relación lineal confirmada en la Lección 4, incluyendo la densidad
# conjunta que revela los dos grupos de productos (baratos vs caros).
# =============================================================================
print("=" * 65)
print("SECCIÓN 4: JOINTPLOT — PRECIO UNITARIO vs. MONTO DE VENTA")
print("=" * 65)

grafico_joint = sns.jointplot(
    data=datos,
    x="precio_unitario",
    y="monto_venta",
    kind="hex",             # Hexbin: mejor que scatter para 1000 puntos (evita solapamiento)
    color="#4C72B0",
    height=8,
    ratio=4,                # Proporción entre gráfico central y marginales
    marginal_kws={"bins": 25, "fill": True}
)

grafico_joint.figure.suptitle(
    "ComercioYA — Jointplot: Precio unitario vs. Monto de venta\n"
    "(densidad hexbin + distribuciones marginales)",
    fontsize=12, fontweight="bold", y=1.02
)
grafico_joint.ax_joint.set_xlabel("Precio unitario (CLP)", fontsize=10)
grafico_joint.ax_joint.set_ylabel("Monto de venta (CLP)", fontsize=10)
grafico_joint.ax_joint.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"${v/1_000_000:.1f}M")
)
grafico_joint.ax_joint.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"${v/1_000_000:.1f}M")
)

plt.tight_layout()
grafico_joint.savefig(
    RUTA_GRAFICOS / "05_jointplot_precio_vs_monto.png",
    dpi=150, bbox_inches="tight"
)
plt.close()
print("   💾 Gráfico exportado: outputs/graphs/05_jointplot_precio_vs_monto.png")

print("""
📝 INSIGHTS — Jointplot:
▸ La distribución marginal de precio_unitario confirma la bimodalidad:
  concentración en precios bajos (~12-35K) y en precios altos (~650-800K).
▸ La distribución conjunta muestra dos clusters claramente separados:
  productos baratos (bajo precio, bajo monto) y caros (alto precio,
  alto monto). No hay productos de precio intermedio en cantidad
  significativa.
▸ La relación lineal es muy clara dentro de cada cluster.
""")

# =============================================================================
# SECCIÓN 5: HEATMAP DE CORRELACIÓN (versión mejorada con Seaborn)
# Se presenta el heatmap con anotaciones más completas que en la Lección 3,
# incluyendo solo las variables más relevantes para el análisis final
# y aplicando una máscara triangular superior para evitar duplicados.
# =============================================================================
print("=" * 65)
print("SECCIÓN 5: HEATMAP — CORRELACIÓN CON MÁSCARA TRIANGULAR")
print("=" * 65)

vars_heatmap = ["edad_cliente", "precio_unitario", "cantidad",
                "descuento_pct", "antiguedad_vendedor", "monto_venta"]

matriz_corr = datos[vars_heatmap].corr(method="pearson")

# Máscara triangular superior: muestra cada par solo una vez
# Esto reduce la redundancia visual y facilita la lectura
mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    matriz_corr,
    mask=mascara,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    linewidths=0.8,
    linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Coeficiente de Pearson (r)"},
    annot_kws={"size": 11, "weight": "bold"},
    ax=ax
)

ax.set_title(
    "ComercioYA — Matriz de correlación (triángulo inferior)\n"
    "Variables numéricas + monto_venta",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

guardar_grafico("05_heatmap_correlacion_triangular")

print("""
📝 INSIGHTS — Heatmap:
▸ La máscara triangular elimina la redundancia: cada par aparece una sola vez.
▸ precio_unitario ↔ monto_venta (r=0.86) y cantidad ↔ monto_venta (r=0.38)
  son las únicas correlaciones relevantes — ambas por construcción algebraica.
▸ Las variables de perfil del cliente (edad) y del vendedor (antigüedad)
  no correlacionan con ninguna otra variable. Son independientes entre sí
  y del comportamiento de compra.
""")

# =============================================================================
# SECCIÓN 6: FACETGRID — SEGMENTACIÓN POR CATEGORÍAS
# FacetGrid permite crear una cuadrícula de gráficos donde cada panel
# corresponde a un subgrupo de los datos (una categoría específica).
# Esto permite comparar visualmente distribuciones entre segmentos
# sin superponer todos los datos en un solo eje.
# Se analiza la distribución de edad_cliente por canal_venta y género,
# ya que estas combinaciones responden a preguntas de segmentación real.
# =============================================================================
print("=" * 65)
print("SECCIÓN 6: FACETGRID — DISTRIBUCIÓN DE EDAD POR CANAL Y GÉNERO")
print("=" * 65)

grilla = sns.FacetGrid(
    datos,
    col="canal_venta",
    hue="genero_cliente",
    col_order=["Tienda", "Web", "App"],
    palette={"Hombre": "#4C72B0", "Mujer": "#DD4444"},
    height=4,
    aspect=1.1,
    sharey=True      # Eje Y compartido para comparación directa entre paneles
)

grilla.map(sns.kdeplot, "edad_cliente", fill=True, alpha=0.45, linewidth=1.5)
grilla.map(sns.kdeplot, "edad_cliente", fill=False, linewidth=2)

grilla.add_legend(title="Género", fontsize=10)
grilla.set_axis_labels("Edad del cliente (años)", "Densidad")
grilla.set_titles(col_template="Canal: {col_name}", size=12, fontweight="bold")

grilla.figure.suptitle(
    "ComercioYA — Distribución de edad por canal de venta y género",
    fontsize=13, fontweight="bold", y=1.03
)

plt.tight_layout()
grilla.savefig(
    RUTA_GRAFICOS / "05_facetgrid_edad_canal_genero.png",
    dpi=150, bbox_inches="tight"
)
plt.close()
print("   💾 Gráfico exportado: outputs/graphs/05_facetgrid_edad_canal_genero.png")

# --- FacetGrid 2: monto_venta por región (top 6) ---
top6_regiones = (
    datos_con_region["region_cliente"]
    .value_counts()
    .head(6)
    .index.tolist()
)
datos_top6 = datos_con_region[datos_con_region["region_cliente"].isin(top6_regiones)]

grilla2 = sns.FacetGrid(
    datos_top6,
    col="region_cliente",
    col_wrap=3,             # Máximo 3 columnas, luego salta a nueva fila
    col_order=top6_regiones,
    height=3.5,
    aspect=1.2
)

grilla2.map(
    sns.histplot, "monto_venta",
    bins=15, color="#4C72B0", edgecolor="white", linewidth=0.4
)
grilla2.set_axis_labels("Monto de venta (CLP)", "N° de ventas")
grilla2.set_titles(col_template="{col_name}", size=10, fontweight="bold")

for ax in grilla2.axes.flat:
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1_000_000:.1f}M")
    )
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)

grilla2.figure.suptitle(
    "ComercioYA — Distribución de monto de venta por región (top 6)",
    fontsize=13, fontweight="bold", y=1.03
)

plt.tight_layout()
grilla2.savefig(
    RUTA_GRAFICOS / "05_facetgrid_monto_por_region.png",
    dpi=150, bbox_inches="tight"
)
plt.close()
print("   💾 Gráfico exportado: outputs/graphs/05_facetgrid_monto_por_region.png")

print("""
📝 INSIGHTS — FacetGrid:
▸ La distribución de edad es muy similar entre canales y géneros:
  no hay un perfil etario diferenciado por canal. La App no atrae
  significativamente a clientes más jóvenes (hipótesis frecuente
  que los datos no confirman).
▸ Valparaíso y Biobío concentran el mayor volumen de ventas, pero
  la distribución de montos es bimodal en todas las regiones,
  reflejando la mezcla de categorías baratas y caras.
▸ La Araucanía muestra menor presencia de ventas de alto valor,
  lo que puede indicar menor penetración de Electrohogar/Tecnología.
""")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("RESUMEN — LECCIÓN 5 COMPLETADA")
print("=" * 65)
print("""
✅ pairplot generado      → 05_pairplot_variables_numericas.png
✅ violinplot generado    → 05_violinplot_monto_por_categoria.png
✅ jointplot generado     → 05_jointplot_precio_vs_monto.png
✅ heatmap generado       → 05_heatmap_correlacion_triangular.png
✅ FacetGrid x2 generado  → 05_facetgrid_edad_canal_genero.png
                          → 05_facetgrid_monto_por_region.png
✅ Estilos, paletas y tamaños ajustados consistentemente
✅ Insights documentados en cada sección

🔜 PRÓXIMA LECCIÓN: Matplotlib — visualizaciones personalizadas y
   presentación final con subplots, anotaciones y exportación.
""")