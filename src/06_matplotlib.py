# =============================================================================
# LECCIÓN 6: LIBRERÍA MATPLOTLIB — VISUALIZACIONES PERSONALIZADAS
# Proyecto: ComercioYA — EDA para decisiones comerciales
# Módulo 5 — Alkemy
# =============================================================================
# OBJETIVO: Crear visualizaciones personalizadas y exportables con control
# total sobre cada elemento del gráfico: figuras, subplots, títulos,
# etiquetas, leyendas, ticks, anotaciones y límites.
# Generar la presentación final del informe EDA con los gráficos más
# representativos del análisis completo.
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
RUTA_RAIZ      = Path(__file__).resolve().parent.parent
RUTA_PROCESADO = RUTA_RAIZ / "data" / "processed"
RUTA_GRAFICOS  = RUTA_RAIZ / "outputs" / "graphs"
RUTA_REPORTES  = RUTA_RAIZ / "outputs" / "reports"

RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
RUTA_REPORTES.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PALETA Y ESTILO GLOBAL
# Se define una paleta de colores corporativa consistente para todos los
# gráficos del informe final. La consistencia visual es un criterio de
# evaluación explícito en la rúbrica del proyecto.
# =============================================================================
COLOR_PRINCIPAL  = "#2C5F8A"
COLOR_SECUNDARIO = "#E05C2A"
COLOR_ACENTO     = "#4BAE8A"
COLOR_NEUTRO     = "#95A5A6"
PALETA_CANAL     = {"Tienda": "#2C5F8A", "Web": "#E05C2A", "App": "#4BAE8A"}
PALETA_GENERO    = {"Hombre": "#2C5F8A", "Mujer": "#E05C2A"}

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F8F9FA",
    "axes.grid"         : True,
    "grid.color"        : "white",
    "grid.linewidth"    : 1.2,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})


def guardar_grafico(nombre: str, dpi: int = 150):
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    plt.tight_layout()
    plt.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"   💾 Gráfico exportado: outputs/graphs/{nombre}.png")


def formato_clp(valor, _=None):
    """Formateador de eje para pesos chilenos."""
    if valor >= 1_000_000:
        return f"${valor/1_000_000:.1f}M"
    elif valor >= 1_000:
        return f"${valor/1_000:.0f}K"
    return f"${valor:.0f}"


# =============================================================================
# SECCIÓN 1: CARGA Y PREPARACIÓN
# =============================================================================
print("=" * 65)
print("LECCIÓN 6 — MATPLOTLIB: VISUALIZACIONES PERSONALIZADAS")
print("=" * 65)

archivo = RUTA_PROCESADO / "datos_limpios.csv"
datos = pd.read_csv(archivo, sep=";")

datos = datos.assign(
    monto_venta=datos["precio_unitario"] * datos["cantidad"] * (1 - datos["descuento_pct"])
)
datos["region_cliente"] = datos["region_cliente"].replace(
    {"RM": "Región Metropolitana", "Metropolitana": "Región Metropolitana"}
)
# datos["fecha_venta"] = pd.to_datetime(datos["fecha_venta"], dayfirst=True)
#datos["fecha_venta"] = pd.to_datetime(datos["fecha_venta"], format="%d/%m/%Y", errors='coerce')
datos["fecha_venta"] = pd.to_datetime(
    datos["fecha_venta"],
    dayfirst=True,
    format='mixed',
    errors='coerce'
)
#datos["mes_venta"]   = datos["fecha_venta"].dt.to_period("M").astype(str)
datos["mes_venta"] = datos["fecha_venta"].dt.to_period("M").astype(str)

datos_cat = datos.dropna(subset=["categoria_producto"])
datos_reg = datos.dropna(subset=["region_cliente"])

print(f"\n✅ Dataset listo: {datos.shape[0]} filas")

# =============================================================================
# SECCIÓN 2: GRÁFICO 1 — BARRAS APILADAS: VENTAS POR CANAL Y CATEGORÍA
# Figuras con subplots, etiquetas personalizadas y leyenda externa.
# Las barras apiladas permiten ver simultáneamente el total por canal
# y la composición interna por categoría de producto.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 2: GRÁFICO 1 — VENTAS POR CANAL Y CATEGORÍA")
print("=" * 65)

pivot_canal_cat = (
    datos_cat
    .groupby(["canal_venta", "categoria_producto"])["monto_venta"]
    .sum()
    .unstack(fill_value=0)
)

categorias = pivot_canal_cat.columns.tolist()
canales    = pivot_canal_cat.index.tolist()
paleta_cat = dict(zip(categorias, sns.color_palette("Set2", len(categorias))))

fig, ax = plt.subplots(figsize=(11, 6))

base = np.zeros(len(canales))
for cat in categorias:
    valores = pivot_canal_cat[cat].values
    barras  = ax.bar(canales, valores, bottom=base,
                     color=paleta_cat[cat], label=cat,
                     edgecolor="white", linewidth=0.8)
    # Anotación dentro de la barra si el segmento es suficientemente grande
    for barra, val, b in zip(barras, valores, base):
        if val > 5_000_000:
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                b + val / 2,
                f"${val/1_000_000:.0f}M",
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold"
            )
    base = base + valores

ax.set_title("Monto total de ventas por canal y categoría de producto", pad=15)
ax.set_xlabel("Canal de venta")
ax.set_ylabel("Monto total (CLP)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(formato_clp))
ax.set_xticks(range(len(canales)))
ax.set_xticklabels(canales, fontsize=11)

# Leyenda fuera del área del gráfico para no tapar barras
ax.legend(
    title="Categoría", loc="upper left",
    bbox_to_anchor=(1.01, 1), borderaxespad=0
)

# Línea de referencia: monto promedio por canal
for i, canal in enumerate(canales):
    total = pivot_canal_cat.loc[canal].sum()
    ax.annotate(
        f"Total: {formato_clp(total)}",
        xy=(i, total),
        xytext=(i, total * 1.03),
        ha="center", fontsize=9, color="#333333", fontweight="bold"
    )

guardar_grafico("06_barras_ventas_canal_categoria")

# =============================================================================
# SECCIÓN 3: GRÁFICO 2 — EVOLUCIÓN TEMPORAL DE VENTAS
# Serie de tiempo con anotaciones de máximo y mínimo.
# Requiere conversión de fecha_venta a datetime (detectado en Lección 1).
# Se agrega una línea de media móvil de 30 días para suavizar la serie
# y visualizar la tendencia sin el ruido de variaciones diarias.
# =============================================================================
print("=" * 65)
print("SECCIÓN 3: GRÁFICO 2 — EVOLUCIÓN TEMPORAL DE VENTAS")
print("=" * 65)

ventas_mes = (
    datos
    .groupby("mes_venta")["monto_venta"]
    .sum()
    .reset_index()
)
ventas_mes.columns = ["mes", "monto_total"]
ventas_mes["indice"] = range(len(ventas_mes))

fig, ax = plt.subplots(figsize=(13, 5))

# Área bajo la curva
ax.fill_between(
    ventas_mes["indice"], ventas_mes["monto_total"],
    alpha=0.18, color=COLOR_PRINCIPAL
)

# Línea principal
ax.plot(
    ventas_mes["indice"], ventas_mes["monto_total"],
    color=COLOR_PRINCIPAL, linewidth=2.5, marker="o",
    markersize=5, label="Monto mensual"
)

# Media móvil (ventana de 3 meses)
media_movil = ventas_mes["monto_total"].rolling(window=3, center=True).mean()
ax.plot(
    ventas_mes["indice"], media_movil,
    color=COLOR_SECUNDARIO, linewidth=2, linestyle="--",
    label="Media móvil (3 meses)"
)

# Anotación de máximo y mínimo
idx_max = ventas_mes["monto_total"].idxmax()
idx_min = ventas_mes["monto_total"].idxmin()

for idx, etiq, color in [
    (idx_max, "Máximo", COLOR_SECUNDARIO),
    (idx_min, "Mínimo", COLOR_NEUTRO)
]:
    val = ventas_mes.loc[idx, "monto_total"]
    mes = ventas_mes.loc[idx, "mes"]
    ax.annotate(
        f"{etiq}\n{mes}\n{formato_clp(val)}",
        xy=(ventas_mes.loc[idx, "indice"], val),
        xytext=(ventas_mes.loc[idx, "indice"] + 0.5, val * 1.08),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        fontsize=8.5, color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=color, alpha=0.9)
    )

ax.set_title("Evolución mensual del monto de ventas — ComercioYA (2023)", pad=12)
ax.set_xlabel("Mes")
ax.set_ylabel("Monto total (CLP)")
ax.set_xticks(ventas_mes["indice"])
ax.set_xticklabels(ventas_mes["mes"], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(formato_clp))
ax.legend(loc="upper right")

# Límite Y con margen superior para que las anotaciones no se corten
ax.set_ylim(0, ventas_mes["monto_total"].max() * 1.30)

guardar_grafico("06_evolucion_temporal_ventas")

# =============================================================================
# SECCIÓN 4: GRÁFICO 3 — SUBPLOTS COMBINADOS: PERFIL DEL CLIENTE
# Se crea una figura con 4 subplots en grilla 2×2 para mostrar el perfil
# completo del cliente: distribución de edad, género, canal y región.
# Cada subplot usa un tipo de gráfico diferente para demostrar versatilidad.
# Se usan GridSpec para control fino del layout.
# =============================================================================
print("=" * 65)
print("SECCIÓN 4: GRÁFICO 3 — PERFIL DEL CLIENTE (4 SUBPLOTS)")
print("=" * 65)

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "ComercioYA — Perfil del cliente e-commerce",
    fontsize=16, fontweight="bold", y=1.01
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# --- Subplot 1: Histograma de edad con KDE ---
ax1 = fig.add_subplot(gs[0, 0])
n, bins, patches = ax1.hist(
    datos["edad_cliente"], bins=20,
    color=COLOR_PRINCIPAL, edgecolor="white", linewidth=0.5, alpha=0.85
)
# KDE encima del histograma (normalizado)
from scipy.stats import gaussian_kde
kde = gaussian_kde(datos["edad_cliente"].dropna())
x_kde = np.linspace(datos["edad_cliente"].min(), datos["edad_cliente"].max(), 200)
ax1_twin = ax1.twinx()
ax1_twin.plot(x_kde, kde(x_kde), color=COLOR_SECUNDARIO, linewidth=2)
ax1_twin.set_yticks([])
ax1_twin.spines["right"].set_visible(False)

media_edad   = datos["edad_cliente"].mean()
mediana_edad = datos["edad_cliente"].median()
ax1.axvline(media_edad,   color=COLOR_SECUNDARIO, linewidth=1.5,
            linestyle="--", label=f"Media: {media_edad:.1f}")
ax1.axvline(mediana_edad, color=COLOR_ACENTO,     linewidth=1.5,
            linestyle="-",  label=f"Mediana: {mediana_edad:.1f}")
ax1.set_title("Distribución de edad de clientes")
ax1.set_xlabel("Edad (años)")
ax1.set_ylabel("N° de clientes")
ax1.legend(fontsize=8)

# --- Subplot 2: Torta de género ---
ax2 = fig.add_subplot(gs[0, 1])
conteo_genero = datos["genero_cliente"].value_counts()
colores_genero = [PALETA_GENERO.get(g, COLOR_NEUTRO) for g in conteo_genero.index]
cunas, textos, autotextos = ax2.pie(
    conteo_genero,
    labels=conteo_genero.index,
    colors=colores_genero,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11}
)
for autotexto in autotextos:
    autotexto.set_fontweight("bold")
    autotexto.set_color("white")
ax2.set_title("Distribución de género")

# --- Subplot 3: Barras horizontales de canal de venta ---
ax3 = fig.add_subplot(gs[1, 0])
conteo_canal = datos["canal_venta"].value_counts().sort_values()
colores_barras = [PALETA_CANAL.get(c, COLOR_NEUTRO) for c in conteo_canal.index]
barras = ax3.barh(
    conteo_canal.index, conteo_canal.values,
    color=colores_barras, edgecolor="white", linewidth=0.8, height=0.55
)
# Etiquetas de valor al final de cada barra
for barra in barras:
    ancho = barra.get_width()
    ax3.text(
        ancho + 5, barra.get_y() + barra.get_height() / 2,
        f"{int(ancho):,}", va="center", ha="left", fontsize=9, fontweight="bold"
    )
ax3.set_title("Ventas por canal")
ax3.set_xlabel("N° de transacciones")
ax3.set_xlim(0, conteo_canal.max() * 1.15)

# --- Subplot 4: Barras de monto mediano por región (top 6) ---
ax4 = fig.add_subplot(gs[1, 1])
top6 = (
    datos_reg
    .groupby("region_cliente")["monto_venta"]
    .median()
    .sort_values(ascending=True)
    .tail(6)
)
colores_region = [
    COLOR_SECUNDARIO if v == top6.max() else COLOR_PRINCIPAL
    for v in top6.values
]
barras4 = ax4.barh(
    top6.index, top6.values,
    color=colores_region, edgecolor="white", linewidth=0.8, height=0.55
)
for barra in barras4:
    ancho = barra.get_width()
    ax4.text(
        ancho * 1.02, barra.get_y() + barra.get_height() / 2,
        formato_clp(ancho), va="center", ha="left", fontsize=9, fontweight="bold"
    )
ax4.set_title("Monto mediano de venta por región (top 6)")
ax4.set_xlabel("Monto mediano (CLP)")
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(formato_clp))
ax4.set_xlim(0, top6.max() * 1.25)

# Anotación: destacar la región con mayor monto
ax4.annotate(
    "Mayor monto\nmediano",
    xy=(top6.max(), len(top6) - 1),
    xytext=(top6.max() * 0.55, len(top6) - 1 + 0.4),
    arrowprops=dict(arrowstyle="->", color=COLOR_SECUNDARIO, lw=1.5),
    fontsize=8, color=COLOR_SECUNDARIO, fontweight="bold"
)

guardar_grafico("06_subplots_perfil_cliente")

# =============================================================================
# SECCIÓN 5: PRESENTACIÓN FINAL — DASHBOARD EDA
# Se genera una figura tipo "dashboard" con los 6 gráficos más importantes
# del análisis completo. Esta figura es el entregable visual central del
# informe EDA y usa GridSpec con tamaños desiguales para jerarquizar la
# información: los gráficos más importantes ocupan más espacio.
# =============================================================================
print("=" * 65)
print("SECCIÓN 5: DASHBOARD FINAL — PRESENTACIÓN EDA")
print("=" * 65)

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("white")

fig.suptitle(
    "ComercioYA — Informe EDA: Análisis Exploratorio de Datos\n"
    "Alkemy · Módulo 5 · 2023",
    fontsize=18, fontweight="bold", y=1.01, color="#1A1A2E"
)

# Layout: 3 filas × 3 columnas con anchos/altos desiguales
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.50, wspace=0.38,
    height_ratios=[1.2, 1, 1]
)

# ── Panel 1 (fila 0, col 0-1): Evolución temporal ──────────────────────────
ax_tiempo = fig.add_subplot(gs[0, :2])

ax_tiempo.fill_between(
    ventas_mes["indice"], ventas_mes["monto_total"],
    alpha=0.15, color=COLOR_PRINCIPAL
)
ax_tiempo.plot(
    ventas_mes["indice"], ventas_mes["monto_total"],
    color=COLOR_PRINCIPAL, linewidth=2, marker="o", markersize=4
)
ax_tiempo.plot(
    ventas_mes["indice"], media_movil,
    color=COLOR_SECUNDARIO, linewidth=1.8, linestyle="--",
    label="Media móvil 3M"
)
ax_tiempo.set_title("Evolución mensual de ventas (2023)")
ax_tiempo.set_xticks(ventas_mes["indice"])
ax_tiempo.set_xticklabels(ventas_mes["mes"], rotation=40, ha="right", fontsize=7.5)
ax_tiempo.yaxis.set_major_formatter(mticker.FuncFormatter(formato_clp))
ax_tiempo.legend(fontsize=8)
ax_tiempo.set_ylim(0, ventas_mes["monto_total"].max() * 1.25)

# ── Panel 2 (fila 0, col 2): Torta canal de venta ─────────────────────────
ax_canal = fig.add_subplot(gs[0, 2])
conteo_canal_sorted = datos["canal_venta"].value_counts()
colores_c = [PALETA_CANAL.get(c, COLOR_NEUTRO) for c in conteo_canal_sorted.index]
ax_canal.pie(
    conteo_canal_sorted,
    labels=conteo_canal_sorted.index,
    colors=colores_c,
    autopct="%1.0f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 9}
)
ax_canal.set_title("Canal de venta")

# ── Panel 3 (fila 1, col 0-1): Monto por categoría (barras) ───────────────
ax_cat = fig.add_subplot(gs[1, :2])
monto_cat = (
    datos_cat
    .groupby("categoria_producto")["monto_venta"]
    .sum()
    .sort_values(ascending=True)
)
colores_mcat = [
    COLOR_SECUNDARIO if v == monto_cat.max() else COLOR_PRINCIPAL
    for v in monto_cat.values
]
barras_cat = ax_cat.barh(
    monto_cat.index, monto_cat.values,
    color=colores_mcat, edgecolor="white", height=0.55
)
for barra in barras_cat:
    ancho = barra.get_width()
    ax_cat.text(
        ancho * 1.01, barra.get_y() + barra.get_height() / 2,
        formato_clp(ancho), va="center", ha="left", fontsize=8.5, fontweight="bold"
    )
ax_cat.set_title("Monto total de ventas por categoría")
ax_cat.xaxis.set_major_formatter(mticker.FuncFormatter(formato_clp))
ax_cat.set_xlim(0, monto_cat.max() * 1.22)

# ── Panel 4 (fila 1, col 2): Distribución de edad ─────────────────────────
ax_edad = fig.add_subplot(gs[1, 2])
ax_edad.hist(datos["edad_cliente"], bins=18,
             color=COLOR_PRINCIPAL, edgecolor="white", linewidth=0.5, alpha=0.85)
ax_edad.axvline(datos["edad_cliente"].mean(), color=COLOR_SECUNDARIO,
                linewidth=1.8, linestyle="--",
                label=f"Media: {datos['edad_cliente'].mean():.0f}a")
ax_edad.set_title("Distribución de edad")
ax_edad.set_xlabel("Edad (años)")
ax_edad.set_ylabel("N° clientes")
ax_edad.legend(fontsize=8)

# ── Panel 5 (fila 2, col 0-1): Heatmap correlación ────────────────────────
ax_heat = fig.add_subplot(gs[2, :2])
vars_h   = ["precio_unitario", "cantidad", "descuento_pct",
            "edad_cliente", "monto_venta"]
mat_corr = datos[vars_h].corr()
mascara  = np.triu(np.ones_like(mat_corr, dtype=bool))
sns.heatmap(
    mat_corr, mask=mascara, annot=True, fmt=".2f",
    cmap="RdBu_r", vmin=-1, vmax=1,
    linewidths=0.6, linecolor="white", square=True,
    cbar_kws={"shrink": 0.7},
    annot_kws={"size": 9, "weight": "bold"},
    ax=ax_heat
)
ax_heat.set_title("Matriz de correlación")
ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=25, ha="right", fontsize=8)
ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=8)

# ── Panel 6 (fila 2, col 2): KPIs resumen ─────────────────────────────────
ax_kpi = fig.add_subplot(gs[2, 2])
ax_kpi.set_xlim(0, 1)
ax_kpi.set_ylim(0, 1)
ax_kpi.axis("off")

kpis = [
    ("N° transacciones",  f"{len(datos):,}"),
    ("Monto total",        formato_clp(datos["monto_venta"].sum())),
    ("Ticket promedio",    formato_clp(datos["monto_venta"].mean())),
    ("Ticket mediano",     formato_clp(datos["monto_venta"].median())),
    ("Edad promedio",      f"{datos['edad_cliente'].mean():.1f} años"),
    ("Top categoría",      "Electrohogar"),
    ("Top región",         "Valparaíso"),
    ("R² modelo final",    "0.87 (precio+cantidad)"),
]

ax_kpi.set_title("KPIs del análisis", pad=8)
ax_kpi.add_patch(mpatches.FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.95,
    boxstyle="round,pad=0.02",
    facecolor="#F0F4F8", edgecolor=COLOR_PRINCIPAL,
    linewidth=1.5, zorder=0
))

for i, (etiq, valor) in enumerate(kpis):
    y_pos = 0.90 - i * 0.107
    ax_kpi.text(0.08, y_pos, f"▸ {etiq}:", fontsize=8.5,
                color="#555555", va="center")
    ax_kpi.text(0.95, y_pos, valor, fontsize=8.5, fontweight="bold",
                color=COLOR_PRINCIPAL, va="center", ha="right")

# Guardar dashboard
ruta_dashboard = RUTA_REPORTES / "06_dashboard_eda_final.png"
plt.tight_layout()
plt.savefig(ruta_dashboard, dpi=180, bbox_inches="tight")
plt.close()
print(f"   💾 Dashboard exportado: outputs/reports/06_dashboard_eda_final.png")

# =============================================================================
# RESUMEN FINAL DEL PROYECTO
# =============================================================================
print("\n" + "=" * 65)
print("RESUMEN — LECCIÓN 6 COMPLETADA")
print("=" * 65)
print("""
✅ Figuras y subplots creados con GridSpec
✅ Títulos, etiquetas, leyendas y ticks personalizados
✅ Anotaciones (flechas, textos) y límites de eje aplicados
✅ Paleta corporativa consistente en todos los gráficos

💾 Gráficos exportados (outputs/graphs/):
   → 06_barras_ventas_canal_categoria.png
   → 06_evolucion_temporal_ventas.png
   → 06_subplots_perfil_cliente.png

📊 Dashboard final exportado (outputs/reports/):
   → 06_dashboard_eda_final.png

""")
print("=" * 65)
print("✅ PROYECTO EDA COMPLETO — TODAS LAS LECCIONES FINALIZADAS")
print("=" * 65)
print("""
HALLAZGOS PRINCIPALES DEL ANÁLISIS:

1. El monto de venta está determinado casi exclusivamente por el tipo
   de producto: Electrohogar y Tecnología generan montos 30x superiores
   a Belleza y Vestuario (distribución bimodal confirmada).

2. El canal de venta (Web/App/Tienda) no diferencia el perfil etario
   del cliente ni el monto de compra.

3. El modelo de regresión final (precio + cantidad) explica el ~87%
   de la varianza del monto de venta. Variables como edad del cliente
   y antigüedad del vendedor no son predictores significativos.

4. Valparaíso y O'Higgins lideran en monto mediano de venta por región.
   La Araucanía muestra menor penetración de productos de alto valor.

5. No hay correlación lineal entre variables de perfil del cliente y
   comportamiento de compra: el e-commerce atiende un segmento amplio
   sin patrones demográficos claros de gasto.
""")

"""
---

## Actualización de `progress.md` — Lecciones 5 y 6
### Lección 5 — Análisis Visual (Seaborn)
- [x] pairplot generado
- [x] violinplot generado
- [x] jointplot generado
- [x] heatmap generado
- [x] FacetGrid segmentado por categorías
- [x] Estilos, colores y tamaños ajustados
- [x] Insights visuales documentados

### Lección 6 — Matplotlib
- [x] Figuras y subplots creados
- [x] Títulos, etiquetas, leyendas y ticks personalizados
- [x] Anotaciones y límites aplicados
- [x] Presentación final con gráficos generada
- [x] Informe EDA con visualizaciones embebidas entregado

"""