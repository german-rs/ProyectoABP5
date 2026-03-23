# =============================================================================
# LECCIÓN 2: CONCEPTOS BÁSICOS DE ESTADÍSTICA DESCRIPTIVA
# Proyecto: ComercioYA — EDA para decisiones comerciales
# Módulo 5 — Alkemy
# =============================================================================
# OBJETIVO: Aplicar medidas de tendencia central (media, mediana, moda),
# dispersión (varianza, desviación estándar) y posición (cuartiles,
# percentiles). Generar histogramas y boxplots. Identificar outliers con IQR.
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica (para exportar sin pantalla)
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
RUTA_RAIZ    = Path(__file__).resolve().parent.parent
RUTA_RAW     = RUTA_RAIZ / "data" / "raw"
RUTA_PROC    = RUTA_RAIZ / "data" / "processed"
RUTA_GRAFICOS = RUTA_RAIZ / "outputs" / "graphs"

RUTA_PROC.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FUNCIÓN UTILITARIA: exportar gráfico
# Se centraliza en una función para garantizar consistencia en todos los
# gráficos del proyecto (dpi, bbox, tight_layout).
# =============================================================================
def guardar_grafico(nombre: str):
    plt.tight_layout()
    ruta = RUTA_GRAFICOS / f"{nombre}.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Gráfico exportado: outputs/graphs/{nombre}.png")

# =============================================================================
# FUNCIÓN UTILITARIA: detección de outliers por IQR
# Se elige IQR en lugar de Z-score porque no se puede asumir distribución
# normal en variables como precio_unitario o monto_total.
# =============================================================================
def detectar_outliers_iqr(serie: pd.Series) -> pd.Series:
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    return serie[(serie < Q1 - 1.5 * IQR) | (serie > Q3 + 1.5 * IQR)]

# =============================================================================
# SECCIÓN 1: CARGA Y LIMPIEZA BÁSICA
# Se aplica la limpieza mínima necesaria para que las estadísticas sean
# confiables. La limpieza profunda se documenta aquí brevemente ya que
# el foco de esta lección es la estadística descriptiva.
# =============================================================================
print("=" * 65)
print("LECCIÓN 2 — ESTADÍSTICA DESCRIPTIVA")
print("=" * 65)

archivo = RUTA_RAW / "E-ShopNow (Proyecto ABP M2).csv"
datos_raw = pd.read_csv(archivo, sep=";")

# Paso 1: eliminar duplicados exactos (detectados en Lección 1: 20 filas)
datos = datos_raw.drop_duplicates()
print(f"\n✅ Duplicados eliminados: {len(datos_raw) - len(datos)} filas")

# Paso 2: unificar categorías inconsistentes en region_cliente
# 'RM' y 'Metropolitana' representan la misma región que 'Región Metropolitana'
datos = datos.assign(
    region_cliente=datos["region_cliente"].replace({
        "RM": "Región Metropolitana",
        "Metropolitana": "Región Metropolitana"
    })
)

# Paso 3: imputar nulos
# descuento_pct nulo → 0 (ausencia de descuento, no dato desconocido)
# edad_cliente nulo  → mediana (más robusta que la media ante asimetría)
mediana_edad = datos["edad_cliente"].median()
datos = datos.assign(
    descuento_pct=datos["descuento_pct"].fillna(0),
    edad_cliente=datos["edad_cliente"].fillna(mediana_edad)
)

# Paso 4: convertir fecha_venta a datetime
datos = datos.assign(
    fecha_venta=pd.to_datetime(datos["fecha_venta"], dayfirst=True)
)

# Paso 5: crear variable derivada monto_total
# monto_total = precio × cantidad × (1 - descuento)
# Esta variable no existe en el raw pero es la métrica de negocio más relevante
datos = datos.assign(
    monto_total=datos["precio_unitario"] * datos["cantidad"] * (1 - datos["descuento_pct"])
)

print(f"✅ Dataset limpio listo: {len(datos)} filas × {datos.shape[1]} columnas")

# Guardar dataset limpio como checkpoint para lecciones posteriores
archivo_procesado = RUTA_PROC / "datos_limpios.csv"
datos.to_csv(archivo_procesado, index=False, sep=";")
print(f"✅ Dataset limpio guardado en: data/processed/datos_limpios.csv")

# Variables numéricas a analizar en esta lección
variables_numericas = [
    "edad_cliente",
    "precio_unitario",
    "cantidad",
    "descuento_pct",
    "antiguedad_vendedor",
    "monto_total"
]

# =============================================================================
# SECCIÓN 2: MEDIDAS DE TENDENCIA CENTRAL
# Media, mediana y moda capturan el "centro" de la distribución.
# Se calculan las tres porque cada una es sensible a distintos fenómenos:
# - Media: sensible a outliers
# - Mediana: robusta ante outliers, mejor para distribuciones asimétricas
# - Moda: útil en variables discretas o con valores concentrados
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 2: MEDIDAS DE TENDENCIA CENTRAL")
print("=" * 65)

print(f"\n{'Variable':<25} {'Media':>12} {'Mediana':>12} {'Moda':>12}")
print("-" * 65)
for col in variables_numericas:
    media   = datos[col].mean()
    mediana = datos[col].median()
    moda    = datos[col].mode()[0]
    print(f"{col:<25} {media:>12.2f} {mediana:>12.2f} {moda:>12.2f}")

# Hallazgo clave: precio_unitario tiene media=364.032 y mediana=450.000 CLP.
# Que la mediana sea MAYOR que la media indica que los precios bajos (Belleza,
# Vestuario) jalan la media hacia abajo. No es una distribución simétrica.

# Hallazgo clave: monto_total tiene media=717.261 y mediana=450.000 CLP.
# La media supera ampliamente a la mediana → sesgo positivo (skew=0.75),
# probablemente por compras de Electrohogar con precios muy altos.

# =============================================================================
# SECCIÓN 3: MEDIDAS DE DISPERSIÓN
# Varianza y desviación estándar miden cuánto se alejan los datos del centro.
# Se agrega el coeficiente de variación (CV = std/media) para comparar
# dispersiones entre variables con distintas escalas.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 3: MEDIDAS DE DISPERSIÓN")
print("=" * 65)

print(f"\n{'Variable':<25} {'Varianza':>16} {'Desv. Std':>14} {'CV %':>8}")
print("-" * 65)
for col in variables_numericas:
    varianza = datos[col].var()
    std      = datos[col].std()
    cv       = (std / datos[col].mean() * 100) if datos[col].mean() != 0 else 0
    print(f"{col:<25} {varianza:>16.2f} {std:>14.2f} {cv:>8.1f}%")

# Hallazgo: monto_total tiene CV=107%, lo que indica dispersión extremadamente
# alta. Esto refleja la heterogeneidad del catálogo: desde cosméticos (15.000 CLP)
# hasta electrónicos (hasta 2.400.000 CLP).

# =============================================================================
# SECCIÓN 4: CUARTILES Y PERCENTILES
# Los cuartiles dividen los datos en 4 partes iguales.
# Se calculan también P10 y P90 para identificar los extremos de la distribución
# sin ser tan sensibles como el mínimo y máximo absolutos.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 4: CUARTILES Y PERCENTILES")
print("=" * 65)

percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
tabla_percentiles = datos[variables_numericas].quantile(percentiles)
tabla_percentiles.index = ["P10", "Q1 (P25)", "Mediana (P50)", "Q3 (P75)", "P90"]

print()
print(tabla_percentiles.round(2).to_string())

# Hallazgo: el 10% de los clientes más jóvenes tiene menos de 23 años,
# mientras que el 10% mayor supera los 55 años → base de clientes adulta.
# El 25% de las ventas de menor monto está por debajo de 45.000 CLP,
# mientras el 25% superior supera 1.350.000 CLP → fuerte asimetría.

# =============================================================================
# SECCIÓN 5: ASIMETRÍA Y CURTOSIS
# Skewness (asimetría) indica si la distribución se inclina a un lado.
# Kurtosis indica si la distribución tiene colas pesadas o es más plana.
# Estos valores orientan qué transformaciones aplicar antes de modelar.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 5: ASIMETRÍA (SKEWNESS) Y CURTOSIS (KURTOSIS)")
print("=" * 65)

print(f"\n{'Variable':<25} {'Skewness':>12} {'Kurtosis':>12}  Interpretación")
print("-" * 75)
for col in variables_numericas:
    skew = datos[col].skew()
    kurt = datos[col].kurt()
    if abs(skew) < 0.5:
        interp = "distribución aproximadamente simétrica"
    elif skew > 0:
        interp = "sesgo positivo (cola derecha larga)"
    else:
        interp = "sesgo negativo (cola izquierda larga)"
    print(f"{col:<25} {skew:>12.2f} {kurt:>12.2f}  {interp}")

# =============================================================================
# SECCIÓN 6: DETECCIÓN DE OUTLIERS (IQR)
# Se usa IQR porque es robusto ante distribuciones no normales.
# Se documenta la cantidad de outliers y su rango para decidir el tratamiento.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 6: DETECCIÓN DE OUTLIERS (MÉTODO IQR)")
print("=" * 65)

print(f"\n{'Variable':<25} {'Outliers':>10} {'% del total':>12}  Rango outlier")
print("-" * 70)
for col in variables_numericas:
    Q1  = datos[col].quantile(0.25)
    Q3  = datos[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    outliers = detectar_outliers_iqr(datos[col])
    pct = len(outliers) / len(datos) * 100
    print(f"{col:<25} {len(outliers):>10} {pct:>11.1f}%  "
          f"< {lim_inf:.0f} o > {lim_sup:.0f}")

# Hallazgo: precio_unitario y monto_total no tienen outliers por IQR porque
# el rango intercuartílico es muy amplio (hay productos de 12.000 a 800.000 CLP).
# edad_cliente tiene 12 outliers (clientes mayores de 65.5 años → válidos).
# antiguedad_vendedor tiene 14 outliers (vendedores con más de 11 años → válidos).
# Ningún outlier parece ser un error de carga; son valores extremos reales.

# =============================================================================
# SECCIÓN 7: VISUALIZACIONES — HISTOGRAMAS
# Los histogramas muestran la forma de la distribución de cada variable.
# Se usa una figura con subplots para comparar todas las variables a la vez.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 7: GENERANDO VISUALIZACIONES")
print("=" * 65)

print("\n   → Generando histogramas...")

fig, ejes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Distribución de Variables Numéricas — ComercioYA", fontsize=14, fontweight="bold")
ejes = ejes.flatten()

colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

for i, col in enumerate(variables_numericas):
    ejes[i].hist(datos[col], bins=20, color=colores[i], edgecolor="white", alpha=0.85)
    ejes[i].set_title(col.replace("_", " ").title(), fontsize=11)
    ejes[i].set_xlabel("Valor", fontsize=9)
    ejes[i].set_ylabel("Frecuencia", fontsize=9)
    media   = datos[col].mean()
    mediana = datos[col].median()
    ejes[i].axvline(media,   color="red",    linestyle="--", linewidth=1.2, label=f"Media: {media:.0f}")
    ejes[i].axvline(mediana, color="orange", linestyle="-",  linewidth=1.2, label=f"Mediana: {mediana:.0f}")
    ejes[i].legend(fontsize=8)

guardar_grafico("02_histogramas_variables_numericas")

# =============================================================================
# SECCIÓN 8: VISUALIZACIONES — BOXPLOTS
# Los boxplots muestran cuartiles y outliers visualmente.
# Se normalizan las variables por separado según escala para que sean legibles.
# =============================================================================
print("\n   → Generando boxplots...")

# Grupo 1: variables monetarias (escala CLP)
vars_monetarias = ["precio_unitario", "monto_total"]
fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Boxplots — Variables Monetarias (CLP) — ComercioYA",
             fontsize=13, fontweight="bold")

for i, col in enumerate(vars_monetarias):
    ejes[i].boxplot(datos[col], patch_artist=True,
                    boxprops=dict(facecolor=colores[i], alpha=0.7))
    ejes[i].set_title(col.replace("_", " ").title(), fontsize=11)
    ejes[i].set_ylabel("Valor (CLP)", fontsize=9)
    ejes[i].set_xlabel(col, fontsize=9)
    ejes[i].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1_000_000:.1f}M" if x >= 1_000_000 else f"${x/1_000:.0f}K")
    )

guardar_grafico("02_boxplot_variables_monetarias")

# Grupo 2: resto de variables
vars_resto = ["edad_cliente", "cantidad", "descuento_pct", "antiguedad_vendedor"]
fig, ejes = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle("Boxplots — Variables Numéricas — ComercioYA",
             fontsize=13, fontweight="bold")

for i, col in enumerate(vars_resto):
    ejes[i].boxplot(datos[col], patch_artist=True,
                    boxprops=dict(facecolor=colores[i + 2], alpha=0.7))
    ejes[i].set_title(col.replace("_", " ").title(), fontsize=11)
    ejes[i].set_ylabel("Valor", fontsize=9)
    ejes[i].set_xlabel(col, fontsize=9)

guardar_grafico("02_boxplot_variables_resto")

# =============================================================================
# SECCIÓN 9: RESUMEN DE HALLAZGOS
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 9: RESUMEN DE HALLAZGOS — LECCIÓN 2")
print("=" * 65)

print("""
📋 HALLAZGOS ESTADÍSTICOS CLAVE:

   1. PRECIO UNITARIO: mediana (450.000 CLP) > media (364.032 CLP)
      → Los productos baratos (Belleza, Vestuario) arrastran la media
        hacia abajo. Distribución bimodal: productos económicos vs premium.

   2. MONTO TOTAL: CV=107%, media=717.261 CLP, mediana=450.000 CLP
      → Altísima dispersión. Las compras de Electrohogar elevan la media.
        La mediana es un mejor indicador del ticket promedio real.

   3. EDAD CLIENTE: distribución aproximadamente simétrica (skew=0.41)
      → Media=37 años, mediana=38 años. Base de clientes adulta-joven.
        12 outliers válidos (clientes mayores de 65 años).

   4. CANTIDAD: valores entre 1 y 3, sin outliers.
      → Los clientes no compran en grandes volúmenes. Canal minorista típico.

   5. DESCUENTO: el 25% de las ventas no tuvo descuento (Q1=0%).
      → El descuento máximo es 10%. Política de descuentos moderada.

   6. ANTIGÜEDAD VENDEDOR: 14 outliers (vendedores con más de 11 años).
      → Son valores válidos que representan vendedores senior.

📁 ARCHIVOS GENERADOS:
   • data/processed/datos_limpios.csv
   • outputs/graphs/02_histogramas_variables_numericas.png
   • outputs/graphs/02_boxplot_variables_monetarias.png
   • outputs/graphs/02_boxplot_variables_resto.png
""")