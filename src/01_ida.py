# =============================================================================
# LECCIÓN 1: ANÁLISIS EXPLORATORIO DE DATOS — IDA (Initial Data Analysis)
# Proyecto: ComercioYA — EDA para decisiones comerciales
# Módulo 5 — Alkemy
# =============================================================================
# OBJETIVO: Comprender el propósito del EDA, cargar el dataset real de
# E-ShopNow, clasificar variables, detectar nulos, duplicados e
# inconsistencias, y documentar los primeros hallazgos.
# =============================================================================

from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# Se usan rutas relativas con pathlib para garantizar portabilidad entre
# sistemas operativos. La raíz del proyecto es el directorio padre de /src.
# -----------------------------------------------------------------------------
RUTA_RAIZ      = Path(__file__).resolve().parent.parent
RUTA_RAW       = RUTA_RAIZ / "data" / "raw"
RUTA_PROCESADO = RUTA_RAIZ / "data" / "processed"

# Crear carpeta processed si no existe
RUTA_PROCESADO.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECCIÓN 1: CARGA DEL DATASET
# Se carga el archivo original sin modificarlo (principio de datos crudos).
# El separador es punto y coma (;) según inspección previa del archivo.
# =============================================================================
print("=" * 65)
print("LECCIÓN 1 — EDA / IDA: ANÁLISIS INICIAL DEL DATASET")
print("=" * 65)

archivo_datos = RUTA_RAW / "E-ShopNow (Proyecto ABP M2).csv"
datos_raw = pd.read_csv(archivo_datos, sep=";")

print(f"\n✅ Dataset cargado correctamente.")
print(f"   Ruta: {archivo_datos}")
print(f"   Filas: {datos_raw.shape[0]} | Columnas: {datos_raw.shape[1]}")

# =============================================================================
# SECCIÓN 2: CLASIFICACIÓN DE VARIABLES
# Se distinguen variables cuantitativas (numéricas) y categóricas.
# Esta clasificación es la base de todo el análisis posterior.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 2: CLASIFICACIÓN DE VARIABLES")
print("=" * 65)

variables_cuantitativas = [
    "edad_cliente",       # Numérica continua — edad en años
    "precio_unitario",    # Numérica continua — precio en CLP
    "cantidad",           # Numérica discreta — unidades compradas
    "descuento_pct",      # Numérica continua — porcentaje de descuento (0 a 1)
    "antiguedad_vendedor" # Numérica discreta — años de antigüedad
]

variables_categoricas = [
    "id_venta",           # Identificador único (no analizable estadísticamente)
    "fecha_venta",        # Temporal — requiere conversión a datetime
    "id_empleado",        # Identificador de empleado (numérico pero categórico)
    "id_cliente",         # Identificador de cliente (numérico pero categórico)
    "nombre_cliente",     # Texto libre — no analizable directamente
    "email_cliente",      # Texto libre — no analizable directamente
    "genero_cliente",     # Categórica nominal binaria: Hombre / Mujer
    "ciudad_cliente",     # Categórica nominal — ciudad de residencia
    "region_cliente",     # Categórica nominal — región (con inconsistencias)
    "canal_venta",        # Categórica nominal: Web / App / Tienda
    "producto",           # Categórica nominal — nombre del producto
    "categoria_producto"  # Categórica nominal — categoría del producto
]

print("\n📊 VARIABLES CUANTITATIVAS (numéricas):")
for var in variables_cuantitativas:
    tipo = datos_raw[var].dtype
    print(f"   → {var:<25} | dtype: {str(tipo)}")

print("\n🏷️  VARIABLES CATEGÓRICAS:")
for var in variables_categoricas:
    tipo = datos_raw[var].dtype
    n_unicos = datos_raw[var].nunique()
    print(f"   → {var:<25} | dtype: {str(tipo):<8} | valores únicos: {n_unicos}")

# =============================================================================
# SECCIÓN 3: DETECCIÓN DE VALORES FALTANTES (NULOS)
# Los nulos se documentan con conteo absoluto y porcentaje sobre el total,
# porque el impacto depende de cuánto representan sobre el dataset completo.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 3: VALORES FALTANTES (NULOS)")
print("=" * 65)

total_filas = len(datos_raw)
nulos = datos_raw.isnull().sum()
nulos_pct = (nulos / total_filas * 100).round(2)

resumen_nulos = pd.DataFrame({
    "nulos_absolutos": nulos,
    "porcentaje_%": nulos_pct
}).query("nulos_absolutos > 0").sort_values("nulos_absolutos", ascending=False)

if resumen_nulos.empty:
    print("\n✅ No se detectaron valores nulos.")
else:
    print(f"\n⚠️  Se detectaron nulos en {len(resumen_nulos)} columna(s):\n")
    print(resumen_nulos.to_string())

# Comentario analítico:
# - region_cliente (38.7%) y categoria_producto (38.1%) tienen una proporción
#   muy alta de nulos. Esto requiere investigación: ¿son registros de venta
#   física donde no se capturó la región? ¿Error de carga?
# - descuento_pct (23.7%) probablemente representa ventas sin descuento
#   (nulo = 0%). Se imputará con 0 en la lección de limpieza.
# - edad_cliente (1.9%) es manejable con imputación por mediana.

# =============================================================================
# SECCIÓN 4: DETECCIÓN DE FILAS DUPLICADAS
# Los duplicados exactos distorsionan todas las métricas estadísticas
# posteriores, por eso se identifican en la etapa IDA.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 4: FILAS DUPLICADAS")
print("=" * 65)

n_duplicados = datos_raw.duplicated().sum()
print(f"\n⚠️  Filas duplicadas exactas encontradas: {n_duplicados}")

if n_duplicados > 0:
    print(f"   Representan el {n_duplicados / total_filas * 100:.2f}% del dataset.")
    print("\n   Ejemplo de filas duplicadas:")
    print(datos_raw[datos_raw.duplicated(keep=False)].head(4).to_string())

# =============================================================================
# SECCIÓN 5: DETECCIÓN DE INCONSISTENCIAS
# Se revisan valores inesperados en variables categóricas clave.
# Una inconsistencia no es un nulo: es un dato presente pero incorrecto
# o que representa la misma categoría con nombres distintos.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 5: INCONSISTENCIAS EN VARIABLES CATEGÓRICAS")
print("=" * 65)

# --- region_cliente ---
print("\n📍 region_cliente — valores únicos detectados:")
conteo_regiones = datos_raw["region_cliente"].value_counts(dropna=False)
print(conteo_regiones.to_string())
print("\n   ⚠️  INCONSISTENCIA: La Región Metropolitana aparece con 3 nombres:")
print("      'RM', 'Metropolitana', 'Región Metropolitana' → deben unificarse.")

# --- genero_cliente ---
print("\n👤 genero_cliente — valores únicos detectados:")
print(datos_raw["genero_cliente"].value_counts(dropna=False).to_string())

# --- canal_venta ---
print("\n🛒 canal_venta — valores únicos detectados:")
print(datos_raw["canal_venta"].value_counts(dropna=False).to_string())

# --- categoria_producto ---
print("\n📦 categoria_producto — valores únicos detectados:")
print(datos_raw["categoria_producto"].value_counts(dropna=False).to_string())

# --- fecha_venta como string ---
print("\n📅 fecha_venta — tipo de dato actual:")
print(f"   dtype: {datos_raw['fecha_venta'].dtype}")
print(f"   Ejemplo de valores: {datos_raw['fecha_venta'].head(3).tolist()}")
print("   ⚠️  INCONSISTENCIA: fecha_venta es string. Debe convertirse a datetime.")

# =============================================================================
# SECCIÓN 6: RESUMEN GENERAL DE HALLAZGOS IDA
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 6: RESUMEN GENERAL DE HALLAZGOS — IDA")
print("=" * 65)

print(f"""
📋 DATASET: E-ShopNow — Comportamiento de clientes e-commerce chileno
   • Filas totales          : {total_filas}
   • Columnas totales       : {datos_raw.shape[1]}
   • Variables cuantitativas: {len(variables_cuantitativas)}
   • Variables categóricas  : {len(variables_categoricas)}

⚠️  PROBLEMAS DETECTADOS:
   1. Nulos críticos en region_cliente ({nulos['region_cliente']} filas, 38.7%)
   2. Nulos críticos en categoria_producto ({nulos['categoria_producto']} filas, 38.1%)
   3. Nulos en descuento_pct ({nulos['descuento_pct']} filas, 23.7%) → probablemente = 0
   4. Nulos en edad_cliente ({nulos['edad_cliente']} filas, 1.9%)
   5. {n_duplicados} filas duplicadas exactas (1.96%)
   6. region_cliente tiene 3 variantes para la Región Metropolitana
   7. fecha_venta está almacenada como string (no datetime)

🔜 PRÓXIMOS PASOS (Lección 2):
   • Limpiar e imputar nulos según criterio por variable
   • Eliminar duplicados
   • Unificar categorías inconsistentes en region_cliente
   • Convertir fecha_venta a datetime
   • Calcular estadísticas descriptivas sobre variables numéricas
""")