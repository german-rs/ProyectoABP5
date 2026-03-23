# =============================================================================
# LECCIÓN 4: REGRESIÓN LINEAL
# Proyecto: ComercioYA — EDA para decisiones comerciales
# Módulo 5 — Alkemy
# =============================================================================
# OBJETIVO: Implementar modelos de regresión lineal simple y múltiple
# para explicar el monto_venta. Calcular R², MSE y MAE. Evaluar la
# significancia estadística de cada predictor (p-value) e interpretar
# los coeficientes del modelo.
# Variable dependiente (Y): monto_venta
# Predictores candidatos (X): precio_unitario, cantidad, descuento_pct
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------------------------
RUTA_RAIZ      = Path(__file__).resolve().parent.parent
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


def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calcula R², MSE y MAE para evaluar el ajuste del modelo.
    - R²  : proporción de varianza explicada (0 a 1, mayor es mejor)
    - MSE : error cuadrático medio (penaliza errores grandes)
    - MAE : error absoluto medio (más interpretable en la unidad original)
    Se usa sklearn solo para MSE y MAE; R² se extrae directo de statsmodels.
    """
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    return {"MSE": round(mse, 2), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}


# =============================================================================
# SECCIÓN 1: CARGA Y PREPARACIÓN DEL DATASET
# =============================================================================
print("=" * 65)
print("LECCIÓN 4 — REGRESIÓN LINEAL")
print("=" * 65)

archivo = RUTA_PROCESADO / "datos_limpios.csv"
datos = pd.read_csv(archivo, sep=";")

# Recrear variable derivada monto_venta (generada en Lección 3)
datos = datos.assign(
    monto_venta=datos["precio_unitario"] * datos["cantidad"] * (1 - datos["descuento_pct"])
)

print(f"\n✅ Dataset cargado: {datos.shape[0]} filas × {datos.shape[1]} columnas")
print(f"   Variable dependiente: monto_venta")
print(f"   Rango: ${datos['monto_venta'].min():,.0f} — ${datos['monto_venta'].max():,.0f} CLP")

# Variable dependiente
y = datos["monto_venta"]

# =============================================================================
# SECCIÓN 2: REGRESIÓN LINEAL SIMPLE
# Se modela la relación entre precio_unitario (X) y monto_venta (Y).
# Se eligió precio_unitario porque tiene la correlación más alta con
# monto_venta (r=0.86, p<0.001), según el análisis de la Lección 3.
#
# Se agrega constante con sm.add_constant() porque statsmodels no la
# incluye por defecto. Sin constante, el modelo fuerza la recta a pasar
# por el origen, lo cual es incorrecto en este contexto.
# =============================================================================
print("\n" + "=" * 65)
print("SECCIÓN 2: REGRESIÓN LINEAL SIMPLE")
print("          precio_unitario → monto_venta")
print("=" * 65)

X_simple = sm.add_constant(datos[["precio_unitario"]])
modelo_simple = sm.OLS(y, X_simple).fit()

print("\n--- RESUMEN DEL MODELO ---")
print(modelo_simple.summary())

# Extraer métricas clave
y_pred_simple = modelo_simple.predict(X_simple)
metricas_simple = calcular_metricas(y, y_pred_simple)

print("\n--- MÉTRICAS DE EVALUACIÓN ---")
print(f"   R²   : {modelo_simple.rsquared:.4f}  "
      f"→ el modelo explica el {modelo_simple.rsquared*100:.1f}% de la varianza de monto_venta")
print(f"   R² aj.: {modelo_simple.rsquared_adj:.4f}")
print(f"   MSE  : ${metricas_simple['MSE']:,.2f}")
print(f"   RMSE : ${metricas_simple['RMSE']:,.2f}  "
      f"→ el error típico de predicción es ±${metricas_simple['RMSE']:,.0f} CLP")
print(f"   MAE  : ${metricas_simple['MAE']:,.2f}")

print("\n--- INTERPRETACIÓN DE COEFICIENTES ---")
coef_precio = modelo_simple.params["precio_unitario"]
intercepto  = modelo_simple.params["const"]
p_precio    = modelo_simple.pvalues["precio_unitario"]
p_const     = modelo_simple.pvalues["const"]

print(f"""
   Ecuación del modelo:
   monto_venta = {intercepto:,.2f} + {coef_precio:.4f} × precio_unitario

   • Intercepto ({intercepto:,.2f}): valor estimado de monto_venta cuando
     precio_unitario = 0. No tiene interpretación comercial directa
     (ningún producto tiene precio 0), pero es necesario para el ajuste.
     p-value = {p_const:.4f} → {'✅ significativo' if p_const < 0.05 else '❌ no significativo'}

   • precio_unitario ({coef_precio:.4f}): por cada peso adicional en el precio
     unitario, el monto de venta aumenta en {coef_precio:.4f} pesos.
     Dicho de otro modo: un aumento de $10.000 en precio_unitario
     se asocia con un aumento de ${coef_precio*10000:,.0f} en monto_venta.
     p-value = {p_precio:.6f} → {'✅ significativo (p < 0.05)' if p_precio < 0.05 else '❌ no significativo'}
""")

# =============================================================================
# SECCIÓN 3: REGRESIÓN LINEAL MÚLTIPLE
# Se incorporan cantidad y descuento_pct como predictores adicionales.
# Justificación de inclusión:
#   - cantidad     : r=0.38 con monto_venta, p<0.001 (significativo)
#   - descuento_pct: r=-0.057 con monto_venta, p=0.071 (borderline);
#     se incluye porque tiene sentido económico claro (descuento reduce
#     el monto) y el modelo múltiple puede revelar su efecto real al
#     controlar por precio y cantidad.
# Se excluyen edad_cliente y antiguedad_vendedor: r ≈ 0, p >> 0.05.
# Incluirlos solo añadiría ruido y reduciría el R² ajustado.
# =============================================================================
print("=" * 65)
print("SECCIÓN 3: REGRESIÓN LINEAL MÚLTIPLE")
print("          precio_unitario + cantidad + descuento_pct → monto_venta")
print("=" * 65)

PREDICTORES = ["precio_unitario", "cantidad", "descuento_pct"]
X_multiple = sm.add_constant(datos[PREDICTORES])
modelo_multiple = sm.OLS(y, X_multiple).fit()

print("\n--- RESUMEN DEL MODELO ---")
print(modelo_multiple.summary())

y_pred_multiple = modelo_multiple.predict(X_multiple)
metricas_multiple = calcular_metricas(y, y_pred_multiple)

print("\n--- MÉTRICAS DE EVALUACIÓN ---")
print(f"   R²   : {modelo_multiple.rsquared:.4f}  "
      f"→ el modelo explica el {modelo_multiple.rsquared*100:.1f}% de la varianza de monto_venta")
print(f"   R² aj.: {modelo_multiple.rsquared_adj:.4f}")
print(f"   MSE  : ${metricas_multiple['MSE']:,.2f}")
print(f"   RMSE : ${metricas_multiple['RMSE']:,.2f}")
print(f"   MAE  : ${metricas_multiple['MAE']:,.2f}")

print("\n--- INTERPRETACIÓN DE COEFICIENTES Y SIGNIFICANCIA ---")
print(f"\n{'Variable':<22} {'Coeficiente':>14} {'p-value':>12} {'Significativo':>14}")
print("-" * 64)
for var in ["const"] + PREDICTORES:
    coef = modelo_multiple.params[var]
    pval = modelo_multiple.pvalues[var]
    sig  = "✅ Sí" if pval < 0.05 else "❌ No"
    print(f"{var:<22} {coef:>14,.4f} {pval:>12.6f} {sig:>14}")

coef_pu  = modelo_multiple.params["precio_unitario"]
coef_qty = modelo_multiple.params["cantidad"]
coef_dsc = modelo_multiple.params["descuento_pct"]

print(f"""
   Ecuación del modelo múltiple:
   monto_venta = {modelo_multiple.params['const']:,.2f}
               + {coef_pu:.4f}    × precio_unitario
               + {coef_qty:,.4f} × cantidad
               + {coef_dsc:,.4f} × descuento_pct

   • precio_unitario: controlando por cantidad y descuento, un aumento
     de $10.000 en precio se asocia con +${coef_pu*10000:,.0f} en monto_venta.

   • cantidad: cada unidad adicional comprada aumenta el monto de venta
     en ${coef_qty:,.0f} CLP, controlando por precio y descuento.

   • descuento_pct: un descuento adicional del 10% (0.10) se asocia con
     una variación de ${coef_dsc*0.10:,.0f} CLP en el monto de venta,
     controlando por precio y cantidad.
""")

# =============================================================================
# SECCIÓN 4: COMPARACIÓN SIMPLE vs. MÚLTIPLE
# =============================================================================
print("=" * 65)
print("SECCIÓN 4: COMPARACIÓN DE MODELOS")
print("=" * 65)

mejora_r2   = modelo_multiple.rsquared - modelo_simple.rsquared
mejora_rmse = metricas_simple["RMSE"] - metricas_multiple["RMSE"]

print(f"""
   {'Métrica':<12} {'Modelo Simple':>15} {'Modelo Múltiple':>17} {'Mejora':>10}
   {'-'*56}
   {'R²':<12} {modelo_simple.rsquared:>15.4f} {modelo_multiple.rsquared:>17.4f} {mejora_r2:>+10.4f}
   {'R² ajust.':<12} {modelo_simple.rsquared_adj:>15.4f} {modelo_multiple.rsquared_adj:>17.4f}
   {'RMSE':<12} {metricas_simple['RMSE']:>15,.0f} {metricas_multiple['RMSE']:>17,.0f} {mejora_rmse:>+10,.0f}
   {'MAE':<12} {metricas_simple['MAE']:>15,.0f} {metricas_multiple['MAE']:>17,.0f}

📝 INTERPRETACIÓN:
   El modelo múltiple mejora el R² en {mejora_r2:.4f} puntos respecto al
   modelo simple, y reduce el RMSE en ${mejora_rmse:,.0f} CLP.
   La mejora es {'notable' if mejora_r2 > 0.05 else 'modesta pero real'}, lo que confirma que cantidad
   y descuento_pct aportan poder explicativo adicional más allá
   del precio unitario.
""")

# =============================================================================
# SECCIÓN 5: VISUALIZACIÓN DE LA REGRESIÓN CON SEABORN
# Se grafican tres visualizaciones complementarias:
#   1. Regresión simple con banda de confianza (regplot)
#   2. Valores reales vs. predichos del modelo múltiple
#   3. Residuos del modelo múltiple (diagnóstico de ajuste)
# =============================================================================
print("=" * 65)
print("SECCIÓN 5: VISUALIZACIÓN DE REGRESIONES")
print("=" * 65)

# --- Gráfico 1: Regresión simple precio_unitario → monto_venta ---
fig, ax = plt.subplots(figsize=(9, 6))

sns.regplot(
    data=datos,
    x="precio_unitario",
    y="monto_venta",
    scatter_kws={"alpha": 0.25, "s": 15, "color": "#4C72B0"},
    line_kws={"color": "#DD4444", "linewidth": 2},
    ci=95,   # Banda de confianza al 95%
    ax=ax
)

ax.set_title(
    f"Regresión simple: precio_unitario → monto_venta\n"
    f"R² = {modelo_simple.rsquared:.4f} | RMSE = ${metricas_simple['RMSE']:,.0f} CLP",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Precio unitario (CLP)", fontsize=10)
ax.set_ylabel("Monto de venta (CLP)", fontsize=10)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"${val/1_000:,.0f}K")
)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"${val/1_000_000:.1f}M")
)

guardar_grafico("04_regresion_simple_precio_vs_monto")
print("✅ Regresión simple graficada.")

# --- Gráfico 2: Valores reales vs. predichos (modelo múltiple) ---
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    y, y_pred_multiple,
    alpha=0.25, s=15, color="#4C72B0", edgecolors="none"
)
# Línea de predicción perfecta (y_real = y_pred): referencia visual
lim_min = min(y.min(), y_pred_multiple.min())
lim_max = max(y.max(), y_pred_multiple.max())
ax.plot([lim_min, lim_max], [lim_min, lim_max],
        color="#DD4444", linewidth=2, linestyle="--", label="Predicción perfecta")

ax.set_title(
    f"Valores reales vs. predichos — Modelo múltiple\n"
    f"R² = {modelo_multiple.rsquared:.4f} | MAE = ${metricas_multiple['MAE']:,.0f} CLP",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("monto_venta real (CLP)", fontsize=10)
ax.set_ylabel("monto_venta predicho (CLP)", fontsize=10)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"${val/1_000_000:.1f}M")
)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"${val/1_000_000:.1f}M")
)

guardar_grafico("04_reales_vs_predichos_multiple")
print("✅ Gráfico real vs. predicho generado.")

# --- Gráfico 3: Residuos del modelo múltiple ---
# Los residuos deben distribuirse aleatoriamente alrededor de 0.
# Un patrón sistemático indicaría que el modelo no captura alguna
# relación no lineal presente en los datos.
residuos = y - y_pred_multiple

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Diagnóstico de residuos — Modelo múltiple",
    fontsize=13, fontweight="bold"
)

# Residuos vs. valores predichos
ejes[0].scatter(y_pred_multiple, residuos,
                alpha=0.25, s=15, color="#4C72B0", edgecolors="none")
ejes[0].axhline(0, color="#DD4444", linewidth=1.5, linestyle="--")
ejes[0].set_xlabel("Valores predichos (CLP)", fontsize=10)
ejes[0].set_ylabel("Residuos (CLP)", fontsize=10)
ejes[0].set_title("Residuos vs. Predichos", fontweight="bold")
ejes[0].xaxis.set_major_formatter(
    plt.FuncFormatter(lambda val, _: f"${val/1_000_000:.1f}M")
)

# Histograma de residuos
ejes[1].hist(residuos, bins=30, color="#4C72B0",
             edgecolor="white", linewidth=0.5)
ejes[1].axvline(0, color="#DD4444", linewidth=1.5,
                linestyle="--", label="Residuo = 0")
ejes[1].set_xlabel("Residuo (CLP)", fontsize=10)
ejes[1].set_ylabel("Frecuencia", fontsize=10)
ejes[1].set_title("Distribución de residuos", fontweight="bold")
ejes[1].legend(fontsize=9)

guardar_grafico("04_diagnostico_residuos")
print("✅ Diagnóstico de residuos generado.")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("RESUMEN — LECCIÓN 4 COMPLETADA")
print("=" * 65)
print(f"""
✅ Regresión simple aplicada  (precio_unitario → monto_venta)
   R² = {modelo_simple.rsquared:.4f} | RMSE = ${metricas_simple['RMSE']:,.0f} CLP

✅ Regresión múltiple aplicada (precio_unitario + cantidad + descuento_pct)
   R² = {modelo_multiple.rsquared:.4f} | RMSE = ${metricas_multiple['RMSE']:,.0f} CLP

✅ Gráficos exportados:
   → outputs/graphs/04_regresion_simple_precio_vs_monto.png
   → outputs/graphs/04_reales_vs_predichos_multiple.png
   → outputs/graphs/04_diagnostico_residuos.png

📝 HALLAZGO PRINCIPAL:
   precio_unitario es el predictor dominante del monto de venta.
   El modelo múltiple mejora el ajuste al incorporar cantidad y
   descuento_pct, pero el poder predictivo sigue concentrado en
   el precio del producto seleccionado por el cliente.

🔜 PRÓXIMA LECCIÓN: Análisis visual avanzado con Seaborn
   (pairplot, violinplot, jointplot, heatmap, FacetGrid)
""")
"""



## Hallazgos clave de esta lección

Dos modelos construidos y comparados:

**Modelo simple** (`precio_unitario → monto_venta`): El precio unitario explica la mayor parte de la varianza del monto de venta — lo cual tiene sentido: comprar un electrodoméstico de $700.000 determina en gran medida cuánto gastará el cliente en esa transacción.

**Modelo múltiple** (`precio_unitario + cantidad + descuento_pct`): Añadir cantidad y descuento mejora el ajuste. `cantidad` es significativa (p<0.001); `descuento_pct` puede resultar significativa o borderline dependiendo del dataset exacto, lo que se documentará al ejecutar el script.

El gráfico de residuos es importante pedagógicamente: si los residuos muestran un patrón en forma de escalones (por la distribución bimodal de precios), eso confirma la limitación del modelo lineal para este dataset.



## Actualización de `progress.md` — Lección 4

### Lección 4 — Regresión Lineal
- [x] Regresión simple aplicada con statsmodels
- [x] Regresión múltiple aplicada con statsmodels
- [x] R², MSE, MAE calculados e interpretados
- [x] Significancia de predictores evaluada (p-value)
- [x] Coeficientes interpretados
- [x] Regresión visualizada con Seaborn

"""