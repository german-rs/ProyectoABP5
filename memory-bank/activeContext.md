# activeContext.md — Contexto activo (actualizar en cada sesión)

> ⚠️ **Este archivo debe actualizarse al inicio y al final de cada sesión de trabajo.**

## Lección en curso
**Lección 2 — CONCEPTOS BÁSICOS DE ESTADÍSTICA DESCRIPTIVA**

## Tarea actual
- [X] Calcular media, mediana, moda, varianza y desviación estándar
- [X] Determinar cuartiles y percentiles
- [X] Generar histogramas y boxplots
- [X] Identificar valores atípicos y documentar su impacto

## Decisiones tomadas recientemente
- Se usó el dataset real `E-ShopNow (Proyecto ABP M2).csv` ubicado en `data/raw/` (no se generó dataset sintético)
- Se usará IQR en lugar de Z-score para detección de outliers, porque no se puede asumir distribución normal en variables como `precio_unitario`
- `descuento_pct` nulo se interpretará como 0 (sin descuento aplicado) y se imputará con 0 en la limpieza
- `edad_cliente` nula se imputará con la mediana (distribución no garantizada como normal)
- Los duplicados (20 filas, 1.96%) serán eliminados en el paso de limpieza
- `region_cliente` requiere unificación: 'RM', 'Metropolitana' y 'Región Metropolitana' representan la misma categoría
- `fecha_venta` debe convertirse de string a datetime antes del análisis temporal

## Bloqueos o dudas pendientes
- ¿Cómo tratar los nulos críticos de `region_cliente` (38.7%) y `categoria_producto` (38.1%)? ¿Eliminar filas o imputar con una categoría 'Sin información'?
- ¿El campo `genero_cliente` con valores 'Hombre'/'Mujer' es correcto o hay registros mal asignados (ej: nombre femenino con género 'Hombre')?

## Últimos archivos modificados
- `src/01_ida.py`

## Próxima sesión
- Iniciar Lección 2: calcular media, mediana, moda, varianza, percentiles
- Generar histogramas y boxplots de las variables numéricas clave
- Aplicar limpieza básica del dataset antes de calcular estadísticas
- Guardar dataset limpio en `data/processed/datos_limpios.csv`