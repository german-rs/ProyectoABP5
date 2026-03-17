# productContext.md — Contexto del negocio

## La empresa: ComercioYA
E-commerce que recolectó datos históricos de comportamiento de clientes.
Necesita pasar de tener datos a **tomar decisiones basadas en evidencia**.

## Variables esperadas del dataset

| Variable           | Tipo        | Descripción                              |
|--------------------|-------------|------------------------------------------|
| `id_cliente`       | Categórica  | Identificador único del cliente          |
| `region`           | Categórica  | Zona geográfica del cliente              |
| `genero`           | Categórica  | Género del cliente                       |
| `edad`             | Numérica    | Edad en años                             |
| `visitas`          | Numérica    | Cantidad de visitas al sitio             |
| `compras`          | Numérica    | Número de compras realizadas             |
| `monto_total`      | Numérica    | Monto total gastado (CLP o USD)          |
| `devoluciones`     | Numérica    | Cantidad de devoluciones                 |
| `calificacion`     | Numérica    | Reseña/puntaje (1–5)                     |
| `categoria`        | Categórica  | Categoría de producto más comprada       |
| `cliente_activo`   | Categórica  | Si el cliente compró en el último mes    |

## Preguntas de negocio a responder con el EDA
1. ¿Qué perfil tienen los clientes que más gastan?
2. ¿Existe relación entre visitas y monto de compra?
3. ¿Las devoluciones afectan la calificación del cliente?
4. ¿Qué región genera mayor volumen de ventas?
5. ¿Qué variables predicen mejor el monto total gastado?

## Decisiones estratégicas que se esperan sustentar
- Segmentación de clientes para campañas de marketing
- Identificación de clientes en riesgo de abandono (churn)
- Optimización de categorías de productos por región
