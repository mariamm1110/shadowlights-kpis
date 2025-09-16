---
-- Calcularemos CAC y ROAS comparando
-- CAC: Costo de Adquisición de Clientes
-- ROAS: Retorno de la Inversión Publicitaria
-- últimos 30 días vs 30 días previos

-- PRIMERO: definir la fecha de corte
CREATE OR REPLACE TEMP VIEW params AS 
-- se busca el último día con datos
SELECT COALESCE(MAX(CAST(date AS DATE)), CURRENT_DATE) AS max_date 
FROM bronze.ads_spend;

-- ver
SELECT * FROM params;


-- PASO 2: definir ventas de tiempo
-- last_30 = últimos 30 días
-- prior_30 = 30 días anteriores 

CREATE OR REPLACE TEMP VIEW ranges AS
SELECT 'last_30' AS period,
        max_date - INTERVAL 29 DAY AS start_date,
        max_date AS end_date
FROM params
UNION ALL
SELECT 'prior_30' AS period,
-- 30 días inmediatamente anteriores a los últimos 30 días
        max_date - INTERVAL 59 DAY AS start_date,
        max_date - INTERVAL 30 DAY AS end_date
FROM params;

-- ver
SELECT * FROM ranges ORDER BY period DESC;



-- PASO 3: agregar métricas base por período
-- suma de spend y conversions
-- convertimos los 2000 registros diarios en 2 registros por período
CREATE OR REPLACE TEMP VIEW agg AS 
SELECT r.period,
        SUM(s.spend) AS spend,
        SUM(s.conversions) AS conversions
FROM ranges r
-- por cada fila en ranges, buscamos filas en ads_spend que tengan la fecha en ese rango 
JOIN bronze.ads_spend s
    ON CAST(s.date AS DATE) BETWEEN r.start_date AND r.end_date
-- agrupamos por la primera columna, que es period
GROUP BY 1;

-- con los SUM, agregamos el gasto toal y conversiones totales de cada rango

-- ver
SELECT * FROM agg ORDER BY period;


-- PASO 4: Calculams CAC = spend / conversions
CREATE OR REPLACE TEMP VIEW metrics_cac AS
SELECT 
    (a_last.spend / NULLIF(a_last.conversions, 0)) AS last_30,
    (a_prev.spend / NULLIF(a_prev.conversions, 0)) AS prior_30,    
FROM agg a_last
JOIN agg a_prev
    ON a_last.period = 'last_30' AND a_prev.period = 'prior_30';

-- ver
SELECT * FROM metrics_cac;


-- PASO 5: Calcular ROAS = revenue / spend (revenue = conversions * 100)

CREATE OR REPLACE TEMP VIEW metrics_roas AS
SELECT
    ((a_last.conversions * 100) / NULLIF(a_last.spend, 0)) AS last_30,
    ((a_prev.conversions * 100) / NULLIF(a_prev.spend, 0)) AS prior_30
FROM agg a_last
JOIN agg a_prev
    ON a_last.period = 'last_30' AND a_prev.period = 'prior_30';

-- ver
SELECT * FROM metrics_roas;


-- PASO 6: tabla compacta de resultados

WITH m AS (
    SELECT 'CAC' AS metric, last_30, prior_30 FROM metrics_cac
    UNION ALL
    SELECT 'ROAS' AS metric, last_30, prior_30 FROM metrics_roas
)
SELECT metric, ROUND(last_30, 4) AS last_30,
       ROUND(prior_30, 4) AS prior_30,
       ROUND(last_30 - prior_30, 4) AS delta_abs,
       ROUND(((last_30 - prior_30) / NULLIF(prior_30, 0))*100, 2) AS delta_pct
FROM m
ORDER BY metric;
