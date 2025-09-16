-- ver columnas
DESCRIBE bronze.ads_spend;

-- número de registros
SELECT COUNT(*) FROM bronze.ads_spend;

-- mostrar de los primeros registros
SELECT * FROM bronze.ads_spend LIMIT 10;

-- rango de fechas disponibles
SELECT MIN(date) AS start_date, MAX(date) AS end_date FROM bronze.ads_spend;

-- distribución de gasto por plataforma
SELECT platform, COUNT(*) AS n_rows, SUM(spend) AS total_spend
FROM bronze.ads_spend
GROUP BY platform
ORDER BY total_spend DESC;

-- distibución de gasto por país
SELECT country, SUM(spend) AS total_spend, SUM(conversions) AS total_conversions
FROM bronze.ads_spend
GROUP BY country
ORDER BY total_spend DESC;

-- gasto y conversiones por campaña
SELECT campaign, SUM(spend) AS total_spend, SUM(conversions) AS total_conversions,
    ROUND(SUM(spend) / NULLIF(SUM(conversions), 0), 2) AS cost_per_conversion
FROM bronze.ads_spend
GROUP BY campaign
ORDER BY total_spend DESC;

