CREATE SCHEMA IF NOT EXISTS mart;

CREATE OR REPLACE VIEW mart.kpi_ads_30d AS
WITH params AS (
  SELECT COALESCE(MAX(date), CURRENT_DATE) AS max_date
  FROM bronze.ads_spend
),
ranges AS (
  SELECT 'last_30'  AS period,
         max_date - INTERVAL 29 DAY AS start_date,
         max_date                  AS end_date
  FROM params
  UNION ALL
  SELECT 'prior_30' AS period,
         max_date - INTERVAL 59 DAY AS start_date,
         max_date - INTERVAL 30 DAY AS end_date
  FROM params
),
agg AS (
  SELECT r.period,
         SUM(s.spend)       AS spend,
         SUM(s.conversions) AS conversions
  FROM ranges r
  JOIN bronze.ads_spend s
    ON s.date BETWEEN r.start_date AND r.end_date
  GROUP BY 1
),
metrics AS (
  SELECT 'CAC'  AS metric,
         (a_last.spend / NULLIF(a_last.conversions, 0))             AS last_30,
         (a_prev.spend / NULLIF(a_prev.conversions, 0))             AS prior_30
  FROM agg a_last
  JOIN agg a_prev
    ON a_last.period = 'last_30'
   AND a_prev.period = 'prior_30'
  UNION ALL
  SELECT 'ROAS' AS metric,
         ((a_last.conversions * 100.0) / NULLIF(a_last.spend, 0))   AS last_30,
         ((a_prev.conversions * 100.0) / NULLIF(a_prev.spend, 0))   AS prior_30
  FROM agg a_last
  JOIN agg a_prev
    ON a_last.period = 'last_30'
   AND a_prev.period = 'prior_30'
)
SELECT metric,
       ROUND(last_30, 4)  AS last_30,
       ROUND(prior_30, 4) AS prior_30,
       ROUND(last_30 - prior_30, 4)                               AS delta_abs,
       ROUND(((last_30 - prior_30) / NULLIF(prior_30,0))*100, 2)  AS delta_pct
FROM metrics
ORDER BY metric;

SELECT * FROM mart.kpi_ads_30d;
