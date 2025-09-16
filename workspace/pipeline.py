
#!/usr/bin/env python3
"""
Pipeline de ingesta y medallion (Bronze -> Silver -> Gold) con DuckDB.

Uso típico:
  python pipeline.py \
    --warehouse /data/warehouse.duckdb \
    --csv /data/ads_spend.csv \
    --source-file ads_spend.csv \
    --mode incremental

Requiere: duckdb (pip install duckdb), pandas (opcional).
"""

import argparse
import sys
import os
import uuid
from datetime import datetime
import duckdb

# ---------------------------
# Utils
# ---------------------------

def info(msg: str):
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)

def err(msg: str):
    print(f"[ERROR] {msg}", flush=True, file=sys.stderr)

# ---------------------------
# SQL helpers
# ---------------------------

DDL_SCHEMAS = r"""
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS audit;
"""

# Tabla "bronze" principal con provenance (similar a tu ads_spend del Colab)
DDL_BRONZE_TABLE = r"""
CREATE TABLE IF NOT EXISTS bronze.ads_spend (
  date DATE,
  platform TEXT,
  account TEXT,
  campaign TEXT,
  country TEXT,
  device TEXT,
  spend DOUBLE,
  clicks INTEGER,
  impressions INTEGER,
  conversions INTEGER,
  load_date TIMESTAMPTZ DEFAULT now(),
  source_file_name TEXT
);
"""

# Tabla de auditoría
DDL_AUDIT = r"""
CREATE TABLE IF NOT EXISTS audit.load_runs (
  load_id TEXT PRIMARY KEY,
  load_ts TIMESTAMPTZ DEFAULT now(),
  source_file_name TEXT,
  rows_staging BIGINT,
  rows_bronze_inserted BIGINT,
  rows_silver BIGINT,
  notes TEXT
);
"""

# Vista/tabla Silver (se recrea sobre bronze, con casts y dedupe por clave de negocio)
# Nota: usamos ROW_NUMBER() para quedarnos con el registro más reciente por load_date
SILVER_REFRESH = r"""
CREATE OR REPLACE TABLE silver.ads_spend_clean AS
WITH typed AS (
  SELECT
    TRY_CAST(date AS DATE) AS date,
    LOWER(TRIM(platform)) AS platform,
    TRIM(account) AS account,
    TRIM(campaign) AS campaign,
    UPPER(TRIM(country)) AS country,
    LOWER(TRIM(device)) AS device,
    TRY_CAST(spend AS DOUBLE) AS spend,
    TRY_CAST(clicks AS INTEGER) AS clicks,
    TRY_CAST(impressions AS INTEGER) AS impressions,
    TRY_CAST(conversions AS INTEGER) AS conversions,
    load_date,
    source_file_name,
    (
      COALESCE(CAST(TRY_CAST(date AS DATE) AS VARCHAR), '') || '||' ||
      COALESCE(LOWER(TRIM(platform)), '') || '||' ||
      COALESCE(TRIM(account), '') || '||' ||
      COALESCE(TRIM(campaign), '') || '||' ||
      COALESCE(UPPER(TRIM(country)), '') || '||' ||
      COALESCE(LOWER(TRIM(device)), '')
    ) AS business_key
  FROM bronze.ads_spend
  WHERE TRY_CAST(date AS DATE) IS NOT NULL
    AND TRY_CAST(spend AS DOUBLE) IS NOT NULL
    AND TRY_CAST(clicks AS INTEGER) IS NOT NULL
    AND TRY_CAST(impressions AS INTEGER) IS NOT NULL
    AND TRY_CAST(conversions AS INTEGER) IS NOT NULL
),
deduped AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY business_key ORDER BY load_date DESC) AS rn
  FROM typed
)
SELECT
  date, platform, account, campaign, country, device,
  spend, clicks, impressions, conversions,
  load_date, source_file_name, business_key
FROM deduped
WHERE rn = 1;
"""

# Gold: vista diaria por plataforma con KPIs (manejo de divisiones por 0)
GOLD_REFRESH = r"""
CREATE OR REPLACE VIEW gold.ads_spend_daily_platform AS
SELECT
  date,
  platform,
  SUM(spend) AS spend,
  SUM(clicks) AS clicks,
  SUM(impressions) AS impressions,
  SUM(conversions) AS conversions,
  CASE WHEN SUM(clicks) > 0 THEN SUM(spend) / SUM(clicks) ELSE NULL END AS cpc,
  CASE WHEN SUM(impressions) > 0 THEN CAST(SUM(clicks) AS DOUBLE) / SUM(impressions) ELSE NULL END AS ctr,
  CASE WHEN SUM(conversions) > 0 THEN SUM(spend) / SUM(conversions) ELSE NULL END AS cpa
FROM silver.ads_spend_clean
GROUP BY date, platform
ORDER BY date DESC, platform;
"""

def create_schemas_and_tables(con):
    con.execute(DDL_SCHEMAS)
    con.execute(DDL_BRONZE_TABLE)
    con.execute(DDL_AUDIT)

def load_to_staging(con, csv_path: str):
    # staging en temp (se recrea cada corrida)
    con.execute("DROP TABLE IF EXISTS staging_ads_spend;")
    con.execute(f"""
        CREATE TABLE staging_ads_spend AS
        SELECT * FROM read_csv_auto('{csv_path}', HEADER=TRUE);
    """)
    rows = con.execute("SELECT COUNT(*) FROM staging_ads_spend;").fetchone()[0]
    return rows

def bronze_append(con, source_file_name: str):
    before = con.execute("SELECT COUNT(*) FROM bronze.ads_spend;").fetchone()[0]
    con.execute("""
        INSERT INTO bronze.ads_spend
        SELECT
          TRY_CAST(s.date AS DATE) AS date,
          s.platform,
          s.account,
          s.campaign,
          s.country,
          s.device,
          TRY_CAST(s.spend AS DOUBLE) AS spend,
          TRY_CAST(s.clicks AS INTEGER) AS clicks,
          TRY_CAST(s.impressions AS INTEGER) AS impressions,
          TRY_CAST(s.conversions AS INTEGER) AS conversions,
          now() AS load_date,
          ? AS source_file_name
        FROM staging_ads_spend s
        WHERE NOT EXISTS (
          SELECT 1
          FROM bronze.ads_spend t
          WHERE t.date = TRY_CAST(s.date AS DATE)
            AND LOWER(t.platform) = LOWER(s.platform)
            AND t.account        = s.account
            AND t.campaign       = s.campaign
            AND UPPER(t.country) = UPPER(s.country)
            AND LOWER(t.device)  = LOWER(s.device)
        );
    """, [source_file_name])
    after = con.execute("SELECT COUNT(*) FROM bronze.ads_spend;").fetchone()[0]
    return after - before


def refresh_silver_and_gold(con):
    con.execute(SILVER_REFRESH)
    con.execute(GOLD_REFRESH)
    rows_silver = con.execute("SELECT COUNT(*) FROM silver.ads_spend_clean;").fetchone()[0]
    return rows_silver

def upsert_audit(con, load_id: str, source_file_name: str, rows_staging: int, rows_bronze_inserted: int, rows_silver: int, notes: str = None):
    con.execute("""
        INSERT INTO audit.load_runs (load_id, source_file_name, rows_staging, rows_bronze_inserted, rows_silver, notes)
        VALUES (?, ?, ?, ?, ?, ?);
    """, [load_id, source_file_name, rows_staging, rows_bronze_inserted, rows_silver, notes])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse", required=True, help="Ruta al archivo .duckdb (p. ej. /data/warehouse.duckdb)")
    parser.add_argument("--csv", required=True, help="Ruta al CSV (p. ej. /data/ads_spend.csv)")
    parser.add_argument("--source-file", default=None, help="Nombre del archivo de origen para provenance")
    parser.add_argument("--mode", choices=["incremental", "full"], default="incremental", help="Modo de ejecución (hoy se usa igual)")
    parser.add_argument("--load-id", default=None, help="UUID de la corrida; si no se especifica se genera uno")
    args = parser.parse_args()

    warehouse = args.warehouse
    csv_path = args.csv
    source_file_name = args.source_file or os.path.basename(csv_path)
    load_id = args.load_id or str(uuid.uuid4())

    if not os.path.exists(os.path.dirname(warehouse)) and os.path.dirname(warehouse) != "":
        os.makedirs(os.path.dirname(warehouse), exist_ok=True)
    if not os.path.exists(csv_path):
        err(f"No se encontró el CSV en: {csv_path}")
        return sys.exit(2)

    info(f"Warehouse: {warehouse}")
    info(f"CSV: {csv_path}")
    info(f"Source file: {source_file_name}")
    info(f"Load ID: {load_id}")

    # Conexión a DuckDB (crea el archivo si no existe)
    con = duckdb.connect(warehouse)

    try:
        create_schemas_and_tables(con)
        rows_staging = load_to_staging(con, csv_path)
        info(f"Filas en staging: {rows_staging}")

        rows_bronze_inserted = bronze_append(con, source_file_name)
        info(f"Filas insertadas en bronze: {rows_bronze_inserted}")

        rows_silver = refresh_silver_and_gold(con)
        info(f"Filas en silver.ads_spend_clean: {rows_silver}")

        upsert_audit(con, load_id, source_file_name, rows_staging, rows_bronze_inserted, rows_silver, notes=None)

        # Resumen final para capturar en n8n
        print("=== RUN SUMMARY ===")
        print(f"load_id={load_id}")
        print(f"source_file_name={source_file_name}")
        print(f"rows_staging={rows_staging}")
        print(f"rows_bronze_inserted={rows_bronze_inserted}")
        print(f"rows_silver={rows_silver}")

        return sys.exit(0)
    except Exception as e:
        err(f"Fallo la corrida: {e}")
        try:
            upsert_audit(con, load_id, source_file_name, 0, 0, 0, notes=str(e))
        except Exception as e2:
            err(f"No se pudo registrar en auditoria: {e2}")
        return sys.exit(1)
    finally:
        con.close()

if __name__ == "__main__":
    main()
