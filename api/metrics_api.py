from datetime import date, timedelta
from typing import Optional, Dict, Any
from dateutil.parser import isoparse
from fastapi import FastAPI, HTTPException, Query
import duckdb
import os

DB_PATH = os.path.expanduser("workspace/warehouse.duckdb")

app = FastAPI(title="Metrics KPIs API")

def _get_max_date(con) -> date:
    maxdate = con.execute("SELECT MAX(CAST(date AS DATE)) FROM bronze.ads_spend").fetchone()[0]
    if maxdate is None:
        raise HTTPException(status_code=404, detail="No data available")
    return maxdate

def _safe_div(n: float, d: float) -> Optional[float]:
    return None if d in (0, None) else (n/d)

def _compute_kpis(cur_spend: float, cur_conv: float, prev_spend: float, prev_conv: float) -> Dict[str, Any]:
    # CAC = spend / conversions
    # CAC (Customer Acquisition Cost)
    cac_cur = _safe_div(cur_spend, cur_conv)
    cac_prev = _safe_div(prev_spend, prev_conv)

    # ROAS = (revenue - spend) / spend 
    # ROAS (Return on Ad Spend)
    roas_cur = _safe_div(cur_conv * 100, cur_spend)
    roas_prev = _safe_div(prev_conv * 100, prev_spend)

    def deltas(now: Optional[float], prev: Optional[float]) -> Dict[str, Optional[float]]:
        if now is None or prev is None:
            return {"current": now, "prior": prev, "delta_abs": None, "delta_pct": None}
        abs_delta = round(now - prev, 4)
        pct_delta = None if prev == 0 else round((now - prev) / prev * 100, 2)
        return {"current": round(now, 4), "prior": round(prev, 4), "delta_abs": abs_delta, "delta_pct": pct_delta}
    
    return {
        "cac": deltas(cac_cur, cac_prev),
        "roas": deltas(roas_cur, roas_prev)
    }

@app.get("/metrics")
def metrics(
        start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
        end: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """
        Returns CAC, ROAS of the range [start, end] compared to the previous period of the same length immediately before it.
        If start/end not send, use the last 30 days based on MAX(date).
    """

    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)

    if start is None or end is None:
        maxdate = _get_max_date(con)
        end_date = maxdate
        start_date = maxdate - timedelta(days=29)
    else: 
        try:
            start_date = isoparse(start).date()
            end_date = isoparse(end).date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start must be before or equal to end")
    length_days = (end_date - start_date).days + 1
    prior_end = start_date - timedelta(days=1)
    prior_start = prior_end - timedelta(days=length_days - 1)

    q = """
        SELECT 'current' AS period, SUM(spend) AS spend, SUM(conversions) AS conversions
        FROM bronze.ads_spend
        WHERE CAST(date AS DATE) BETWEEN ? AND ?
        UNION ALL
        SELECT 'prior' AS period, SUM(spend) AS spend, SUM(conversions) AS conversions
        FROM bronze.ads_spend
        WHERE CAST(date AS DATE) BETWEEN ? AND ?
        """
    
    rows = con.execute(q, [start_date, end_date, prior_start, prior_end]).fetchall()
    con.close()

    cur_spend = cur_conv = prv_spend = prv_conv = 0.0
    for period, spend, conv in rows:
        spend = float(spend or 0.0)
        conv = float(conv or 0.0)
        if period == 'current':
            cur_spend, cur_conv = spend, conv
        else:
            prv_spend, prv_conv = spend, conv
    kpis = _compute_kpis(cur_spend, cur_conv, prv_spend, prv_conv)

    return {
        "period": {
            "current": {"start": str(start_date), "end": str(end_date), "days": length_days},
            "prior": {"start": str(prior_start), "end": str(prior_end), "days": length_days},
        },
        "totals": {
            "current": {"spend": cur_spend, "conversions": cur_conv},
            "prior": {"spend": prv_spend, "conversions": prv_conv},
        },
        "kpis": kpis
    }