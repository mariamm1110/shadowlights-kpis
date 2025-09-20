from datetime import date, timedelta
from typing import Optional, Dict, Any
from dateutil.parser import isoparse
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import duckdb
import os
import re

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

def _parse_natural_language(question: str) -> Optional[tuple[str, str]]:
    """Parse natural language questions and extract date parameters"""
    
    patterns = {
        r"last (\d+) days": lambda m: _get_last_n_days(int(m.group(1))),
        r"past (\d+) days": lambda m: _get_last_n_days(int(m.group(1))),
        r"from ([\d-]+) to ([\d-]+)": lambda m: (m.group(1), m.group(2)),
        r"between ([\d-]+) and ([\d-]+)": lambda m: (m.group(1), m.group(2)),
        r"this month": lambda m: _get_current_month(),
        r"last month": lambda m: _get_last_month(),
        r"this week": lambda m: _get_current_week(),
        r"last week": lambda m: _get_last_week(),
        r"today": lambda m: _get_today(),
        r"yesterday": lambda m: _get_yesterday(),
        r"last (\d+) weeks?": lambda m: _get_last_n_weeks(int(m.group(1))),
        r"past (\d+) weeks?": lambda m: _get_last_n_weeks(int(m.group(1))),
        r"last (\d+) months?": lambda m: _get_last_n_months(int(m.group(1))),
        r"past (\d+) months?": lambda m: _get_last_n_months(int(m.group(1))),
    }
    
    question_lower = question.lower()
    
    for pattern, handler in patterns.items():
        match = re.search(pattern, question_lower)
        if match:
            return handler(match)
    
    return None, None

def _get_last_n_days(n: int) -> tuple[str, str]:
    """Get date range for last N days"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    end_date = maxdate
    start_date = maxdate - timedelta(days=n-1)
    return str(start_date), str(end_date)

def _get_current_month() -> tuple[str, str]:
    """Get current month date range"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    # Use data max date as "current"
    start_date = maxdate.replace(day=1)
    return str(start_date), str(maxdate)

def _get_last_month() -> tuple[str, str]:
    """Get last month date range"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    # Calculate last month relative to data max date
    first_this_month = maxdate.replace(day=1)
    last_month_end = first_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    return str(last_month_start), str(last_month_end)

def _get_current_week() -> tuple[str, str]:
    """Get current week date range (Monday to Sunday)"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    # Get Monday of the week containing maxdate
    days_since_monday = maxdate.weekday()
    start_date = maxdate - timedelta(days=days_since_monday)
    return str(start_date), str(maxdate)

def _get_last_week() -> tuple[str, str]:
    """Get last week date range (Monday to Sunday)"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    # Get Monday of current week, then go back 7 days
    days_since_monday = maxdate.weekday()
    current_week_monday = maxdate - timedelta(days=days_since_monday)
    last_week_monday = current_week_monday - timedelta(days=7)
    last_week_sunday = last_week_monday + timedelta(days=6)
    return str(last_week_monday), str(last_week_sunday)

def _get_today() -> tuple[str, str]:
    """Get today's date range"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    return str(maxdate), str(maxdate)

def _get_yesterday() -> tuple[str, str]:
    """Get yesterday's date range"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    yesterday = maxdate - timedelta(days=1)
    return str(yesterday), str(yesterday)

def _get_last_n_weeks(n: int) -> tuple[str, str]:
    """Get date range for last N weeks"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    end_date = maxdate
    start_date = maxdate - timedelta(weeks=n) + timedelta(days=1)
    return str(start_date), str(end_date)

def _get_last_n_months(n: int) -> tuple[str, str]:
    """Get date range for last N months (approximately)"""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse not found")
    
    con = duckdb.connect(DB_PATH, read_only=True)
    maxdate = _get_max_date(con)
    con.close()
    
    end_date = maxdate
    # Approximate: 30 days per month
    start_date = maxdate - timedelta(days=n*30-1)
    return str(start_date), str(end_date)

def _format_response(data: dict) -> str:
    """Format API response into natural language"""
    
    current_period = data["period"]["current"]
    prior_period = data["period"]["prior"]
    cac = data["kpis"]["cac"]
    roas = data["kpis"]["roas"]
    
    def format_delta(delta_abs, delta_pct):
        if delta_abs is None or delta_pct is None:
            return "N/A"
        direction = "📈" if delta_abs > 0 else "📉" if delta_abs < 0 else "➡️"
        sign = "+" if delta_abs > 0 else ""
        return f"{direction} {sign}{delta_abs:.2f} ({sign}{delta_pct:.1f}%)"
    
    def format_currency(value):
        return f"${value:.2f}" if value is not None else "N/A"
    
    def format_multiplier(value):
        return f"{value:.2f}x" if value is not None else "N/A"
    
    response = f"""📊 **Metrics Analysis**

**Period Comparison:**
• Current: {current_period['start']} to {current_period['end']} ({current_period['days']} days)
• Prior: {prior_period['start']} to {prior_period['end']} ({prior_period['days']} days)

**CAC (Customer Acquisition Cost):**
• Current: {format_currency(cac['current'])}
• Prior: {format_currency(cac['prior'])}
• Change: {format_delta(cac['delta_abs'], cac['delta_pct'])}

**ROAS (Return on Ad Spend):**
• Current: {format_multiplier(roas['current'])}
• Prior: {format_multiplier(roas['prior'])}
• Change: {format_delta(roas['delta_abs'], roas['delta_pct'])}"""

    return response

@app.get("/ask")
def ask_question(
    question: str = Query(..., description="Natural language question about metrics"),
    format: str = Query(default="json", description="Response format: 'json' or 'text'")
):
    """
    Natural language interface for metrics queries.
    Example: 'Compare CAC and ROAS for last 30 days vs prior 30 days'
    """
    
    # Parse the question
    start, end = _parse_natural_language(question)
    
    # Get metrics data using existing function
    try:
        data = metrics(start=start, end=end)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process question: {str(e)}")
    
    # Format as natural language
    formatted_response = _format_response(data)
    
    # Return plain text if requested
    if format == "text":
        return PlainTextResponse(formatted_response)
    
    # Default JSON response
    return {
        "question": question,
        "detected_params": {"start": start, "end": end} if start and end else {"default": "last 30 days"},
        "formatted_answer": formatted_response,
        "raw_data": data
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