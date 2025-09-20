# Shadowlights KPIs - AI Data Engineer Test

This project demonstrates a complete data engineering pipeline for analyzing advertising performance metrics, including data ingestion, KPI modeling, and natural language query capabilities.

## 🏗️ Project Overview

The solution implements a modern data stack for processing advertising spend data and computing key performance indicators (CAC and ROAS) with period-over-period analysis.

### Architecture
```
ads_spend.csv → n8n (Orchestration) → DuckDB (Warehouse) → FastAPI (Analytics) → JSON/Text Output
```

---

## 📊 Part 1 - Ingestion (Foundation)

### Implementation
- **Orchestration Tool**: n8n workflow automation
- **Data Warehouse**: DuckDB (lightweight, embedded SQL database)
- **Source Data**: `ads_spend.csv` with advertising performance metrics

### Dataset Schema
```sql
CREATE TABLE bronze.ads_spend (
    date DATE,
    platform VARCHAR,
    account VARCHAR, 
    campaign VARCHAR,
    country VARCHAR,
    device VARCHAR,
    spend DECIMAL(10,2),
    clicks INTEGER,
    impressions INTEGER,
    conversions INTEGER,
    load_date TIMESTAMP,     -- Metadata: when data was loaded
    source_file_name VARCHAR  -- Metadata: source file provenance
);
```

### Key Features
- **Data Persistence**: DuckDB file storage ensures data survives container restarts
- **Metadata Tracking**: Each record includes load timestamp and source file name
- **Automated Ingestion**: n8n workflow handles file processing and database loading

### Files
- `n8n/Dockerfile` - Custom n8n container configuration
- `docker-compose.yaml` - Complete environment orchestration
- `workspace/ads_spend.csv` - Source advertising data
- `workspace/warehouse.duckdb` - DuckDB database file

---

## 🧮 Part 2 - KPI Modeling (SQL)

### KPI Definitions

#### Customer Acquisition Cost (CAC)
```sql
CAC = spend / conversions
```
- Measures cost to acquire one customer
- Lower values indicate more efficient spending

#### Return on Ad Spend (ROAS)  
```sql
ROAS = revenue / spend
-- Where revenue = conversions × $100 (assumed conversion value)
```
- Measures return for every dollar spent
- Higher values indicate better performance

### Analysis Framework
- **Period Comparison**: Last 30 days vs prior 30 days
- **Delta Calculations**: Absolute change and percentage change
- **Compact Results**: Single view with current, prior, and delta metrics

### Implementation Files
- `workspace/sql/01_exploration.sql` - Data exploration queries
- `workspace/sql/02_kpi_modeling_steps.sql` - KPI calculation logic  
- `workspace/sql/03_kpi_mart_views.sql` - Final mart views
- `api/metrics_api.py` - Embedded SQL logic in Python API

### Sample Output Format
```json
{
  "kpis": {
    "cac": {
      "current": 25.50,
      "prior": 22.00, 
      "delta_abs": 3.50,
      "delta_pct": 15.91
    },
    "roas": {
      "current": 3.92,
      "prior": 4.55,
      "delta_abs": -0.63,
      "delta_pct": -13.85
    }
  }
}
```

---

## 🔌 Part 3 - Analyst Access

### FastAPI Metrics Endpoint

**Primary Endpoint**: `GET /metrics`

#### Parameters
- `start` (optional): Start date in YYYY-MM-DD format
- `end` (optional): End date in YYYY-MM-DD format
- If no dates provided: defaults to last 30 days

#### Response Structure
```json
{
  "period": {
    "current": {"start": "2024-01-01", "end": "2024-01-31", "days": 31},
    "prior": {"start": "2023-12-01", "end": "2023-12-31", "days": 31}
  },
  "totals": {
    "current": {"spend": 65075.36, "conversions": 2232},
    "prior": {"spend": 61304.74, "conversions": 2158}
  },
  "kpis": {
    "cac": {"current": 29.16, "prior": 28.41, "delta_abs": 0.75, "delta_pct": 2.6},
    "roas": {"current": 3.43, "prior": 3.52, "delta_abs": -0.09, "delta_pct": -2.6}
  }
}
```

#### Usage Examples
```bash
# Default analysis (last 30 days)
curl http://127.0.0.1:8000/metrics

# Custom date range
curl "http://127.0.0.1:8000/metrics?start=2024-01-01&end=2024-01-31"

# Interactive API docs
open http://127.0.0.1:8000/docs
```

### Features
- **Automatic Period Calculation**: Prior period automatically calculated as same length immediately before current period
- **Safe Division**: Handles zero conversions without errors
- **Date Validation**: Ensures proper date formats and logical ranges
- **Database Connection Management**: Efficient DuckDB connection handling

---

## 🤖 Part 4 - Agent Demo (Natural Language Interface)

### Overview
This bonus feature demonstrates how natural language questions can be automatically converted into API calls and formatted responses, simulating an AI assistant for data analysis.

### Natural Language Endpoint

**Endpoint**: `GET /ask`

#### Parameters
- `question` (required): Natural language question about metrics
- `format` (optional): Response format - `"json"` (default) or `"text"`

#### Supported Question Patterns

**Time-based Patterns:**
- `"Compare CAC and ROAS for last 30 days vs prior 30 days"`
- `"Show me metrics for the past 7 days"`
- `"last 3 weeks"` / `"past 2 months"`
- `"this month"` / `"last month"`
- `"this week"` / `"last week"`
- `"today"` / `"yesterday"`

**Date Range Patterns:**
- `"How did we perform from 2024-01-01 to 2024-01-31?"`
- `"Analyze performance between 2025-06-01 and 2025-06-15"`

**Pattern Categories:**
- **Relative Days**: `last/past N days`
- **Relative Weeks**: `last/past N weeks` (supports singular/plural)
- **Relative Months**: `last/past N months` (supports singular/plural)
- **Named Periods**: `this/last month`, `this/last week`
- **Single Days**: `today`, `yesterday`
- **Explicit Ranges**: `from DATE to DATE`, `between DATE and DATE`

### Technical Implementation

#### 1. Pattern Recognition Engine
```python
def _parse_natural_language(question: str) -> Optional[tuple[str, str]]:
    patterns = {
        r"last (\d+) days": lambda m: _get_last_n_days(int(m.group(1))),
        r"past (\d+) days": lambda m: _get_last_n_days(int(m.group(1))),
        r"from ([\d-]+) to ([\d-]+)": lambda m: (m.group(1), m.group(2)),
        r"between ([\d-]+) and ([\d-]+)": lambda m: (m.group(1), m.group(2)),
    }
```

Uses regex patterns to extract:
- **Time periods**: "last 30 days", "past 7 days"
- **Date ranges**: "from 2024-01-01 to 2024-01-31"
- **Alternative phrasing**: "between X and Y"

#### 2. Response Formatting Engine
```python
def _format_response(data: dict) -> str:
    """Convert JSON metrics to natural language explanation"""
```

Converts structured API response into human-readable analysis with:
- Period comparison summary
- Current vs prior metrics
- Visual indicators (📈📉) for trends
- Percentage change calculations

#### 3. API Integration
The `/ask` endpoint:
1. Parses natural language question
2. Extracts date parameters
3. Calls existing `/metrics` endpoint
4. Formats response as readable text
5. Returns both formatted answer and raw data

### Usage Examples

#### Request
```bash
curl "http://127.0.0.1:8000/ask?question=Compare CAC and ROAS for last 7 days vs prior 7 days"
```

#### Response Options

**JSON Format** (default):
```json
{
  "question": "Compare CAC and ROAS for last 7 days vs prior 7 days",
  "detected_params": {"start": "2025-06-24", "end": "2025-06-30"},
  "formatted_answer": "📊 **Metrics Analysis**\n\n**Period Comparison:**...",
  "raw_data": { /* full metrics JSON */ }
}
```

**Text Format** (with `?format=text`):
```
📊 **Metrics Analysis**

**Period Comparison:**
• Current: 2025-06-24 to 2025-06-30 (7 days)
• Prior: 2025-06-17 to 2025-06-23 (7 days)

**CAC (Customer Acquisition Cost):**
• Current: $29.16
• Prior: $28.41
• Change: 📈 +$0.75 (+2.6%)

**ROAS (Return on Ad Spend):**
• Current: 3.43x
• Prior: 3.52x
• Change: 📉 -0.09x (-2.6%)
```

### Question → API Mapping Examples

| Natural Language Question | Detected Pattern | API Call |
|---------------------------|------------------|----------|
| "Compare CAC and ROAS for last 30 days" | `last (\d+) days` | `/metrics?start=2025-05-21&end=2025-06-20` |
| "Show metrics for past 7 days" | `past (\d+) days` | `/metrics?start=2025-06-14&end=2025-06-20` |
| "last 3 weeks" | `last (\d+) weeks?` | `/metrics?start=2025-06-10&end=2025-06-30` |
| "past 2 months" | `past (\d+) months?` | `/metrics?start=2025-04-30&end=2025-06-30` |
| "this month" | `this month` | `/metrics?start=2025-06-01&end=2025-06-30` |
| "last month" | `last month` | `/metrics?start=2025-05-01&end=2025-05-31` |
| "this week" | `this week` | `/metrics?start=2025-06-30&end=2025-06-30` |
| "last week" | `last week` | `/metrics?start=2025-06-23&end=2025-06-29` |
| "today" | `today` | `/metrics?start=2025-06-30&end=2025-06-30` |
| "yesterday" | `yesterday` | `/metrics?start=2025-06-29&end=2025-06-29` |
| "From 2024-01-01 to 2024-01-31" | `from ([\d-]+) to ([\d-]+)` | `/metrics?start=2024-01-01&end=2024-01-31` |
| "Between 2024-02-01 and 2024-02-28" | `between ([\d-]+) and ([\d-]+)` | `/metrics?start=2024-02-01&end=2024-02-28` |

### Advanced Features

#### Smart Date Calculation
- **Relative dates**: "last 30 days" uses actual data max date, not current date
- **Period alignment**: Prior period automatically calculated as same duration
- **Data-aware**: Respects actual data availability in warehouse

#### Error Handling
- **Pattern not recognized**: Falls back to default 30-day analysis
- **Invalid dates**: Returns clear error messages
- **Database errors**: Graceful error handling with helpful messages

#### Extensibility
The pattern recognition system can easily be extended with new question types:
```python
# Add new patterns
r"this month": lambda m: _get_current_month(),
r"last quarter": lambda m: _get_last_quarter(),
r"year over year": lambda m: _get_yoy_comparison(),
```

### Business Value

This natural language interface demonstrates:
1. **Accessibility**: Non-technical users can query data using plain English
2. **Efficiency**: Instant insights without learning API syntax
3. **Scalability**: Pattern-based approach easily extends to new question types
4. **Integration Ready**: Foundation for chatbot or voice assistant integration

The implementation showcases modern data democratization principles, making analytics accessible through natural conversation while maintaining the precision of structured APIs underneath.

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+

### Setup
```bash
# Clone and navigate to project
cd shadowlights-kpis

# Start the environment
docker-compose up -d

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn duckdb python-dateutil

# Start the API
uvicorn api.metrics_api:app --reload
```

### Test the Implementation
```bash
# Test basic metrics
curl http://127.0.0.1:8000/metrics

# Test natural language interface
curl "http://127.0.0.1:8000/ask?question=Compare CAC and ROAS for last 7 days"

# Get formatted text output
curl "http://127.0.0.1:8000/ask?question=last 30 days&format=text"

# View interactive docs
open http://127.0.0.1:8000/docs
```

---

## 📁 Project Structure

```
shadowlights-kpis/
├── README.md                 # This file
├── docker-compose.yaml       # Environment orchestration
├── api/
│   └── metrics_api.py        # FastAPI application
├── n8n/
│   └── Dockerfile           # Custom n8n configuration
├── workspace/
│   ├── ads_spend.csv        # Source data
│   ├── warehouse.duckdb     # DuckDB database
│   ├── pipeline.py          # Data processing scripts
│   └── sql/                 # SQL analysis files
│       ├── 01_exploration.sql
│       ├── 02_kpi_modeling_steps.sql
│       └── 03_kpi_mart_views.sql
└── n8n_data/               # n8n persistent data
```

---

## 🔧 Technical Decisions

### Why DuckDB?
- **Lightweight**: Single file database, perfect for development
- **SQL Compatible**: Full ANSI SQL support
- **Python Integration**: Excellent Python bindings
- **Performance**: Fast analytical queries

### Why FastAPI?
- **Modern**: Async support, automatic API docs
- **Type Safety**: Pydantic validation
- **Developer Experience**: Interactive documentation
- **Performance**: Fast JSON serialization

### Why n8n?
- **Visual Workflows**: Easy to understand and modify
- **Extensible**: Rich ecosystem of integrations
- **Self-hosted**: Full control over data processing
- **Docker Ready**: Easy deployment and scaling

---

## 📈 Future Enhancements

1. **Advanced Analytics**: Add more KPIs (LTV, retention, attribution)
2. **Real-time Processing**: Stream processing for live dashboards  
3. **Data Quality**: Implement validation and monitoring
4. **Visualization**: Add charts and dashboards
5. **Machine Learning**: Predictive analytics and anomaly detection
6. **Natural Language**: Enhanced NLP with LLM integration

---

## 📝 Notes

- The project demonstrates practical data engineering patterns suitable for production scaling
- All components are containerized for easy deployment
- The API design follows RESTful principles with comprehensive error handling
- SQL queries are optimized for analytical workloads
- The natural language interface showcases modern AI integration possibilities