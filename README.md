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

## 🧠 Part 5 - LangChain + AI Integration (Advanced Natural Language Processing)

### Overview
The project now includes a sophisticated LangChain integration powered by Ollama, providing enterprise-grade natural language processing for both KPI analysis and SQL query generation.

### Architecture

```
Natural Language Question → LangChain Intent Chain → LangChain SQL Chain → Database → Results
                         ↓                        ↓
                  Intent Parser              SQL Generator
                  (Ollama LLM)               (Ollama LLM)
```

### LangChain Components

#### 1. **Intent Recognition Chain**
```python
self.intent_chain = LLMChain(
    llm=OllamaLLM(model="llama3.2:3b"),
    prompt=intent_prompt,
    output_parser=MetricsIntentParser()
)
```

Automatically detects:
- **Question Type**: explain, compare, analyze, show, sql
- **Metrics Focus**: CAC, ROAS, or both
- **Time Context**: extracted time periods
- **SQL Intent**: whether user wants raw SQL queries

#### 2. **SQL Generation Chain**
```python
self.sql_chain = LLMChain(
    llm=OllamaLLM(model="llama3.2:3b"),
    prompt=sql_prompt,
    output_parser=SQLQueryParser()
)
```

Features:
- **Database Schema Aware**: Real-time schema reflection
- **Gold Table Priority**: Uses `mart.kpi_ads_30d` for KPI queries
- **Raw Data Access**: Uses `bronze.ads_spend` for granular breakdowns
- **SQL Validation**: Safety checks prevent dangerous operations

#### 3. **Database Schema Reflection**
```python
class DatabaseSchemaReflector:
    def get_schema_info(self) -> Dict[str, Any]:
        # Discovers actual database structure
        # Provides sample data for context
        # Creates LLM-friendly schema descriptions
```

### Enhanced Endpoints

#### **`/ask-ai` - Advanced Natural Language Interface**
- **Multi-modal responses**: data, sql, both, auto
- **Real-time analysis**: Uses actual Ollama LLM when available
- **Intelligent fallbacks**: Works even when LLM is offline

#### **`/ask-sql` - SQL Generation Service**
- **Natural language → SQL**: Convert questions to executable queries
- **Query execution**: Optional SQL execution with results
- **Safety validation**: Read-only operations only

### Usage Examples

#### KPI Analysis (Gold Table)
```bash
curl "http://localhost:8001/ask-ai?question=Why did CAC get worse last month?"
```

**Response**: Deep root cause analysis using actual KPI data from `mart.kpi_ads_30d`

```json
{
  "question": "Why did CAC get worse last month?",
  "langchain_processing": {
    "intent_detection": {"intent": "explain", "confidence": 0.9},
    "llm_powered": true,
    "framework": "LangChain + Ollama",
    "model": "llama3.2:3b"
  },
  "ai_analysis": "🔍 **Data-Driven Root Cause Analysis**\n\n**Real Data Analysis**:\n• **CAC**: $29.81 (↑7.6% vs $32.27)\n• **Spend**: $1,690,764 (↑15.2%)\n• **Conversions**: 54,917 (↓8.1%)\n\n**📊 Root Causes**: Efficiency Drop: Spend increased but conversions dropped..."
}
```

#### SQL Generation
```bash
curl "http://localhost:8001/ask-sql?question=Show me CAC and ROAS performance&execute=true"
```

**Response**: Intelligent SQL generation with execution

```json
{
  "sql_generation": {
    "sql": "SELECT metric, last_30, prior_30, delta_abs, delta_pct FROM mart.kpi_ads_30d WHERE metric IN ('CAC', 'ROAS')",
    "source": "langchain_llm",
    "validated": true
  },
  "execution": {
    "success": true,
    "results": [
      {"metric": "CAC", "last_30": 29.8093, "prior_30": 32.2715, "delta_pct": -7.63},
      {"metric": "ROAS", "last_30": 3.3547, "prior_30": 3.0987, "delta_pct": 8.26}
    ]
  }
}
```

#### Platform Breakdown (Raw Data)
```bash
curl "http://localhost:8001/ask-sql?question=Break down spend by platform&execute=true"
```

**Response**: Automatically uses raw data table for granular analysis

```json
{
  "sql_generation": {
    "sql": "SELECT platform, SUM(spend) AS total_spend FROM bronze.ads_spend GROUP BY platform",
    "source": "langchain_llm"
  },
  "execution": {
    "results": [
      {"platform": "Google", "total_spend": 847845.56},
      {"platform": "Meta", "total_spend": 842918.76}
    ]
  }
}
```

### Advanced Features

#### Intelligent Table Selection
The LangChain system automatically chooses the appropriate data source:

- **KPI Questions** → `mart.kpi_ads_30d` (Gold Table)
  - "What is our CAC performance?"
  - "Show me ROAS trends"
  - "Compare current vs prior metrics"

- **Breakdown Questions** → `bronze.ads_spend` (Raw Data)
  - "Spend by platform"
  - "Daily conversion trends"
  - "Campaign performance breakdown"

#### Response Modes
```bash
# Return analysis only
?mode=data

# Return SQL query only
?mode=sql

# Return both SQL and analysis
?mode=both

# Let AI decide based on question
?mode=auto
```

#### Real-time Schema Context
The LLM receives actual database structure:

```
DATABASE TABLES:

Table: mart.kpi_ads_30d
Columns:
  - metric: VARCHAR (nullable)
  - last_30: DOUBLE (nullable)
  - prior_30: DOUBLE (nullable)
  - delta_abs: DOUBLE (nullable)
  - delta_pct: DOUBLE (nullable)

SAMPLE DATA:
Table: mart.kpi_ads_30d
  Row 1: {"metric": "CAC", "last_30": 29.8093, "delta_pct": -7.63}
  Row 2: {"metric": "ROAS", "last_30": 3.3547, "delta_pct": 8.26}
```

### Installation & Setup

#### Prerequisites
```bash
# Install Ollama (macOS)
brew install ollama
brew services start ollama

# Download model
ollama pull llama3.2:3b
```

#### Dependencies
```bash
pip install langchain langchain_ollama
```

#### LangChain-specific Environment
```bash
# Start enhanced API with LangChain
cd api
python langchain_api.py
```

The API runs on `http://localhost:8001` with full LangChain capabilities.

### Technical Implementation

#### Core LangChain Architecture
```python
class MetricsLLM:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)
        self.db_reflector = DatabaseSchemaReflector(DB_PATH)

        # LangChain Chains
        self.intent_chain = LLMChain(llm, intent_prompt, intent_parser)
        self.sql_chain = LLMChain(llm, sql_prompt, sql_parser)
        self.analysis_chain = LLMChain(llm, analysis_prompt, response_parser)
```

#### Data Flow
1. **Question Input**: "Why did CAC worsen?"
2. **Intent Detection**: LangChain classifies as "explain" + "CAC focus"
3. **Context Injection**: Database schema + sample data provided to LLM
4. **Query Generation**: LLM generates appropriate SQL for gold table
5. **Execution**: Query runs safely with validation
6. **Analysis**: LLM provides root cause analysis of actual results

### Business Value

#### For Data Analysts
- **Natural Queries**: Ask questions in plain English
- **Instant SQL**: Get optimized queries for complex analysis
- **Context-Aware**: System knows your data structure

#### For Business Users
- **Self-Service**: Direct access to insights without SQL knowledge
- **Real-time**: Immediate answers to business questions
- **Explanatory**: AI explains the "why" behind metric changes

#### For Engineering Teams
- **Extensible**: Easy to add new question patterns and tables
- **Safe**: Built-in SQL validation prevents harmful operations
- **Scalable**: LangChain architecture supports complex workflows

### Comparison: Basic vs LangChain Integration

| Feature | Basic Agent (Part 4) | LangChain Integration (Part 5) |
|---------|----------------------|--------------------------------|
| **NL Processing** | Regex patterns | Real LLM understanding |
| **SQL Generation** | None | Dynamic query creation |
| **Data Awareness** | Static responses | Real-time schema reflection |
| **Question Types** | Time-based only | Unlimited natural language |
| **Explanations** | Template-based | AI-generated insights |
| **Table Selection** | Fixed | Intelligent routing |
| **Extensibility** | Manual patterns | Self-learning from data |

The LangChain integration transforms the system from a pattern-matching interface into a truly intelligent data assistant that understands your business context and generates actionable insights.

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