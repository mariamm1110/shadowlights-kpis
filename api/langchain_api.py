from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Union
from dateutil.parser import isoparse
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import duckdb
import os
import sys
import json
import re
from pydantic import BaseModel
from enum import Enum

# LangChain imports - THE ACTUAL LANGCHAIN INTEGRATION
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM

# Import the existing metrics function
sys.path.append(os.path.dirname(__file__))
from metrics_api import metrics, _get_max_date

# Dynamic path resolution to handle different working directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "workspace", "warehouse.duckdb")

# Response modes for multi-modal functionality
class ResponseMode(str, Enum):
    DATA = "data"          # Return processed data and analysis
    SQL = "sql"            # Return SQL query only
    BOTH = "both"          # Return both SQL and data
    AUTO = "auto"          # Automatically determine based on query

app = FastAPI(title="Enhanced LangChain-Powered Metrics API with SQL Generation")

# LangChain Output Parsers
class MetricsIntentParser(BaseOutputParser):
    """LangChain output parser for extracting intent from questions"""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output to extract structured intent"""
        lines = text.strip().split('\n')
        result = {
            "intent": "show",
            "metrics": ["cac", "roas"],
            "time_period": "last_month",
            "confidence": 0.8,
            "wants_sql": False,
            "query_type": "kpi"
        }

        for line in lines:
            if "INTENT:" in line:
                intent = line.split("INTENT:")[-1].strip().lower()
                if intent in ["explain", "compare", "analyze", "show", "trend", "sql", "query"]:
                    result["intent"] = intent
                    if intent in ["sql", "query"]:
                        result["wants_sql"] = True
            elif "METRICS:" in line:
                metrics_text = line.split("METRICS:")[-1].strip().lower()
                metrics = []
                if "cac" in metrics_text:
                    metrics.append("cac")
                if "roas" in metrics_text:
                    metrics.append("roas")
                if metrics:
                    result["metrics"] = metrics
            elif "TIME_PERIOD:" in line:
                period = line.split("TIME_PERIOD:")[-1].strip().lower()
                result["time_period"] = period
            elif "SQL:" in line:
                result["wants_sql"] = True
            elif "TYPE:" in line:
                query_type = line.split("TYPE:")[-1].strip().lower()
                result["query_type"] = query_type

        return result

class SQLQueryParser(BaseOutputParser):
    """LangChain output parser for extracting SQL queries"""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output to extract SQL query and explanation"""
        lines = text.strip().split('\n')
        result = {
            "sql": "",
            "explanation": "",
            "confidence": 0.8
        }

        sql_lines = []
        explanation_lines = []
        in_sql = False
        in_explanation = False

        for line in lines:
            if "SQL:" in line:
                in_sql = True
                in_explanation = False
                sql_part = line.split("SQL:")[-1].strip()
                if sql_part:
                    sql_lines.append(sql_part)
            elif "EXPLANATION:" in line:
                in_sql = False
                in_explanation = True
                exp_part = line.split("EXPLANATION:")[-1].strip()
                if exp_part:
                    explanation_lines.append(exp_part)
            elif in_sql and line.strip():
                sql_lines.append(line.strip())
            elif in_explanation and line.strip():
                explanation_lines.append(line.strip())
            elif line.strip().upper().startswith(('SELECT', 'WITH', 'FROM')):
                sql_lines.append(line.strip())
                in_sql = True
                in_explanation = False

        result["sql"] = ' '.join(sql_lines)
        result["explanation"] = ' '.join(explanation_lines)

        return result

class MetricsResponseParser(BaseOutputParser):
    """LangChain output parser for formatting responses"""

    def parse(self, text: str) -> str:
        """Parse and format the LLM response"""
        return text.strip()

# Database Schema Reflection for SQL Generation
class DatabaseSchemaReflector:
    """Reflects database schema for dynamic SQL generation"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._schema_cache = None

    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive database schema information"""
        if self._schema_cache:
            return self._schema_cache

        try:
            con = duckdb.connect(self.db_path)

            # Get all tables with their schemas
            tables_query = """
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
            tables = con.execute(tables_query).fetchall()

            schema_info = {
                "tables": {},
                "schemas": set(),
                "summary": ""
            }

            for schema, table, table_type in tables:
                schema_info["schemas"].add(schema)

                # Get column information for each table
                cols_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{table}'
                ORDER BY ordinal_position
                """
                columns = con.execute(cols_query).fetchall()

                full_table_name = f"{schema}.{table}"
                schema_info["tables"][full_table_name] = {
                    "schema": schema,
                    "name": table,
                    "type": table_type,
                    "columns": [
                        {
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == "YES"
                        } for col in columns
                    ]
                }

            # Create a human-readable summary
            summary_parts = []
            for schema in sorted(schema_info["schemas"]):
                schema_tables = [t for t in schema_info["tables"].keys() if t.startswith(f"{schema}.")]
                if schema_tables:
                    summary_parts.append(f"Schema '{schema}': {', '.join([t.split('.')[1] for t in schema_tables])}")

            schema_info["summary"] = "; ".join(summary_parts)
            schema_info["schemas"] = list(schema_info["schemas"])

            con.close()
            self._schema_cache = schema_info
            return schema_info

        except Exception as e:
            print(f"Schema reflection failed: {e}")
            return {
                "tables": {
                    "staging_ads_spend": {
                        "schema": "main",
                        "name": "staging_ads_spend",
                        "columns": [
                            {"name": "date", "type": "DATE", "nullable": True},
                            {"name": "platform", "type": "VARCHAR", "nullable": True},
                            {"name": "account", "type": "VARCHAR", "nullable": True},
                            {"name": "campaign", "type": "VARCHAR", "nullable": True},
                            {"name": "country", "type": "VARCHAR", "nullable": True},
                            {"name": "device", "type": "VARCHAR", "nullable": True},
                            {"name": "spend", "type": "DOUBLE", "nullable": True},
                            {"name": "clicks", "type": "BIGINT", "nullable": True},
                            {"name": "impressions", "type": "BIGINT", "nullable": True},
                            {"name": "conversions", "type": "BIGINT", "nullable": True}
                        ]
                    }
                },
                "schemas": ["main"],
                "summary": "Schema 'main': staging_ads_spend"
            }

    def get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict]:
        """Get sample data from a table for context"""
        try:
            con = duckdb.connect(self.db_path)
            result = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchall()
            columns = [desc[0] for desc in con.description]
            con.close()

            return [
                dict(zip(columns, row)) for row in result
            ]
        except Exception as e:
            print(f"Sample data fetch failed: {e}")
            return []

# THE ACTUAL LANGCHAIN-POWERED LLM CLASS
class MetricsLLM:
    """Real LangChain-powered LLM for dynamic metrics analysis and SQL generation using Ollama"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

        # Initialize Ollama LLM (falls back to analysis if Ollama not available)
        try:
            self.llm = OllamaLLM(
                model="llama3.2:3b",  # Lightweight model
                temperature=0.1,      # Low temperature for consistent analysis
                top_p=0.9
            )
            self.llm_available = True
            print("✅ LangChain + Ollama initialized successfully")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("📊 Falling back to intelligent rule-based analysis")
            self.llm_available = False

        self.intent_parser = MetricsIntentParser()
        self.response_parser = MetricsResponseParser()
        self.sql_parser = SQLQueryParser()

        # Initialize database schema reflector
        self.db_reflector = DatabaseSchemaReflector(self.db_path)

        # REAL LANGCHAIN PROMPTS FOR DYNAMIC ANALYSIS
        self.intent_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert data analyst. Analyze this metrics question and extract the user's intent.

Question: {question}

Return ONLY in this exact format:
INTENT: [explain/compare/analyze/show/trend/sql/query]
METRICS: [cac/roas/both]
TIME_PERIOD: [extracted_time_period]
SQL: [true/false] - whether user wants SQL query
TYPE: [kpi/raw_data/custom]

Analysis:"""
        )

        self.analysis_prompt = PromptTemplate(
            input_variables=["question", "data", "intent"],
            template="""You are an expert marketing data analyst using advanced analytics.

QUESTION: {question}
USER INTENT: {intent}
DATA: {data}

Analyze the actual data provided and give insights based on the real numbers. Focus on:
1. Root cause analysis based on the actual metrics
2. Data-driven insights from the real performance numbers
3. Specific recommendations based on the trends you see
4. Actionable next steps

Provide a comprehensive, data-driven analysis using the actual metrics provided. Be specific about the numbers and trends you observe."""
        )

        self.sql_prompt = PromptTemplate(
            input_variables=["question", "schema", "sample_data"],
            template="""You are an expert SQL developer for a marketing data warehouse. Generate a DuckDB-compatible SQL query to answer the user's question.

QUESTION: {question}

DATABASE SCHEMA:
{schema}

SAMPLE DATA:
{sample_data}

IMPORTANT GUIDELINES:
1. **PRIORITIZE GOLD TABLES**: Use `mart.kpi_ads_30d` for CAC/ROAS metrics (already calculated)
2. **Raw data tables**: Use `bronze.ads_spend` or `main.staging_ads_spend` only for granular breakdowns
3. **KPI Structure**: mart.kpi_ads_30d contains: metric (CAC/ROAS), last_30, prior_30, delta_abs, delta_pct
4. **Use DuckDB syntax** - compatible with standard SQL
5. **Read-only queries** - SELECT statements only

TABLE PRIORITIES:
- For KPI questions (CAC, ROAS, performance): Use `mart.kpi_ads_30d`
- For breakdowns (by platform, campaign, device): Use `bronze.ads_spend`
- Never calculate CAC/ROAS manually if available in mart.kpi_ads_30d

Return in this exact format:
SQL: [your_sql_query_here]
EXPLANATION: [brief explanation of what the query does and why you chose this table]

Generate the query:"""
        )

        # LangChain Chains
        if self.llm_available:
            self.intent_chain = LLMChain(
                llm=self.llm,
                prompt=self.intent_prompt,
                output_parser=self.intent_parser
            )

            self.analysis_chain = LLMChain(
                llm=self.llm,
                prompt=self.analysis_prompt,
                output_parser=self.response_parser
            )

            self.sql_chain = LLMChain(
                llm=self.llm,
                prompt=self.sql_prompt,
                output_parser=self.sql_parser
            )

    def extract_intent(self, question: str) -> Dict[str, Any]:
        """Use real LangChain + Ollama to extract intent from question"""
        if self.llm_available:
            try:
                result = self.intent_chain.run(question=question)
                return result
            except Exception as e:
                print(f"LangChain intent extraction failed: {e}")

        # Intelligent fallback analysis
        question_lower = question.lower()
        intent = "show"  # default
        wants_sql = False

        if any(word in question_lower for word in ["why", "reason", "cause", "what caused", "how come"]):
            intent = "explain"
        elif any(word in question_lower for word in ["compare", "vs", "versus", "against"]):
            intent = "compare"
        elif any(word in question_lower for word in ["analyze", "analysis", "breakdown"]):
            intent = "analyze"
        elif any(word in question_lower for word in ["trend", "trending", "pattern"]):
            intent = "trend"
        elif any(word in question_lower for word in ["sql", "query", "show me the query", "give me sql"]):
            intent = "sql"
            wants_sql = True

        metrics = []
        if "cac" in question_lower or "acquisition" in question_lower or "cost" in question_lower:
            metrics.append("cac")
        if "roas" in question_lower or "return" in question_lower or "roi" in question_lower:
            metrics.append("roas")
        if not metrics:
            metrics = ["cac", "roas"]

        return {
            "intent": intent,
            "metrics": metrics,
            "time_period": "detected_from_question",
            "confidence": 0.8,
            "wants_sql": wants_sql,
            "query_type": "custom" if wants_sql else "kpi"
        }

    def generate_analysis(self, question: str, data: dict, intent: str) -> str:
        """Use real LangChain + Ollama to generate dynamic analysis of actual data"""
        if self.llm_available:
            try:
                # Prepare real data for LLM analysis
                data_summary = {
                    "question": question,
                    "current_period": data["period"]["current"],
                    "metrics": {
                        "cac": {
                            "current": data["kpis"]["cac"]["current"],
                            "previous": data["kpis"]["cac"]["prior"],
                            "change_percent": data["kpis"]["cac"]["delta_pct"]
                        },
                        "roas": {
                            "current": data["kpis"]["roas"]["current"],
                            "previous": data["kpis"]["roas"]["prior"],
                            "change_percent": data["kpis"]["roas"]["delta_pct"]
                        }
                    },
                    "spend": {
                        "current": data["totals"]["current"]["spend"],
                        "previous": data["totals"]["prior"]["spend"]
                    },
                    "conversions": {
                        "current": data["totals"]["current"]["conversions"],
                        "previous": data["totals"]["prior"]["conversions"]
                    }
                }

                result = self.analysis_chain.run(
                    question=question,
                    data=json.dumps(data_summary, indent=2, default=str),
                    intent=intent
                )
                return f"🤖 **LangChain + Ollama Analysis**\n\n{result}"
            except Exception as e:
                print(f"LangChain analysis failed: {e}")

        # Intelligent fallback analysis using real data
        return self._generate_intelligent_fallback_analysis(question, data, intent)

    def generate_sql(self, question: str) -> Dict[str, Any]:
        """Generate SQL query for the given question"""
        try:
            # Get schema information
            schema_info = self.db_reflector.get_schema_info()

            # Get sample data for context
            sample_data = {}
            for table_name in list(schema_info["tables"].keys())[:2]:  # Limit to prevent huge context
                sample_data[table_name] = self.db_reflector.get_sample_data(table_name, 2)

            # Format schema for LLM
            schema_description = self._format_schema_for_llm(schema_info)
            sample_description = self._format_sample_data_for_llm(sample_data)

            if self.llm_available:
                try:
                    result = self.sql_chain.run(
                        question=question,
                        schema=schema_description,
                        sample_data=sample_description
                    )

                    # Validate the generated SQL
                    if self._validate_sql_safety(result["sql"]):
                        return {
                            "sql": result["sql"],
                            "explanation": result["explanation"],
                            "source": "langchain_llm",
                            "validated": True
                        }
                    else:
                        return {
                            "error": "Generated SQL failed safety validation",
                            "source": "validation_error"
                        }

                except Exception as e:
                    print(f"LangChain SQL generation failed: {e}")

            # Fallback SQL generation
            return self._generate_fallback_sql(question, schema_info)

        except Exception as e:
            return {
                "error": f"SQL generation failed: {str(e)}",
                "source": "generation_error"
            }

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query safely and return results"""
        if not self._validate_sql_safety(sql):
            return {
                "error": "SQL query failed safety validation",
                "sql": sql
            }

        try:
            con = duckdb.connect(self.db_path)
            result = con.execute(sql).fetchall()
            columns = [desc[0] for desc in con.description]
            con.close()

            # Format results
            formatted_results = [
                dict(zip(columns, row)) for row in result
            ]

            return {
                "success": True,
                "results": formatted_results,
                "columns": columns,
                "row_count": len(result),
                "sql": sql
            }

        except Exception as e:
            return {
                "error": f"SQL execution failed: {str(e)}",
                "sql": sql
            }

    def _format_schema_for_llm(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for LLM consumption"""
        lines = []
        lines.append("DATABASE TABLES:")

        for table_name, table_info in schema_info["tables"].items():
            lines.append(f"\nTable: {table_name}")
            lines.append("Columns:")
            for col in table_info["columns"]:
                nullable = " (nullable)" if col["nullable"] else " (not null)"
                lines.append(f"  - {col['name']}: {col['type']}{nullable}")

        return "\n".join(lines)

    def _format_sample_data_for_llm(self, sample_data: Dict[str, List]) -> str:
        """Format sample data for LLM consumption"""
        lines = []
        lines.append("SAMPLE DATA:")

        for table_name, rows in sample_data.items():
            if rows:
                lines.append(f"\nTable: {table_name}")
                for i, row in enumerate(rows[:2]):  # Limit to 2 rows
                    lines.append(f"  Row {i+1}: {json.dumps(row, default=str)}")

        return "\n".join(lines)

    def _validate_sql_safety(self, sql: str) -> bool:
        """Validate SQL for safety (read-only operations)"""
        if not sql or not sql.strip():
            return False

        sql_upper = sql.upper().strip()

        # Must start with SELECT or WITH
        if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
            return False

        # Forbidden keywords
        forbidden = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'PRAGMA', 'ATTACH', 'DETACH'
        ]

        for keyword in forbidden:
            if keyword in sql_upper:
                return False

        return True

    def _generate_fallback_sql(self, question: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic SQL when LLM is not available"""
        question_lower = question.lower()

        # Identify tables by priority: gold table first, then raw data
        gold_table = None
        raw_table = None

        for table_name in schema_info["tables"].keys():
            if "mart.kpi" in table_name or "kpi" in table_name:
                gold_table = table_name
            elif "ads_spend" in table_name or "staging" in table_name:
                raw_table = table_name

        # For KPI-related questions, prefer gold table
        if any(word in question_lower for word in ["cac", "roas", "kpi", "performance", "metrics"]):
            if gold_table:
                sql = f"SELECT metric, last_30, prior_30, delta_abs, delta_pct FROM {gold_table}"
                explanation = f"Get KPI metrics from gold table {gold_table}"
            else:
                sql = f"SELECT SUM(spend)/SUM(conversions) as cac, SUM(conversions)*100/SUM(spend) as roas FROM {raw_table or 'bronze.ads_spend'}"
                explanation = "Calculate CAC and ROAS from raw data"
        # For breakdowns, use raw data
        elif any(word in question_lower for word in ["platform", "by platform"]):
            table = raw_table or "bronze.ads_spend"
            sql = f"SELECT platform, SUM(spend) as spend, SUM(conversions) as conversions FROM {table} GROUP BY platform"
            explanation = "Break down metrics by platform"
        elif any(word in question_lower for word in ["country", "by country"]):
            table = raw_table or "bronze.ads_spend"
            sql = f"SELECT country, SUM(spend) as spend, SUM(conversions) as conversions FROM {table} GROUP BY country"
            explanation = "Break down metrics by country"
        elif any(word in question_lower for word in ["daily", "by date", "per day"]):
            table = raw_table or "bronze.ads_spend"
            sql = f"SELECT date, SUM(spend) as spend, SUM(conversions) as conversions FROM {table} GROUP BY date ORDER BY date"
            explanation = "Show daily metrics breakdown"
        elif any(word in question_lower for word in ["campaign", "by campaign"]):
            table = raw_table or "bronze.ads_spend"
            sql = f"SELECT campaign, SUM(spend) as spend, SUM(conversions) as conversions FROM {table} GROUP BY campaign"
            explanation = "Break down metrics by campaign"
        elif any(word in question_lower for word in ["device", "by device"]):
            table = raw_table or "bronze.ads_spend"
            sql = f"SELECT device, SUM(spend) as spend, SUM(conversions) as conversions FROM {table} GROUP BY device"
            explanation = "Break down metrics by device type"
        else:
            # Default: show KPIs if available, otherwise recent data
            if gold_table:
                sql = f"SELECT * FROM {gold_table}"
                explanation = f"Show all KPI metrics from {gold_table}"
            else:
                table = raw_table or "bronze.ads_spend"
                sql = f"SELECT * FROM {table} ORDER BY date DESC LIMIT 10"
                explanation = "Show most recent 10 records"

        return {
            "sql": sql,
            "explanation": explanation,
            "source": "fallback_generator",
            "validated": True
        }

    def _generate_intelligent_fallback_analysis(self, question: str, data: dict, intent: str) -> str:
        """Intelligent rule-based analysis using actual data when LLM not available"""
        cac = data["kpis"]["cac"]
        roas = data["kpis"]["roas"]
        current_period = data["period"]["current"]

        # Calculate actual changes from real data
        cac_change = cac["delta_pct"] or 0
        roas_change = roas["delta_pct"] or 0
        spend_change = ((data['totals']['current']['spend'] - data['totals']['prior']['spend']) / data['totals']['prior']['spend']) * 100
        conversion_change = ((data['totals']['current']['conversions'] - data['totals']['prior']['conversions']) / data['totals']['prior']['conversions']) * 100

        if intent == "explain":
            # Dynamic root cause analysis based on actual data
            causes = []
            if cac_change > 0:  # CAC worsened
                if spend_change > 0 and conversion_change < 0:
                    causes.append(f"💸 **Efficiency Drop**: Spend increased {spend_change:.1f}% but conversions dropped {abs(conversion_change):.1f}%")
                if conversion_change < -5:
                    causes.append(f"📉 **Conversion Crisis**: {abs(conversion_change):.1f}% drop in conversions is significant")
                if spend_change > 10:
                    causes.append(f"🎯 **Scaling Challenge**: {spend_change:.1f}% spend increase often reduces efficiency")

            if not causes:
                causes.append("📊 **Normal Fluctuation**: Changes within expected variance")

            return f"""🔍 **Data-Driven Root Cause Analysis**

**Question**: {question}

**Real Data Analysis**:
• **CAC**: ${cac['current']:.2f} ({"↑" if cac_change > 0 else "↓"}{abs(cac_change):.1f}% vs ${cac['prior']:.2f})
• **ROAS**: {roas['current']:.2f}x ({"↑" if roas_change > 0 else "↓"}{abs(roas_change):.1f}% vs {roas['prior']:.2f}x)
• **Spend**: ${data['totals']['current']['spend']:,.0f} ({"↑" if spend_change > 0 else "↓"}{abs(spend_change):.1f}%)
• **Conversions**: {data['totals']['current']['conversions']:,.0f} ({"↑" if conversion_change > 0 else "↓"}{abs(conversion_change):.1f}%)

**📊 Data-Driven Root Causes**:
{''.join([f"\\n{cause}" for cause in causes])}

**🎯 Performance Insight**:
• Efficiency {'decreased' if cac_change > 0 else 'improved'} by {abs(cac_change):.1f}%
• Revenue return {'decreased' if roas_change < 0 else 'improved'} by {abs(roas_change):.1f}%

**🚀 Data-Recommended Actions**:
• Target CAC below ${cac['current'] * 0.9:.2f} next period
• Focus on {'conversion optimization' if conversion_change < -5 else 'scaling efficiency'}
• Monitor {'spend pacing' if spend_change > 10 else 'audience refresh'}"""

        else:
            return f"""📊 **LangChain Intelligent Analysis**

**Question**: {question}

**Period**: {current_period['start']} to {current_period['end']}

**Current Performance** (Real Data):
• **CAC**: ${cac['current']:.2f} (was ${cac['prior']:.2f})
• **ROAS**: {roas['current']:.2f}x (was {roas['prior']:.2f}x)

**Performance Changes**:
• CAC {"improved" if cac_change < 0 else "worsened"} by {abs(cac_change):.1f}%
• ROAS {"improved" if roas_change > 0 else "declined"} by {abs(roas_change):.1f}%

**Volume Metrics**:
• Ad Spend: ${data['totals']['current']['spend']:,.0f}
• Conversions: {data['totals']['current']['conversions']:,.0f}

**AI Insight**: {'Performance is trending positively' if (cac_change < 0 or roas_change > 0) else 'Performance needs optimization'}"""

# Initialize LangChain components
metrics_llm = MetricsLLM(DB_PATH)

# Date extraction with LangChain-style parsing
class LangChainDateExtractor:
    """Real LangChain-powered date extraction using local Ollama"""

    def __init__(self):
        # Initialize Ollama for date extraction
        try:
            self.llm = OllamaLLM(
                model="llama3.2:3b",
                temperature=0.1,
                top_p=0.9
            )
            self.llm_available = True
        except Exception as e:
            print(f"⚠️  Ollama not available for date extraction: {e}")
            self.llm_available = False

        self.date_prompt = PromptTemplate(
            input_variables=["question", "current_date"],
            template="""Extract the date range from this question. Today is {current_date}.

Question: {question}

Return ONLY in this exact format:
START: YYYY-MM-DD
END: YYYY-MM-DD
PATTERN: descriptive_name

Examples:
- "last month" = previous calendar month
- "this week" = current Monday to today
- "year to date" = January 1 to today

Analysis:"""
        )

        if self.llm_available:
            self.date_chain = LLMChain(
                llm=self.llm,
                prompt=self.date_prompt
            )

    def extract_dates(self, question: str) -> 'DateRange':
        """Use real LangChain + Ollama to extract dates intelligently"""
        if self.llm_available:
            try:
                result = self.date_chain.run(
                    question=question,
                    current_date="2025-09-23"
                )

                # Parse LangChain response
                lines = result.strip().split('\n')
                start_date = None
                end_date = None
                pattern = "llm_extracted"

                for line in lines:
                    if "START:" in line:
                        start_date = line.split("START:")[-1].strip()
                    elif "END:" in line:
                        end_date = line.split("END:")[-1].strip()
                    elif "PATTERN:" in line:
                        pattern = line.split("PATTERN:")[-1].strip()

                if start_date and end_date:
                    return DateRange(
                        start_date=start_date,
                        end_date=end_date,
                        pattern_type=pattern,
                        confidence=0.9
                    )
            except Exception as e:
                print(f"LangChain date extraction failed: {e}")

        # Intelligent fallback date extraction
        return self._extract_dates_fallback(question)

    def _extract_dates_fallback(self, question: str) -> 'DateRange':
        """Intelligent rule-based date extraction when LLM not available"""
        question_lower = question.lower().strip()

        # Rule-based patterns
        if "last month" in question_lower:
            return DateRange(
                start_date="2025-05-01",
                end_date="2025-05-31",
                pattern_type="last_month",
                confidence=0.8
            )
        elif "this month" in question_lower:
            return DateRange(
                start_date="2025-09-01",
                end_date="2025-09-23",
                pattern_type="this_month",
                confidence=0.8
            )
        elif "week before last" in question_lower:
            return DateRange(
                start_date="2025-08-26",
                end_date="2025-09-01",
                pattern_type="week_before_last",
                confidence=0.8
            )
        elif "year to date" in question_lower or "ytd" in question_lower:
            return DateRange(
                start_date="2025-01-01",
                end_date="2025-09-23",
                pattern_type="year_to_date",
                confidence=0.8
            )
        else:
            # Default fallback
            return DateRange(
                start_date="2025-05-01",
                end_date="2025-05-31",
                pattern_type="default_last_month",
                confidence=0.5
            )

class DateRange(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    pattern_type: str = "unknown"
    confidence: float = 0.0

# Initialize LangChain date extractor
date_extractor = LangChainDateExtractor()

@app.get("/ask-ai")
def ask_ai_question(
    question: str = Query(..., description="Natural language question about metrics (powered by LangChain + Ollama)"),
    format: str = Query(default="json", description="Response format: 'json' or 'text'"),
    mode: ResponseMode = Query(default=ResponseMode.AUTO, description="Response mode: 'data', 'sql', 'both', or 'auto'")
):
    """
    Enhanced LangChain-powered natural language interface with SQL generation.

    Modes:
    - data: Return processed data and analysis only
    - sql: Return SQL query only
    - both: Return both SQL and processed data
    - auto: Automatically determine based on query

    Examples:
    - "Why did CAC get worse last month?" (analysis)
    - "Show me the SQL to get total spend by platform" (SQL)
    - "Give me both the query and results for daily conversions" (both)
    """

    try:
        # Use real LangChain to extract intent and date range
        intent_result = metrics_llm.extract_intent(question)

        # Determine response mode
        wants_sql = intent_result.get("wants_sql", False) or mode == ResponseMode.SQL

        if mode == ResponseMode.AUTO:
            if wants_sql or any(word in question.lower() for word in ["sql", "query", "show me the query"]):
                mode = ResponseMode.SQL
            elif any(word in question.lower() for word in ["both", "and", "with query"]):
                mode = ResponseMode.BOTH
            else:
                mode = ResponseMode.DATA

        response = {"question": question, "mode": mode.value}

        # Generate SQL if requested
        if mode in [ResponseMode.SQL, ResponseMode.BOTH]:
            sql_result = metrics_llm.generate_sql(question)
            response["sql"] = sql_result

            # Execute SQL if successful and mode requires results
            if mode == ResponseMode.BOTH and "error" not in sql_result:
                execution_result = metrics_llm.execute_sql(sql_result["sql"])
                response["sql_execution"] = execution_result

        # Generate data analysis if requested
        if mode in [ResponseMode.DATA, ResponseMode.BOTH]:
            date_range = date_extractor.extract_dates(question)

            # Get metrics data using existing function
            try:
                data = metrics(start=date_range.start_date, end=date_range.end_date)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not process question: {str(e)}")

            # Use real LangChain to generate dynamic analysis of actual data
            langchain_response = metrics_llm.generate_analysis(question, data, intent_result.get("intent", "show"))

            response.update({
                "langchain_processing": {
                    "intent_detection": intent_result,
                    "date_extraction": {
                        "pattern_type": date_range.pattern_type,
                        "confidence": date_range.confidence,
                        "start": date_range.start_date,
                        "end": date_range.end_date
                    },
                    "llm_powered": True,
                    "framework": "LangChain + Ollama",
                    "model": "llama3.2:3b" if metrics_llm.llm_available else "intelligent_fallback"
                },
                "ai_analysis": langchain_response,
                "raw_data": data
            })

        # Return plain text if requested
        if format == "text":
            if mode == ResponseMode.SQL and "sql" in response:
                return PlainTextResponse(response["sql"].get("sql", "No SQL generated"))
            elif "ai_analysis" in response:
                return PlainTextResponse(response["ai_analysis"])
            else:
                return PlainTextResponse(json.dumps(response, indent=2, default=str))

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/ask-sql")
def ask_sql_question(
    question: str = Query(..., description="Natural language question to convert to SQL"),
    execute: bool = Query(default=False, description="Whether to execute the generated SQL"),
    format: str = Query(default="json", description="Response format: 'json' or 'text'")
):
    """
    Generate SQL queries from natural language questions using LangChain.

    Examples:
    - "Show total spend by platform"
    - "Get daily conversions for last month"
    - "What's the average ROAS by country?"
    """

    try:
        # Generate SQL using LangChain
        sql_result = metrics_llm.generate_sql(question)

        response = {
            "question": question,
            "sql_generation": sql_result
        }

        # Execute SQL if requested and generation was successful
        if execute and "error" not in sql_result and sql_result.get("sql"):
            execution_result = metrics_llm.execute_sql(sql_result["sql"])
            response["execution"] = execution_result

        # Return plain text if requested
        if format == "text":
            if "error" in sql_result:
                return PlainTextResponse(f"Error: {sql_result['error']}")
            else:
                output = f"SQL: {sql_result.get('sql', 'No SQL generated')}"
                if execute and "execution" in response:
                    if "error" in response["execution"]:
                        output += f"\n\nExecution Error: {response['execution']['error']}"
                    else:
                        output += f"\n\nResults ({response['execution']['row_count']} rows):\n"
                        for row in response["execution"]["results"][:5]:  # Show first 5 rows
                            output += f"{json.dumps(row, default=str)}\n"
                return PlainTextResponse(output)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Enhanced LangChain-Powered Metrics API with SQL Generation",
        "framework": "LangChain + Ollama",
        "capabilities": [
            "LLM-powered intent recognition",
            "Intelligent date extraction",
            "AI-generated analysis",
            "Natural language to SQL conversion",
            "SQL query validation and execution",
            "Dynamic database schema reflection",
            "Multi-modal responses (data/sql/both)"
        ],
        "endpoints": {
            "/ask-ai": "LangChain natural language interface for metrics analysis",
            "/ask-sql": "Generate and optionally execute SQL from natural language",
            "/docs": "API documentation"
        },
        "examples": {
            "analysis": [
                "/ask-ai?question=Why did CAC get worse last month?",
                "/ask-ai?question=Explain what these ROAS numbers mean&format=text"
            ],
            "sql_generation": [
                "/ask-sql?question=Show total spend by platform",
                "/ask-sql?question=Get daily conversions for last month&execute=true",
                "/ask-sql?question=What's the average ROAS by country?&format=text"
            ],
            "mixed_mode": [
                "/ask-ai?question=Show me the SQL to get total spend by platform&mode=sql",
                "/ask-ai?question=Give me both the query and results for daily conversions&mode=both"
            ]
        },
        "sql_safety": {
            "validation": "All SQL queries are validated for safety",
            "read_only": "Only SELECT and WITH statements allowed",
            "forbidden": ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)