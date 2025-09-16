FROM n8nio/n8n:latest

USER root

# Python, pip y runtime C++ necesario para duckdb
RUN apk add --no-cache python3 py3-pip libstdc++

# Crear venv y poner duckdb ahí (sin tocar el Python del sistema)
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir duckdb==1.1.0

# Asegurar que el venv esté en PATH para n8n/Execute Command
ENV PATH="/opt/venv/bin:${PATH}"

USER node
