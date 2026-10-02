# Stage 1: Builder stage to install system build tools and Python dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install system build dependencies required for FAISS and C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final minimal production runtime image
FROM python:3.13-slim AS runner

WORKDIR /app

# Install runtime shared libraries (libgomp for FAISS CPU execution)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code and directory structures
COPY src/ ./src/
COPY utils/ ./utils/
COPY app.py .
COPY api.py .
COPY pytest.ini .
COPY README.md .

# Create data directories for FAISS index and PDF uploads
RUN mkdir -p data/raw data/index

# Expose Streamlit default web server port
EXPOSE 8501

# Environment flags for unbuffered logs & Streamlit telemetry disable
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check to monitor Streamlit app availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
