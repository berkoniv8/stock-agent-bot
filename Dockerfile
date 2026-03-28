FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir peewee beautifulsoup4 lxml

# Download NLTK data for TextBlob
RUN python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# Copy application code
COPY *.py ./
COPY watchlist.csv .
COPY portfolio.json .
COPY Makefile .

# Create directories
RUN mkdir -p logs data

# Expose health check port
EXPOSE 10000

# Run unified web server (bot + scheduler + health endpoint)
CMD ["python3", "web_server.py"]
