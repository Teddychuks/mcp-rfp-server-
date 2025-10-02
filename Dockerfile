FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VIRTUALENVS_CREATE=false

ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Install system dependencies and SSL certificates
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies (including aiohttp for SharePoint)
RUN poetry install --no-root --without dev

# Download spaCy model and SentenceTransformer model
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Install certifi for proper SSL handling
RUN pip install certifi

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Create cache directories and app user
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence_transformers \
    && addgroup --system app \
    && adduser --system --group app \
    && chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Start the application
CMD ["./startup.sh"]