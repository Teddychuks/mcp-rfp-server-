FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VIRTUALENVS_CREATE=false

ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --without dev

RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN chmod +x startup.sh

RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence_transformers
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app

USER app

EXPOSE 8000

CMD ["./startup.sh"]