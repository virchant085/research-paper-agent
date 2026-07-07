# Research Paper Agent — single image for both the FastAPI API and the Streamlit UI.
# pymupdf, chromadb and litellm all ship manylinux wheels, so no apt build deps
# are required; the slim base stays slim.
FROM python:3.11-slim

# Faster, cleaner container Python: no .pyc files, unbuffered stdout/stderr.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first so this layer is cached across source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# Storage location for SQLite, Chroma, uploads and exports (see backend.config).
RUN mkdir -p /app/data

EXPOSE 8000

# Default: run the API. docker-compose overrides `command` for the UI service.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
