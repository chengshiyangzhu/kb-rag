# syntax=docker/dockerfile:1.7

# ============ Builder stage ============
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System build dependencies (git for some installs, build-essential for native
# wheels, libgl1 for OCR runtime libs that leak into build).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install --prefix=/install .

# ============ Runtime stage ============
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime system packages: libgl1 / libglib2.0-0 needed by rapidocr-onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder.
COPY --from=builder /install /usr/local

WORKDIR /app

# Application code.
COPY app/ ./app/
COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
