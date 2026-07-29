# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgl1 / libglib2.0-0 required by some streamlit image-related deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "streamlit>=1.35" "httpx>=0.27" "requests>=2.31"

WORKDIR /app

COPY ui/requirements.txt ./ui/requirements.txt
RUN pip install -r ui/requirements.txt

COPY ui/ ./ui/

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
