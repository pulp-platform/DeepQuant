FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip setuptools wheel && \
    pip install toml-to-requirements && \
    toml-to-req --toml-file pyproject.toml && \
    pip install -r requirements.txt
