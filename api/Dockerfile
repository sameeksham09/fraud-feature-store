FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── System deps ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ─────────────────────────────────────────────────────────
# Use the single root requirements.txt as source of truth.
# api/requirements.txt is kept for standalone installs but the image uses root.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── App code ────────────────────────────────────────────────────────────
COPY api ./api
COPY features ./features
COPY models ./models
# mlruns is NOT baked in — mount it at runtime so host training is visible:
#   volumes: ["./mlruns:/app/mlruns:ro"]
# If no mlruns exists yet, the API starts in degraded mode (503 on /score) until you `make train`.

EXPOSE 8000

# Env defaults for Docker network (override with compose or -e)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
