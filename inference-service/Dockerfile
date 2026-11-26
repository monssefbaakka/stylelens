# Contexte de build : racine du dépôt (le service réutilise ml/stylelens).
#   docker build -f inference-service/Dockerfile -t stylelens-inference .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    TORCH_HOME=/cache/torch

WORKDIR /app

# PyTorch CPU (~200 Mo au lieu de ~2 Go pour la build CUDA)
RUN pip install "torch>=2.5" "torchvision>=0.20" --index-url https://download.pytorch.org/whl/cpu
COPY inference-service/requirements.txt .
RUN pip install -r requirements.txt

COPY ml/stylelens ./stylelens
COPY inference-service/app ./app

RUN useradd --create-home --uid 1000 app && mkdir -p /cache/torch && chown app /cache/torch
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
