FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DROWSINESS_DEVICE=cpu \
    DROWSINESS_CHECKPOINT_PATH=saved_weights/best_loss_v3.pt \
    DROWSINESS_HOST=0.0.0.0 \
    DROWSINESS_PORT=8001

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api
COPY model /app/model
COPY pre_process /app/pre_process
COPY dataloader.py device.py inference.py loss.py metrics.py train.py /app/
COPY saved_weights/best_loss_v3.pt /app/saved_weights/best_loss_v3.pt

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001"]
