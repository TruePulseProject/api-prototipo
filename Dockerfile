FROM python:3.11-slim

# ffmpeg: normalizacao para 30fps | libgl/libglib: dependencias do opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# storage/ e models/ devem ser volumes: o checkpoint nao vai na imagem e os
# videos biometricos nao devem sobreviver ao container.
VOLUME ["/srv/storage", "/srv/models"]
EXPOSE 8000

# --workers 1 de proposito: o JobStore vive em memoria (ver README).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
