FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fontconfig fonts-dejavu-core curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/fonts/truetype/montserrat \
 && (curl -fsSL -o /usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf || true) \
 && fc-cache -f
RUN pip install --no-cache-dir fastapi==0.143.2 uvicorn==0.54.0 faster-whisper==1.2.1 httpx==0.28.1 numpy==2.5.4

WORKDIR /app
COPY app/ /app/
ENV DATA_DIR=/data HF_HOME=/models CPU_THREADS=2 RETENTION_DAYS=14
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
