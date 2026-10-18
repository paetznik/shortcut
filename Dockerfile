FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fontconfig fonts-dejavu-core curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/fonts/truetype/montserrat \
 && (curl -fsSL -o /usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf || true) \
 && fc-cache -f
RUN pip install --no-cache-dir fastapi==0.143.2 uvicorn==0.54.0 faster-whisper==1.2.1 httpx==0.28.1 numpy==2.5.4 \
    opencv-python-headless==4.14.0.94
# PyTorch (CPU) for the word-exact forced aligner (torchaudio MMS_FA; its model is downloaded on first use)
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
# small model files: YuNet face detector (OpenCV model zoo) and an RNNoise speech denoising model
RUN mkdir -p /opt/models \
 && curl -fsSL -o /opt/models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
 && curl -fsSL -o /opt/models/sh.rnnn https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn

WORKDIR /app
COPY app/ /app/
ENV DATA_DIR=/data HF_HOME=/models TORCH_HOME=/models/torch CPU_THREADS=2 RETENTION_DAYS=14
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
