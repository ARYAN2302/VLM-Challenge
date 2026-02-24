FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0 \
    pydantic==2.10.5 \
    python-multipart==0.0.20 \
    transformers==4.49.0 \
    accelerate==1.2.1 \
    decord==0.6.0 \
    torchvision==0.16.2 \
    bitsandbytes==0.45.2 \
    qwen-vl-utils==0.0.8

COPY main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
