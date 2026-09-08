FROM python:3.10-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-lite.txt /tmp/requirements-lite.txt
RUN python -m pip install --no-cache-dir \
      --index-url "${PYTORCH_CPU_INDEX_URL}" \
      torch==2.5.1+cpu torchvision==0.20.1+cpu && \
    python -m pip install --no-cache-dir \
      --index-url "${PIP_INDEX_URL}" \
      -r /tmp/requirements-lite.txt
COPY app.py models.py skeleton.py ./
COPY weights/stroke_embedding.pt ./weights/stroke_embedding.pt

EXPOSE 8899
CMD ["python", "app.py", "--weights", "/app/weights/stroke_embedding.pt", "--host", "0.0.0.0", "--port", "8899"]
