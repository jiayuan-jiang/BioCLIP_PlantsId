FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 安装 Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 先装 CUDA 版 torch；requirements.txt 里 torch>=2.2.0 已满足，pip 不会覆盖
RUN pip3 install --no-cache-dir \
    "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 代码 & 静态资源
COPY inference_core.py inference.py handler.py ./
COPY static/ ./static/

# 模型权重 & 向量库（体积较大，单独层便于缓存）
COPY weights/ ./weights/
COPY bioclip_full_index.npz .

EXPOSE 8000

CMD ["python3.11", "handler.py"]
