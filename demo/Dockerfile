FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# torch 已内置于基础镜像，pip 安装其余依赖时会直接跳过 torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 代码 & 静态资源
COPY inference_core.py inference.py handler.py ./
COPY static/ ./static/

# 模型权重 & 向量库（体积较大，单独层便于缓存）
COPY weights/ ./weights/
COPY bioclip_full_index.npz .

EXPOSE 8000

CMD ["python", "handler.py"]
