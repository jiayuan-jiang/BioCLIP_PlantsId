# BioCLIP 2 Demo

本地植物 / 动物图像识别服务，基于 BioCLIP 2 模型 + 全量物种向量索引。

---

## 文件结构

```
demo/
├── inference.py              # FastAPI 服务主文件
├── requirements.txt          # Python 依赖
├── CLAUDE.md                 # 本文件
├── bioclip_full_index.npz    # 全量物种向量索引（~1GB，需手动复制）
├── weights/                  # 模型权重（需手动复制）
│   ├── open_clip_pytorch_model.bin
│   ├── open_clip_model.safetensors
│   └── ...（tokenizer 等配套文件）
└── static/
    └── index.html            # Web UI（拖拽上传 + 结果展示）
```

> `bioclip_full_index.npz` 和 `weights/` 体积较大，不纳入 git，需在本机手动准备。

---

## 首次部署

### 1. 复制数据文件

从项目根目录执行：

```bash
cp data/bioclip_full_index.npz demo/
cp -r data/weights/bioclip2/   demo/weights
```

### 2. 安装依赖

```bash
pip install -r demo/requirements.txt
```

### 3. 启动服务

```bash
cd demo
uvicorn inference:app --host 0.0.0.0 --port 8000
```

服务就绪后终端会打印：

```
物种数：N,NNN  维度：768  元字段：[...]
✓ 就绪  device=cpu
INFO  Uvicorn running on http://0.0.0.0:8000
```

首次启动需加载约 1GB 索引 + 模型权重，约需 30–60 秒。

---

## 使用方式

### Web UI

浏览器打开 `http://localhost:8000`，拖拽或点击上传图片，选择返回数量（Top-K），点击**识别**。

### API

**POST** `/predict?topk=5`

```bash
curl -X POST http://localhost:8000/predict?topk=5 \
     -F "file=@your_image.jpg"
```

**响应示例：**

```json
{
  "results": [
    {
      "rank": 1,
      "similarity": 0.42318,
      "similarity_pct": 42.32,
      "confidence": "high",
      "sci_names": "Acer rubrum",
      "common_names": "Red Maple",
      "taxon_paths": "Plantae > ... > Acer > Acer rubrum",
      "genera": "Acer",
      "families": "Sapindaceae",
      "kingdoms": "Plantae",
      "taxon_ids": "47727"
    }
  ]
}
```

**GET** `/health` — 健康检查，返回设备类型和已加载物种数。

---

## 置信度说明

| 置信度 | 相似度阈值 | 建议 |
|--------|-----------|------|
| 高 (high)   | ≥ 35% | 结果可信 |
| 中 (medium) | 22–35% | 仅供参考，建议结合属级信息判断 |
| 低 (low)    | < 22% | 建议换清晰图片或局部特写（叶片 / 花 / 果实）|

---

## 模型说明

- **模型**：BioCLIP 2（`imageomics/bioclip-2`），架构 ViT-L-14
- **索引**：`bioclip_full_index.npz`，包含全量物种的文本嵌入向量及分类学元数据
- **推理**：图像编码后与向量库做余弦相似度检索，返回 Top-K 物种

权重优先从本地 `weights/` 加载；若目录不存在，自动从 HuggingFace 下载（需联网）。
