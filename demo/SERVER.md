# BioCLIP 2 Server

**地址：** `http://195.26.240.249:8000`

---

## 启动 / 停止

```bash
# 启动（静默后台）
/opt/bioclip/server_start.sh

# 查看日志
tail -f /opt/bioclip/bioclip.log

# 停止
kill $(pgrep -f "uvicorn inference:app")
```

---

## API

### 健康检查

**GET** `/health`

```bash
curl http://195.26.240.249:8000/health
```

响应：
```json
{ "status": "ok", "device": "cpu", "species_count": 59458 }
```

---

### 物种识别

**POST** `/predict?topk=5`

```bash

```

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | multipart/form-data | 图片文件（jpg / png / webp） |
| `topk` | int（1–20）| 返回结果数量，默认 5 |

响应：
```json
{
  "results": [
    {
      "rank": 1,
      "similarity": 0.75159,
      "similarity_pct": 75.16,
      "confidence": "high",
      "sci_names": "Catharanthus roseus",
      "common_names": "Madagascar periwinkle",
      "taxon_paths": "Plantae Tracheophyta Magnoliopsida Gentianales Apocynaceae Catharanthus roseus",
      "kingdoms": "Plantae",
      "families": "Apocynaceae",
      "genera": "Catharanthus",
      "taxon_ids": null
    }
  ]
}
```


---

## Web UI

浏览器打开 `http://195.26.240.249:8000`，拖拽上传图片即可。
