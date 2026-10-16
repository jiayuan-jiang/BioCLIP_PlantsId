"""本地对比页：拖一张图 → 同时返回线上 energy 方案与二元头(probe)的判定。 python compare_app.py → http://127.0.0.1:8765"""
import io
from pathlib import Path
import numpy as np, torch, open_clip, uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[2]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ENERGY_THR = 68.537  # 线上 gate v1
head = np.load(Path(__file__).parent / "gate_head.npz")
W, B, PROBE_THR = head["w"], float(head["b"]), float(head["threshold"])
h2 = np.load(Path(__file__).parent / "gate_head_v2.npz")
h3 = np.load(Path(__file__).parent / "gate_head_v3.npz")
W3, B3, THR3 = h3["w"], float(h3["b"]), float(h3["threshold"])
W2, B2, THR2 = h2["w"], float(h2["b"]), float(np.load(Path(__file__).parent / "gate_head_v2_r995.npz")["threshold"])  # recall 99.5% 阈值

model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(ROOT / "data/weights/bioclip2/open_clip_model.safetensors"))
model = model.to(DEV).eval()
LS = model.logit_scale.exp().item()
raw = np.load(ROOT / "data/bioclip_full_index.npz", allow_pickle=True)
IDX = raw["embeddings"].astype(np.float32); NAMES = raw["sci_names"]
print("logit_scale", LS, "index", IDX.shape, flush=True)
app = FastAPI()

@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        e = model.encode_image(pre(img).unsqueeze(0).to(DEV)).float()
        e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    sims = IDX @ e
    z = LS * sims; m = z.max(); energy = float(m + np.log(np.exp(z - m).sum()))
    ps = float(e @ W + B)
    ps2 = float(e @ W2 + B2)
    ps3 = float(e @ W3 + B3)
    top = np.argsort(sims)[::-1][:3]
    return {"energy": {"score": round(energy, 3), "threshold": ENERGY_THR, "is_plant": energy >= ENERGY_THR},
            "probe": {"score": round(ps, 3), "threshold": round(PROBE_THR, 3), "is_plant": ps >= PROBE_THR,
                      "prob": round(1 / (1 + np.exp(-ps)), 4)},
            "probe_v2": {"score": round(ps2, 3), "threshold": round(THR2, 3), "is_plant": ps2 >= THR2, "prob": round(1 / (1 + np.exp(-ps2)), 4)},
            "probe_v3": {"score": round(ps3, 3), "threshold": round(THR3, 3), "is_plant": ps3 >= THR3, "prob": round(1 / (1 + np.exp(-ps3)), 4)},
            "top3": [{"name": str(NAMES[i]), "sim": round(float(sims[i]), 4)} for i in top]}

PAGE = """<!doctype html><meta charset=utf-8><title>gate compare</title>
<style>body{font:15px system-ui;max-width:900px;margin:30px auto;padding:0 16px}
#dz{border:2px dashed #888;border-radius:12px;padding:40px;text-align:center;color:#666}
#dz.on{background:#eef}.row{display:flex;gap:14px;margin:12px 0;align-items:flex-start}
img{max-width:220px;max-height:180px;border-radius:6px}.card{flex:1;padding:12px;border-radius:8px;border:1px solid #ccc}
.y{background:#e6f6e6}.n{background:#fde8e8}table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}</style>
<h2>植物 gate：energy vs 二元头</h2><div id=dz>把图片拖到这里（可一次拖多张），或点击选择<input id=f type=file accept="image/*" multiple hidden></div><div id=out></div>
<script>
const dz=document.getElementById('dz'),f=document.getElementById('f'),out=document.getElementById('out');
dz.onclick=()=>f.click();f.onchange=()=>go(f.files);
['dragenter','dragover'].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.add('on')}));
['dragleave','drop'].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.remove('on')}));
dz.addEventListener('drop',e=>go(e.dataTransfer.files));
document.addEventListener('dragover',e=>e.preventDefault());document.addEventListener('drop',e=>{e.preventDefault();if(!dz.contains(e.target))go(e.dataTransfer.files)});
async function go(files){for(const file of files){const row=document.createElement('div');row.className='row';
row.innerHTML='<img src="'+URL.createObjectURL(file)+'"><div class=card>处理中…</div>';out.prepend(row);
const fd=new FormData();fd.append('file',file);
try{const r=await (await fetch('/compare',{method:'POST',body:fd})).json();
const c=(t,o,extra)=>'<div class="card '+(o.is_plant?'y':'n')+'"><b>'+t+'</b>：'+(o.is_plant?'植物 ✓':'非植物 ✗')+'<br>score '+o.score+' / 阈值 '+o.threshold+(extra||'')+'</div>';
row.querySelector('.card').outerHTML=c('energy（线上）',r.energy)+c('二元头 v1',r.probe,'<br>p='+r.probe.prob)+c('二元头 v2（多样性+裁切）',r.probe_v2,'<br>p='+r.probe_v2.prob)+c('二元头 v3（+网图/场景）',r.probe_v3,'<br>p='+r.probe_v3.prob)+
'<div class=card><b>top3 物种</b><table>'+r.top3.map(t=>'<tr><td>'+t.name+'</td><td>'+t.sim+'</td></tr>').join('')+'</table></div>';
}catch(e){row.querySelector('.card').textContent='出错：'+e}}}
</script>"""
@app.get("/", response_class=HTMLResponse)
def home(): return PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
