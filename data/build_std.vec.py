import pandas as pd, numpy as np, torch, open_clip, math, os
from tqdm import tqdm

import os
CSV         = './taxonomy_enriched.csv'
print(os.path.abspath('./'))
WEIGHTS_DIR = 'weights/bioclip2'
ARCH        = 'ViT-L-14'
OUTPUT      = 'my_plant_index.npz'
BATCH       = 256
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv(CSV, dtype={'taxon_id': str})
print(f'读取 {len(df)} 行')

weight_file = None
for c in ['open_clip_pytorch_model.bin', 'model.safetensors']:
    p = os.path.join(WEIGHTS_DIR, c)
    if os.path.isfile(p):
        weight_file = p
        break

if weight_file:
    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=weight_file)
    tokenizer   = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
else:
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    tokenizer   = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

model = model.to(DEVICE).eval()

RANK_COLS = ['kingdom','phylum','class_','order','family','genus','scientific_name']
texts = []
for _, row in df.iterrows():
    parts = [str(row.get(c,'') or '').strip() for c in RANK_COLS if str(row.get(c,'') or '').strip()]
    texts.append(' '.join(parts))

print('示例：')
for t in texts[:3]:
    print(' ', t)

all_embs = []
with torch.no_grad():
    for i in tqdm(range(math.ceil(len(texts)/BATCH)), desc='编码'):
        batch  = texts[i*BATCH:(i+1)*BATCH]
        tokens = tokenizer(batch).to(DEVICE)
        embs   = model.encode_text(tokens)
        embs   = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().float().numpy())

embeddings = np.concatenate(all_embs, axis=0)
np.savez_compressed(
    OUTPUT,
    embeddings   = embeddings.astype(np.float32),
    taxon_ids    = df['taxon_id'].fillna('').astype(str).values,
    sci_names    = df['scientific_name'].fillna('').astype(str).values,
    common_names = df['common_name'].fillna('').astype(str).values if 'common_name' in df.columns else np.array(['']*len(df)),
    taxon_paths  = np.array(texts, dtype=object),
)
print(f'完成：{OUTPUT}  shape={embeddings.shape}')
