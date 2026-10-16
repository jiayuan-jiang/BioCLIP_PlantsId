"""
读官方 txt_emb_species.npy 的指定列，不下载整个文件 (Botanical Garden pipeline · step 72)
=======================================================================================
官方 npy 形状 (768, 867455)，C 顺序（按列=按物种存），2.7GB。读一个物种 = 对 CDN 发 768 个 4 字节 Range 请求。
用途：拿官方原值当标准，核对线上词表 / 调试文字向量复现。
核对结果（2026-10-10）：好行与线上余弦 1.000000；零行官方也是 0（零向量是上游缺口）。
线上词表行 k ↔ 官方 json 第 516084+k 条（Plantae 连续段；200 抽样对齐）。
需要先下载 txt_emb_species.json（91MB，同目录 embeddings/）到 S 路径。
"""
U="https://huggingface.co/datasets/imageomics/TreeOfLife-200M/resolve/main/embeddings/txt_emb_species.npy"
class NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**k): return None
def resolve():
    try: urllib.request.build_opener(NoRedir).open(urllib.request.Request(U,method="HEAD"))
    except urllib.error.HTTPError as e: return e.headers["Location"]
URL=resolve()
def rng(a,b):
    r=urllib.request.Request(URL,headers={"Range":f"bytes={a}-{b}"}); return urllib.request.urlopen(r,timeout=60).read()
h=rng(0,255); hl=struct.unpack("<H",h[8:10])[0]; off=10+hl; hdr=h[10:10+hl].decode(); print(hdr.strip())
D,N=768,867455
def col(j):
    def one(d):
        a=off+(d*N+j)*4; return np.frombuffer(rng(a,a+3),dtype="<f4")[0]
    with cf.ThreadPoolExecutor(32) as ex: return np.array(list(ex.map(one,range(D))),dtype=np.float32)
d=json.load(open(S+"/txt_emb_species.json"))
z=np.load("bioclip_full_index.npz",allow_pickle=True); E=z["embeddings"]; path=z["taxon_paths"]
base=516084
# 对齐检查：我们的行 k ↔ json 的 Plantae 第 k 个？
bad=0
for k in random.Random(0).sample(range(len(path)),200):
    if " ".join(d[base+k][0])!=str(path[k]): bad+=1
print("行对齐不一致数(200抽样):",bad)
ok=np.where(np.linalg.norm(E,axis=1)>0.5)[0]; zero=np.where(np.linalg.norm(E,axis=1)<=0.5)[0]
pick=[("好行",k) for k in random.Random(1).sample(list(ok),4)]+[("零行",k) for k in random.Random(1).sample(list(zero),3)]
out={}
for tag,k in pick:
    v=col(base+k); n=np.linalg.norm(v)
    cs=float(v@E[k]/(n*np.linalg.norm(E[k])+1e-12)) if n>0 else float("nan")
    print(f"{tag} {path[k]!s:80.80}  官方范数 {n:.4f} 线上范数 {np.linalg.norm(E[k]):.4f} 余弦 {cs:.6f}  俗名={d[base+k][1]!r}",flush=True)
    out[int(k)]=v.tolist()
json.dump(out,open(S+"/official_cols.json","w"))
