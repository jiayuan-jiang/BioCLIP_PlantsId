"""用 Pl@ntNet-300K 真 organ 标签验证：
 A) zero-shot 部位分类：raw argmax vs image-mean 校准
 B) linear probe (LogisticRegression on frozen 768-d emb)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch, open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "organ_eval.npz")
W = os.path.normpath(os.path.join(HERE, "..", "..", "demo", "weights", "open_clip_model.safetensors"))
ORG = ["leaf", "flower", "fruit", "bark", "habit"]

d = np.load(NPZ, allow_pickle=True)
emb, organs = d["emb"], d["organs"].astype(str)
y = np.array([ORG.index(o) for o in organs])
print("eval set:", {o: int((organs == o).sum()) for o in ORG}, " dim", emb.shape)

# ---------- A) zero-shot ----------
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=W)
model.eval()
tok = open_clip.get_tokenizer("ViT-L-14")

@torch.no_grad()
def proto(ps):
    e = model.encode_text(tok(ps)).float()
    e = (e / e.norm(dim=-1, keepdim=True)).numpy().mean(0)
    return e / np.linalg.norm(e)

OP = {
 "leaf":  ["a photo of a leaf", "a close-up photo of the leaves of a plant", "plant foliage, leaves"],
 "flower":["a photo of a flower", "a close-up photo of a flower", "a plant in bloom with flowers"],
 "fruit": ["a photo of a fruit", "a close-up photo of berries and fruit", "the fruit and seeds of a plant"],
 "bark":  ["a photo of tree bark", "a close-up photo of the bark of a tree trunk", "the trunk of a tree"],
 "habit": ["a photo of a whole plant", "a photo of a whole tree outdoors", "a shrub growing in a landscape"],
}
P = np.stack([proto(OP[o]) for o in ORG])

def report(tag, pred):
    acc = (pred == y).mean()
    bacc = balanced_accuracy_score(y, pred)
    print(f"\n--- {tag}: acc={acc:.3f}  balanced_acc={bacc:.3f} ---")
    print(classification_report(y, pred, target_names=ORG, digits=3, zero_division=0))
    print("confusion (rows=true):")
    print("      " + " ".join(f"{o[:5]:>6}" for o in ORG))
    for i, o in enumerate(ORG):
        print(f"{o:5} " + " ".join(f"{v:6d}" for v in confusion_matrix(y, pred, labels=range(5))[i]))

report("zero-shot RAW", (emb @ P.T).argmax(1))
mu = emb.mean(0)
ec = emb - mu
ec = ec / np.linalg.norm(ec, axis=1, keepdims=True)
report("zero-shot CENTERED (mean subtracted)", (ec @ P.T).argmax(1))

# ---------- B) linear probe ----------
print("\n\n==================  LINEAR PROBE (5-fold CV)  ==================")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof = np.zeros_like(y)
for tr, te in skf.split(emb, y):
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(emb[tr], y[tr])
    oof[te] = clf.predict(emb[te])
report("linear probe (out-of-fold)", oof)
