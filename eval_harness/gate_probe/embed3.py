"""v3：补"通用网图风格"植物正样本（用户实测 v1/v2 在商图/网图/窗外取景上漏判）。三个源，每源 full+cc60+rrc 三视图：
  oxford_flowers(网图花) · plant_village(叶片特写) · sun397_plant(森林/花园/果园/树等真实场景)。"""
import numpy as np, torch, open_clip
from concurrent.futures import ThreadPoolExecutor
import embed as E1, embed2 as E2

PLANT_CLASSES = {"bamboo forest", "botanical garden", "corn field", "cottage garden", "field cultivated", "field wild",
                 "forest broadleaf", "forest needleleaf", "forest path", "forest road", "formal garden", "greenhouse indoor",
                 "greenhouse outdoor", "hayfield", "herb garden", "nursery", "orchard", "rainforest", "topiary garden",
                 "tree farm", "vegetable garden", "vineyard", "wheat field", "swamp", "marsh"}
SRC = {
    "oxford_flowers": E1.hf("nelorth/oxford-flowers", 1500, "pos"),
    "plant_village":  E1.hf("DScomp380/plant_village", 1200, "pos"),
    "sun397_plant":   E1.hf("tanganke/sun397", 2000, "pos", labelcol="label",
                            exclude=lambda y, n: n[y].lower() not in PLANT_CLASSES, buf=5000),
}
if __name__ == "__main__":
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(E1.WEIGHTS))
    model = model.to(E1.DEVICE).eval(); pool = ThreadPoolExecutor(6)
    for name, spec in SRC.items():
        try: E2.run(name, spec["gen"](), ["full", "cc60", "rrc"], model, pre, pool, n=spec["n"])
        except Exception as e: print("[FAIL]", name, type(e).__name__, str(e)[:200], flush=True)
