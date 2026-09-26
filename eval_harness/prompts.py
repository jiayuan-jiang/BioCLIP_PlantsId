"""Zero-shot 文本提示模板。

- plant / species：物种识别基准（Pl@ntNet-300K、Rare Species、iNat 等）
- imagenet：通用留存基准（OpenAI 80 模板的精简子集）
- bird：NABirds（tree-of-life 非植物留存）

render_class() 对单个类别渲染出一组 prompt 字符串，编码后取均值即该类的文本原型。
"""

PLANT_TEMPLATES = [
    "a photo of {name}.",
    "a photo of {name}, a species of plant.",
    "a close-up photo of {name}.",
    "a photo of the plant {name}.",
]

# 有俗名时额外补充
PLANT_COMMON_TEMPLATES = [
    "a photo of {common}.",
    "a photo of {common}, a plant.",
]

BIRD_TEMPLATES = [
    "a photo of a {name}, a type of bird.",
    "a photo of the {name}.",
]

# OpenAI ImageNet 模板子集（取 7 条，兼顾速度与稳定）
IMAGENET_TEMPLATES = [
    "a photo of a {name}.",
    "a bad photo of a {name}.",
    "a photo of many {name}.",
    "a close-up photo of a {name}.",
    "a bright photo of a {name}.",
    "a photo of the {name}.",
    "itap of a {name}.",
]

_REGISTRY = {
    "plant": PLANT_TEMPLATES,
    "species": PLANT_TEMPLATES,
    "bird": BIRD_TEMPLATES,
    "imagenet": IMAGENET_TEMPLATES,
    "general": IMAGENET_TEMPLATES,
}


def render_class(name: str, template_set: str = "plant", common: str | None = None) -> list[str]:
    tpls = list(_REGISTRY.get(template_set, PLANT_TEMPLATES))
    out = [t.format(name=name) for t in tpls]
    if common and template_set in ("plant", "species"):
        out += [t.format(common=common) for t in PLANT_COMMON_TEMPLATES]
    return out
