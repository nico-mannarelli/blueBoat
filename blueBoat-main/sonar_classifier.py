"""
sonar_classifier.py
Few-shot classification of sonar contact crops: log / rock / man_made /
background (classes come from your library folder names — nothing hardcoded).

How it works: a frozen backbone turns each 224x224 crop into an embedding
vector; each class is represented by the mean (centroid) of its labeled
examples' embeddings; a new crop is assigned to the nearest centroid by
cosine similarity, with a softmax confidence over all classes. No training,
no GPU required — a usable library needs only ~10-30 labeled crops per class.

Backbones (auto-selected, best first):
    dinov2        DINOv2 ViT-S/14 via torch.hub (384-d)  - needs torch + one
                  online session to cache the weights, offline afterwards
    resnet18      torchvision ResNet-18 penultimate layer (512-d)
    deterministic numpy/cv2 intensity+gradient features   - zero-dependency
                  fallback so the pipeline never dies in the field

A library is tied to the backbone that built it (stored in the .npz);
classify refuses a library/backbone mismatch rather than mixing spaces.

Usage:
    python sonar_classifier.py build --crops sorted_crops/ --lib library.npz
        sorted_crops/ has one subfolder per class: log/ rock/ man_made/
        background/ with the labeled PNGs inside.
    python sonar_classifier.py classify --lib library.npz img1.png img2.png
    python sonar_classifier.py selftest [--backbone deterministic]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Gates for "don't guess": below either one the answer is 'unknown'.
MIN_SIMILARITY = 0.45   # best cosine sim to any centroid (OOD gate)
MIN_CONFIDENCE = 0.40   # softmax over class sims (ambiguity gate)
SOFTMAX_TEMP   = 0.05   # cosine sims cluster tightly; sharpen before softmax
IMG_SIZE       = 224


# ---- image loading ----------------------------------------------------------

def load_gray(path_or_array):
    """Path or ndarray -> 2-D uint8 IMG_SIZE x IMG_SIZE grayscale."""
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(str(path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path_or_array}")
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


# ---- backbones --------------------------------------------------------------

def _embed_deterministic(gray):
    """Torch-free features. Deliberately translation-invariant (histograms
    and profile stats, no absolute positions) — a target must classify the
    same wherever it sits in the crop. Weak next to DINOv2 but deterministic,
    instant, and dependency-free."""
    g = gray.astype(np.float32) / 255.0
    g = (g - g.mean()) / max(g.std(), 1e-6)      # contrast-normalize
    hist = np.histogram(g, bins=32, range=(-3, 5))[0].astype(np.float32)
    hist /= max(hist.sum(), 1e-6)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)
    maghist = np.histogram(mag, bins=16, range=(0, 20))[0].astype(np.float32)
    maghist /= max(maghist.sum(), 1e-6)
    ori = np.histogram(ang, bins=16, range=(-np.pi, np.pi),
                       weights=mag)[0].astype(np.float32)
    ori /= max(ori.sum(), 1e-6)
    # highlight/shadow coverage + how concentrated each is per row/column
    bright = (g > 2.0).astype(np.float32)
    dark = (g < -1.2).astype(np.float32)
    blobs = np.array([bright.mean() * 50, dark.mean() * 50,
                      bright.sum(axis=1).std() / 10,
                      bright.sum(axis=0).std() / 10,
                      dark.sum(axis=1).std() / 10,
                      dark.sum(axis=0).std() / 10], dtype=np.float32)
    e = np.concatenate([hist * 2, maghist, ori, blobs])
    return e / max(np.linalg.norm(e), 1e-9)


def _torch_preprocess(gray):
    import torch
    g = gray.astype(np.float32) / 255.0
    x = np.stack([g, g, g])                      # gray -> 3 channels
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy((x - mean) / std).unsqueeze(0)


def _make_dinov2():
    import torch
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    def embed(gray):
        with torch.no_grad():
            e = model(_torch_preprocess(gray).to(device))[0].cpu().numpy()
        return e / max(np.linalg.norm(e), 1e-9)
    return embed


def _make_resnet18():
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    def embed(gray):
        with torch.no_grad():
            e = model(_torch_preprocess(gray).to(device))[0].cpu().numpy()
        return e / max(np.linalg.norm(e), 1e-9)
    return embed


def get_backbone(name="auto"):
    """Return (resolved_name, embed_fn). 'auto' falls through the chain so a
    missing torch install degrades the classifier instead of killing it."""
    if name in ("auto", "dinov2"):
        try:
            return "dinov2", _make_dinov2()
        except Exception as e:
            if name == "dinov2":
                raise
            print(f"[classifier] dinov2 unavailable ({e.__class__.__name__}) "
                  "- trying resnet18")
    if name in ("auto", "resnet18"):
        try:
            return "resnet18", _make_resnet18()
        except Exception as e:
            if name == "resnet18":
                raise
            print(f"[classifier] resnet18 unavailable ({e.__class__.__name__})"
                  " - using deterministic features")
    return "deterministic", _embed_deterministic


# ---- library build / classify ----------------------------------------------

def build_library(crops_dir, lib_path, backbone="auto"):
    """Embed every labeled crop; save one L2-normalized centroid per class."""
    name, embed = get_backbone(backbone)
    classes, centroids, counts = [], [], []
    for cls in sorted(os.listdir(crops_dir)):
        cls_dir = os.path.join(crops_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        vecs = [embed(load_gray(os.path.join(cls_dir, f)))
                for f in sorted(os.listdir(cls_dir))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not vecs:
            continue
        c = np.mean(vecs, axis=0)
        classes.append(cls)
        centroids.append(c / max(np.linalg.norm(c), 1e-9))
        counts.append(len(vecs))
    if len(classes) < 2:
        raise SystemExit(f"need >= 2 class folders with images under "
                         f"{crops_dir}, found {classes}")
    np.savez(lib_path, centroids=np.array(centroids, dtype=np.float32),
             classes=np.array(classes), backbone=name)
    print(f"[classifier] library {lib_path}: backbone={name}, " +
          ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
    return classes


class Classifier:
    def __init__(self, lib_path, backbone="auto"):
        lib = np.load(lib_path, allow_pickle=False)
        self.centroids = lib["centroids"]
        self.classes = [str(c) for c in lib["classes"]]
        self.lib_backbone = str(lib["backbone"])
        name, self.embed = get_backbone(
            self.lib_backbone if backbone == "auto" else backbone)
        if name != self.lib_backbone:
            raise SystemExit(
                f"library was built with '{self.lib_backbone}' but the "
                f"'{name}' backbone loaded - rebuild the library or fix "
                "the environment; mixing embedding spaces gives garbage")

    def classify(self, img):
        e = self.embed(load_gray(img))
        sims = self.centroids @ e
        z = np.exp((sims - sims.max()) / SOFTMAX_TEMP)
        conf = z / z.sum()
        i = int(np.argmax(sims))
        label = self.classes[i]
        if sims[i] < MIN_SIMILARITY or conf[i] < MIN_CONFIDENCE:
            label = "unknown"
        return {"label": label, "best_class": self.classes[i],
                "similarity": round(float(sims[i]), 4),
                "confidence": round(float(conf[i]), 4)}


# ---- selftest ----------------------------------------------------------------

def _synthetic(cls, rng):
    """Crude sonar-ish textures, one recipe per fake class."""
    img = rng.normal(90, 12, (IMG_SIZE, IMG_SIZE)).clip(0, 255)
    if cls == "ridge":       # bright along-track ridge + shadow (log-like)
        y = IMG_SIZE // 2 + int(rng.integers(-30, 30))
        img[y - 6:y + 6, 30:200] += 110
        img[y + 8:y + 30, 30:200] -= 55
    elif cls == "mound":     # bright roundish blob + shadow (rock-like)
        c = (int(rng.integers(70, 150)), int(rng.integers(70, 150)))
        cv2.circle(img, c, 22, 200, -1)
        cv2.ellipse(img, (c[0], c[1] + 34), (22, 12), 0, 0, 360, 40, -1)
    # 'seafloor' = plain noise
    return img.clip(0, 255).astype(np.uint8)


def selftest(backbone="deterministic", tmp="_clf_selftest"):
    import shutil
    rng = np.random.default_rng(7)
    classes = ["ridge", "mound", "seafloor"]
    shutil.rmtree(tmp, ignore_errors=True)
    for cls in classes:
        d = os.path.join(tmp, cls)
        os.makedirs(d)
        for i in range(8):
            cv2.imwrite(os.path.join(d, f"{i}.png"), _synthetic(cls, rng))
    lib = os.path.join(tmp, "library.npz")
    build_library(tmp, lib, backbone=backbone)

    clf = Classifier(lib, backbone=backbone)
    ok = total = 0
    for cls in classes:
        for _ in range(5):                       # unseen samples
            r = clf.classify(_synthetic(cls, rng))
            ok += (r["best_class"] == cls)
            total += 1
    acc = ok / total
    print(f"[selftest] backbone={backbone}  held-out accuracy {ok}/{total}"
          f" = {acc:.0%}")
    shutil.rmtree(tmp, ignore_errors=True)
    if acc < 0.8:
        print("[selftest] FAIL (< 80%)")
        return 1
    print("[selftest] PASS")
    return 0


# ---- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="few-shot sonar contact classifier")
    ap.add_argument("--backbone", default="auto",
                    choices=["auto", "dinov2", "resnet18", "deterministic"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build");    b.add_argument("--crops", required=True)
    b.add_argument("--lib", default="library.npz")
    c = sub.add_parser("classify"); c.add_argument("--lib", default="library.npz")
    c.add_argument("images", nargs="+")
    sub.add_parser("selftest")
    args = ap.parse_args()

    if args.cmd == "build":
        build_library(args.crops, args.lib, backbone=args.backbone)
    elif args.cmd == "classify":
        clf = Classifier(args.lib, backbone=args.backbone)
        for img in args.images:
            print(json.dumps({"image": img, **clf.classify(img)}))
    elif args.cmd == "selftest":
        bb = "deterministic" if args.backbone == "auto" else args.backbone
        sys.exit(selftest(backbone=bb))


if __name__ == "__main__":
    main()
