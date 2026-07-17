"""
DATA IN:  a 224x224 grayscale crop (file path or ndarray)
DATA OUT: {label, best_class, similarity, confidence} 

Quick check from a terminal:
    python sonar_classifier.py some_image.png
"""

import sys
import cv2
import numpy as np


MIN_SIMILARITY = 0.65   # below this vs the best centroid -> 'unknown'
SOFTMAX_TEMP   = 0.05   # sharpens the confidence math; leave it alone
IMG_SIZE       = 224    # every crop is normalized to this size

_MODEL = None   # the DINOv2 model, loaded once on first use, then reused

# ---- entry -------------------------------------------------------------

# normalization, greyscaling, cropping etc...
def load_gray(path_or_array):
    """File path or ndarray in -> clean 224x224 uint8 grayscale out."""
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


# ---- embedding -----------------------------------------------------------------

def embed(gray):
    """Grayscale image in -> unit-length 384-dim DINOv2 fingerprint out.
    Loads the (frozen) model once on first call, reuses it after."""
    global _MODEL
    import torch
    if _MODEL is None:
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-small") # FOR SEAN: not sure which model you downloaded but can swap here
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = (m.eval().to(device), device)
    model, device = _MODEL

    g = gray.astype(np.float32) / 255.0
    x = np.stack([g, g, g])                       # grayscale -> 3 channels
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    t = torch.from_numpy((x - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():                          # frozen: never learns
        e = model(pixel_values=t).last_hidden_state[0, 0].cpu().numpy()
    return e / max(np.linalg.norm(e), 1e-9)


# ---- classification ------------------------------------------------------------

class Classifier:
    def __init__(self, lib_path):
        """Load the class centroids from library.npz."""
        lib = np.load(lib_path, allow_pickle=False)
        self.centroids = lib["centroids"]
        self.classes = [str(c) for c in lib["classes"]]

    def classify(self, img):
        """Fingerprint img, return the class of the most similar centroid —
        or "unknown" when nothing clears the similarity gate. A class may
        have several centroids (one per visual variety of that class);
        confidence is summed over the winning class's centroids."""
        e = embed(load_gray(img))
        sims = self.centroids @ e                  # similarity to each centroid
        z = np.exp((sims - sims.max()) / SOFTMAX_TEMP)
        conf = z / z.sum()
        i = int(np.argmax(sims))
        label = self.classes[i]
        class_conf = float(sum(c for c, cl in zip(conf, self.classes)
                               if cl == label))
        if sims[i] < MIN_SIMILARITY:
            label = "unknown"
        return {"label": label, "best_class": self.classes[i],
                "similarity": round(float(sims[i]), 4),
                "confidence": round(class_conf, 4)}


# ---- map annotation ------------------------------------------------------------

def annotate(png_bytes, bbox, text):
    """Draw the detection box + a filled label tag on a snapshot image.
    PNG bytes in -> PNG bytes out; bbox = (x, y, w, h) in snapshot pixels.
    On any decode problem the original bytes come back unchanged."""
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return png_bytes
    x, y, w, h = (int(v) for v in bbox)
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    color = (90, 220, 90)                      # BGR green
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # label tag: filled bar above the box (below it if the box hugs the
    # top), shifted left if it would run off the right edge
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = y - 6 if y - th - 10 >= 0 else y + h + th + 8
    tx = max(0, min(x, W - tw - 8))
    cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 8, ty + base - 2),
                  color, -1)
    cv2.putText(img, text, (tx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else png_bytes


if __name__ == "__main__":
    clf = Classifier("library.npz")
    for path in sys.argv[1:]:
        print(path, "->", clf.classify(path))
