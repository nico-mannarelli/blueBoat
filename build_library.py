"""
build_library.py — build library.npz from hand-sorted example crops.

DATA IN:  a folder with one subfolder per GROUP of visually similar
          labeled crops. Folder name before '__' is the class, so
          log__c02/ and log__c09/ are two varieties of class 'log'
          (a plain log/ folder works too: one variety).
DATA OUT: library.npz (one centroid per folder, class names repeat)
          for sonar_classifier.py

Keeping varieties separate matters: averaging visually different groups
into one centroid blurs it until nothing matches.

Run whenever the sorted examples change, then copy library.npz to
wherever classification runs:
    python build_library.py --crops sorted_crops/ --lib library.npz
"""

import argparse
import os

import numpy as np

from sonar_classifier import embed, load_gray


def build_library(crops_dir, lib_path):
    """Fingerprint every image under crops_dir/<folder>/, average each
    folder into one centroid (class = folder name before '__'), save
    centroids + class names to lib_path (.npz)."""
    classes, centroids, counts = [], [], []
    for folder in sorted(os.listdir(crops_dir)):
        cls_dir = os.path.join(crops_dir, folder)
        if not os.path.isdir(cls_dir):
            continue
        vecs = [embed(load_gray(os.path.join(cls_dir, f)))
                for f in sorted(os.listdir(cls_dir))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not vecs:
            continue
        c = np.mean(vecs, axis=0)                  # the group average
        classes.append(folder.split("__")[0])
        centroids.append(c / max(np.linalg.norm(c), 1e-9))
        counts.append(len(vecs))
    if len(set(classes)) < 2:
        raise SystemExit(f"need >= 2 classes with images under "
                         f"{crops_dir}, found {sorted(set(classes))}")
    np.savez(lib_path, centroids=np.array(centroids, dtype=np.float32),
             classes=np.array(classes))
    print(f"[classifier] library {lib_path}: " +
          ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
    return classes


def main():
    ap = argparse.ArgumentParser(
        description="build library.npz from sorted example crops")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--lib", default="library.npz")
    args = ap.parse_args()
    build_library(args.crops, args.lib)


if __name__ == "__main__":
    main()
