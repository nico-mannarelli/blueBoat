# Detection approach: precedent, ML vs. signal processing, and the edge server

Short version: the field has largely moved to **YOLO-family CNNs**, which report
90–95%+ mAP versus the sub-70% that classical CFAR is usually quoted at on
complex bottoms. But those numbers assume a **labeled dataset**, which we don't
have. The right move for this project is the **two-stage workflow the literature
itself converges on**: keep CFAR as a label-free candidate generator, and add an
ML *classifier* on the survivors once we've accumulated labels. The Dell XR4000
is where that second stage lives.

## What comparable projects do

Side-scan target detection splits into two eras:

**Classical / signal processing.** Adaptive thresholding (CFAR), shadow analysis,
and blob/keypoint detectors (the Hough + MSER path we already have). CFAR gives a
statistical, label-free detector that runs in real time on a CPU, but it
generalizes poorly across changing seabeds and is typically cited below ~70%
detection rate in cluttered scenes. This is exactly what our README documents:
CFAR finds *anomalies*, not *object types*, and seabed clutter still trips it.

**Deep learning.** The current literature is dominated by YOLO variants tuned for
sonar (SS-YOLO, RCDI-YOLO, MAL-YOLO, DFSE-YOLO, etc.). Reported results cluster
around **mAP 0.90–0.96**, with lightweight variants (YOLOv8n) hitting **45–60 FPS
on embedded GPUs** — fast enough for onboard use. The catch is data: these are
supervised models trained on annotated wrecks/mines/debris.

**The data problem, and how people get around it.** Because annotated SSS data is
scarce, the practical literature leans on: (1) **transfer learning** from
COCO/ImageNet weights fine-tuned on a few hundred sonar chips; and (2)
**unsupervised / domain-adaptive** detection, which transfers knowledge from
labeled (often optical or other-survey) domains to unlabeled SSS and reports
**~92% AP50 / ~98% recall without annotating the target set**. These exist
precisely because nobody starts with labels — which is our situation.

**The recurring architecture.** Real-time AUV/USV systems repeatedly land on a
**two-stage pipeline**: a cheap detector proposes candidate regions, then a CNN
classifies only those crops (e.g. MobileViT pre-classifier → RepVGG+YOLOv5
detector). It's faster (the classifier never sees blank seabed) and it cleanly
separates "is something here" (label-free) from "what is it" (needs labels).

## Recommendation for this project

Don't replace CFAR — **wrap it.**

1. **Now (this is what we have):** CFAR remains the detector. It's the correct
   tool for a no-label cold start: it adapts per-pixel to the bottom, needs no
   training data, and runs real-time on CPU. The dashboard + contact log we just
   built turn its hits into a clean, de-duplicated contact list.

2. **Collect labels for free during operations.** Every contact the dashboard
   shows is a candidate crop with a georeference. Add a one-key "confirm /
   dismiss" on review (or label the revisit-scan imagery) and you accumulate a
   labeled dataset at zero extra survey cost. A few hundred confirmed chips is
   enough to fine-tune a small YOLO via transfer learning.

3. **Add stage two: an ML classifier on CFAR survivors.** Once labels exist,
   run a lightweight CNN/YOLO on each CFAR candidate crop to assign a class
   (log / rock / mine-like / clutter) and a confidence. CFAR controls recall
   (don't miss anything); the classifier controls precision (don't revisit junk).
   This is the architecture the README already anticipates and the literature
   keeps rediscovering.

4. **Only consider an end-to-end YOLO detector** (replacing CFAR) if, after you
   have a substantial labeled set, it measurably beats CFAR-recall on *your*
   bottoms. Until then a learned detector trained on someone else's seabed is a
   downgrade in generalization.

Net: ML wins on classification and precision; signal processing (CFAR) wins on
label-free recall and cold-start. Use each where it's strong.

## Using the Dell PowerEdge XR4000 edge server

The XR4000 (the GPU-capable **XR4520c** sled) takes a 250 W double-width GPU or
two 150 W single-width cards — Dell lists NVIDIA **A2 / A30** support — and is a
rugged, wide-temperature, NEBS/MIL-810H box built for exactly this kind of field
compute. That GPU is overkill for CFAR and ideal for the ML and survey stages.
Concrete jobs for it, roughly in order of payoff:

- **Stage-two classifier (primary win).** Host the CNN/YOLO that classifies CFAR
  candidate crops here. The boat's onboard CPU keeps doing real-time CFAR; crops
  (or the raw ping stream) go to the XR4000 over the boat network for
  classification, so the heavy model never has to fit on the vehicle.

- **GPU CFAR backend.** `sonar_detect.py` already ships a `cfar_backend="torch"`
  path that's numerically identical to the numpy one. Point it at the XR4000's
  GPU when you want to run CFAR at very high ping rates or over many channels /
  replayed surveys at once — the integral-image math is trivially parallel.

- **Survey mosaicking & offline reprocessing.** After a mission, stitch the
  full waterfall into a georeferenced mosaic and re-run detection at higher
  sensitivity than is affordable in real time. The GPU makes whole-survey
  reprocessing minutes instead of hours.

- **Training and retraining.** Fine-tune the classifier on newly confirmed
  contacts on the same box — collect labels in the field, retrain on the edge,
  push the updated weights back to the inference service. No cloud round-trip.

Suggested split: **boat CPU** = SonarLink ingest + CFAR + dashboard + contact
log (everything in this repo today, real-time, no GPU needed). **XR4000 GPU** =
classifier inference service, batch CFAR/mosaicking, and model training. Connect
them with the same WebSocket/REST style already used for SonarLink and
mavlink2rest, so the edge server is just another network service the pipeline
talks to.

---

### Sources

- DFSE-YOLO: deep feature selective enhancement for shipwreck detection in SSS — ResearchGate
- RCDI-YOLO (improved YOLOv8 for complex-environment SSS) — Frontiers in Marine Science, 2025
- SS-YOLO: lightweight SSS target detection — J. Marine Sci. Eng. (MDPI), 13(1):66
- YOLO Variant Evaluation & Transfer Learning for SSS — J. Marine Sci. Eng. (MDPI), 14(6):550
- Real-time underwater target detection for AUV using SSS + deep learning — Ocean Engineering (ScienceDirect)
- AUV-Based SSS Real-Time Method for Underwater-Target Detection — J. Marine Sci. Eng. (MDPI), 11(4):690
- Sparsity-Regularization Real-Time SSS Recognition on Embedded GPU — J. Marine Sci. Eng. (MDPI), 11(3):487
- Unsupervised underwater shipwreck detection via domain-adaptive techniques — Scientific Reports, 2024
- Dell PowerEdge XR4000 / XR4520c Technical Guide & Spec Sheet — Dell Technologies
