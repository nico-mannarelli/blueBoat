"""
sonar_display.py
Cosmetic rendering for the live waterfall windows — purely visual, with no
effect on detection. Detection runs on the dB data in sonar_detect.py; this
just makes the picture look like a real sonar display rather than flat
grayscale.

SonarView's "nice" look is three cheap operations on the display image:
  1. gamma < 1   — lift the mid-tones so faint seabed texture shows
  2. a sonar LUT — the classic amber/bronze ramp (black→brown→amber→white)
  3. upscaling   — smooth bilinear render

The grayscale image fed in is expected to be "strong return = bright"
(WaterfallDetector._build_display_image), matching SonarView's convention.
"""

import numpy as np
import cv2


def _ramp_lut(stops):
    """Build a smooth 256-entry BGR LUT by linearly interpolating through a list
    of (position, (R,G,B)) colour stops. Interpolating between hand-placed
    anchors gives a controlled, even gradient — no abrupt hue shift and no
    power-curve crush — which is what makes SonarView's colourised waterfall
    read as smooth rather than posterised."""
    pos = np.array([s[0] for s in stops], dtype=np.float32)
    cols = np.array([s[1] for s in stops], dtype=np.float32)   # R,G,B rows
    t = np.linspace(0.0, 1.0, 256)
    r = np.interp(t, pos, cols[:, 0])
    g = np.interp(t, pos, cols[:, 1])
    b = np.interp(t, pos, cols[:, 2])
    return np.stack([b, g, r], axis=1).clip(0, 255).astype(np.uint8).reshape(256, 1, 3)


def _bronze_lut():
    """Amber/bronze ramp: black -> deep brown -> amber -> warm white."""
    return _ramp_lut([
        (0.00, (0,   0,   0)),
        (0.18, (40,  18,  6)),
        (0.42, (120, 60,  18)),
        (0.68, (205, 130, 40)),
        (0.86, (245, 200, 120)),
        (1.00, (255, 250, 235)),
    ])


def _blue_lut():
    """Ocean-blue ramp: black -> navy -> teal -> cyan -> white. Strong returns
    whiten so targets stay clearly visible against the blue field."""
    return _ramp_lut([
        (0.00, (0,   0,   0)),
        (0.16, (6,   16,  48)),
        (0.40, (10,  60,  120)),
        (0.64, (20,  140, 190)),
        (0.84, (140, 220, 235)),
        (1.00, (250, 252, 255)),
    ])


_LUTS = {"amber": _bronze_lut(), "blue": _blue_lut()}   # "gray" → no LUT


def _apply_tone(gray, brightness=0.0, contrast=1.0, gamma=0.65):
    """SonarView-style tonal controls on a grayscale image, in float for no
    banding: contrast pivots around mid-grey, brightness shifts, gamma<1 lifts
    mid-tones. Returns uint8."""
    x = gray.astype(np.float32) / 255.0
    if contrast != 1.0:
        x = (x - 0.5) * contrast + 0.5
    if brightness:
        x = x + brightness
    x = np.clip(x, 0.0, 1.0)
    if gamma and gamma != 1.0:
        x = np.power(x, gamma)
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def enhance_waterfall(gray):
    """SonarView-style display enhancement (prototyped on the 2026-06-16 scan;
    raw-vs-enhanced comparisons in enhance_preview/). Order matters:

    1. flat-field   — normalize each range column by its typical intensity,
                      removing near-nadir-bright / far-range-dark banding
                      (same idea the CFAR detector uses internally)
    2. 3-ping along-track average — SonarView's smoothness is largely ping
                      averaging; vertical-only so range resolution is untouched
    3. bilateral    — edge-preserving speckle removal
    4. mild CLAHE   — local contrast, blended 60/40 with input so it never
                      looks over-processed

    Display path ONLY — classifier crops keep raw pixel statistics (the
    93% real-data baseline was measured on unenhanced crops).
    """
    f = gray.astype(np.float32)
    med = np.median(f, axis=0)
    med = cv2.GaussianBlur(med[None, :], (31, 1), 0)[0]
    x = (f * (np.median(med) / np.maximum(med, 1.0))).clip(0, 255).astype(np.uint8)
    x = cv2.blur(x, (1, 3))
    x = cv2.bilateralFilter(x, d=5, sigmaColor=35, sigmaSpace=5)
    e = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(16, 16)).apply(x)
    return cv2.addWeighted(e, 0.6, x, 0.4, 0)


def colorize(gray, palette="amber", gamma=0.85, scale=1,
             brightness=0.0, contrast=1.12):
    """Map a grayscale waterfall (strong=bright) to a BGR sonar display image.

    palette    : "amber" (default sonar look), "blue", or "gray"
    gamma      : <1 brightens mid-tones; 1.0 disables
    brightness : additive lift in [-1,1] units (0 = none)
    contrast   : multiplicative gain around mid-grey (1.0 = none). SonarView's
                 default look is a touch above 1; lower it to reveal weak signal.
    scale      : upscale factor for a larger window (1 = native). Non-integer is
                 fine; uses bicubic up / area down for a smooth, artefact-free
                 render rather than blocky nearest-neighbour.
    """
    g = _apply_tone(gray, brightness=brightness, contrast=contrast, gamma=gamma)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    lut = _LUTS.get(palette)
    if lut is not None:
        bgr = cv2.LUT(bgr, lut)
    if scale and scale != 1:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        bgr = cv2.resize(bgr, (int(round(bgr.shape[1] * scale)),
                               int(round(bgr.shape[0] * scale))),
                         interpolation=interp)
    return bgr
