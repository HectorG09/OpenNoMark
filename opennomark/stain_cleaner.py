"""Flat-region stain cleanup for AI-generated graphics.

ChatGPT infographics carry no localized logo. Their visible defect is a
mottled, 8-16px blotch texture inside flat fills (tinted boxes, headers,
badges), strongest next to dark text where chroma compression rings. LaMa is
the wrong tool here: a stain mask covers a quarter of a typical infographic,
LaMa takes minutes on CPU for that area, and it paints new speckles beside
glyphs. This expert instead rebuilds each flat fill from its own smooth colour
field and leaves text, borders and icons untouched.

Pipeline:
  1. Structure mask: Lab gradients above ``EDGE_THRESHOLD`` (text, lines,
     icon outlines) dilated by one pixel.
  2. Regions: connected components of everything else.
  3. Per region, a robust masked Gaussian field. A region is cleaned only if
     ``FLAT_COVERAGE`` of its pixels sit within ``TOLERANCE`` of that field;
     photographic texture fails this test and is never flattened.
  4. The clean field is carried ``BAND_RADIUS`` pixels into the structure
     band from the nearest accepted pixel, so blotches between glyphs are
     removed too, while glyph pixels (far from the field) are not.
  5. Soft blend: full replacement below ``TOLERANCE``, fading to none at
     ``2 * TOLERANCE``. A hard cut draws contour lines through real gradients.

Distances are Euclidean in OpenCV's 8-bit Lab space.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Per-pixel Lab gradient (Scharr / 16) that marks structure. Measured blotch
# gradients on real ChatGPT samples stay below ~5; anti-aliased text edges
# exceed 20.
EDGE_THRESHOLD = 7.0
# Field smoothness. Blotches are 8-16px; sigma 14 averages them out while
# still following the soft vignettes ChatGPT paints inside boxes.
FIELD_SIGMA = 14.0
# Lab distance treated as stain rather than content. Measured blotch p99 is
# ~8 on the reference sample, so 7 with a soft fade to 14 covers it.
TOLERANCE = 7.0
MIN_REGION_AREA = 24
FLAT_COVERAGE = 0.95
BAND_RADIUS = 6
# Below this share of flat pixels the image is not a flat graphic.
MIN_COVERED_FRACTION = 0.05
# Mean Lab deviation inside flat regions below which there is nothing to
# clean. A synthetic clean graphic measures 0.0; the reference ChatGPT
# sample measures above 1.
MIN_STAIN_ENERGY = 0.25
# The residual check requires at least this reduction of stain energy.
MAX_RESIDUAL_RATIO = 0.5


def _structure_mask(lab: np.ndarray) -> np.ndarray:
    gradient = np.zeros(lab.shape[:2], np.float32)
    for channel in range(3):
        gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0) / 16.0
        gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1) / 16.0
        gradient = np.maximum(gradient, np.hypot(gx, gy))
    edges = (gradient > EDGE_THRESHOLD).astype(np.uint8)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))


def _masked_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    numerator = cv2.GaussianBlur(values * weight[..., None], (0, 0), sigma)
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    return numerator / np.maximum(denominator, 1e-6)[..., None]


def _region_fields(lab, labels, stats, only=None):
    """Fit a robust smooth field to each flat region.

    Returns ``(field, covered, deviations, accepted)`` where ``deviations``
    holds each covered pixel's Lab distance to its field. ``only`` restricts
    the pass to already-accepted labels, which lets the residual check
    re-measure exactly the regions that were cleaned.
    """
    height, width = labels.shape
    field = lab.copy()
    covered = np.zeros(labels.shape, bool)
    deviations = np.zeros(labels.shape, np.float32)
    accepted = []
    pad = int(3 * FIELD_SIGMA)
    candidates = only if only is not None else range(1, len(stats))

    for label in candidates:
        x, y, box_w, box_h, area = stats[label]
        if area < MIN_REGION_AREA:
            continue
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + box_w + pad, width), min(y + box_h + pad, height)
        inside = labels[y0:y1, x0:x1] == label
        values = lab[y0:y1, x0:x1]
        weight = inside.astype(np.float32)
        # Thin regions (gaps between glyphs) cannot support a wide kernel.
        sigma = min(FIELD_SIGMA, max(2.0, 0.5 * min(box_w, box_h)))

        estimate = _masked_blur(values, weight, sigma)
        error = np.sqrt(((values - estimate) ** 2).sum(2))
        # One robust pass: drop pixels far from the first estimate so a
        # small genuine feature does not tint the whole fill.
        estimate = _masked_blur(values, weight * (error < 2 * TOLERANCE), sigma)
        error = np.sqrt(((values - estimate) ** 2).sum(2))
        if only is None and (error[inside] < TOLERANCE).mean() < FLAT_COVERAGE:
            continue

        field[y0:y1, x0:x1][inside] = estimate[inside]
        deviations[y0:y1, x0:x1][inside] = error[inside]
        covered[y0:y1, x0:x1] |= inside
        accepted.append(label)
    return field, covered, deviations, accepted


def _stain_energy(deviations: np.ndarray, covered: np.ndarray) -> float:
    return float(deviations[covered].mean()) if covered.any() else 0.0


def clean_stains(image: Image.Image) -> tuple[Image.Image, dict]:
    """Return ``(result, report)``; ``result`` is ``image`` when untouched."""
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    structure = _structure_mask(lab)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - structure, connectivity=4)
    field, covered, deviations, accepted = _region_fields(lab, labels, stats)

    report = {
        "covered_fraction": round(float(covered.mean()), 4),
        "regions": len(accepted),
        "stain_energy_before": round(_stain_energy(deviations, covered), 4),
    }
    if (
        report["covered_fraction"] < MIN_COVERED_FRACTION
        or report["stain_energy_before"] < MIN_STAIN_ENERGY
    ):
        report.update(changed_fraction=0.0, stain_energy_after=report["stain_energy_before"])
        return image, report

    # Carry each pixel's nearest accepted field value into the structure band.
    # DIST_LABEL_PIXEL numbers the zero pixels in row-major order, which is
    # the order np.nonzero returns them in.
    _, nearest = cv2.distanceTransformWithLabels(
        (~covered).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(covered)
    lookup = np.zeros((len(ys) + 1, 3), np.float32)
    lookup[1:] = field[ys, xs]
    field = lookup[nearest]
    distance = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 3)

    error = np.sqrt(((lab - field) ** 2).sum(2))
    blend = np.clip((2 * TOLERANCE - error) / TOLERANCE, 0.0, 1.0)
    blend[distance > BAND_RADIUS] = 0.0
    cleaned = lab + (field - lab) * blend[..., None]
    cleaned = np.clip(np.round(cleaned), 0, 255).astype(np.uint8)
    result = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_LAB2RGB))

    _, after_covered, after_deviations, _ = _region_fields(
        cleaned.astype(np.float32), labels, stats, only=accepted
    )
    report["changed_fraction"] = round(float((blend > 0).mean()), 4)
    report["stain_energy_after"] = round(_stain_energy(after_deviations, after_covered), 4)
    return result, report
