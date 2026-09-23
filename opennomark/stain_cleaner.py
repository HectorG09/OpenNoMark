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
  3. Per region, a robust masked Gaussian field. A region is flat only if
     nearly all of its pixels sit near that field (``FLAT_COVERAGE``), and
     it is a designed fill only if the field itself is near-constant
     (``FILL_RANGE``). Smooth photo areas (sky, skin) are shaded and fail the
     second test; images with too little fill area (``MIN_FILL_SHARE``) are
     returned untouched, so photographs pass through.
  4. Soft blend inside accepted regions: full replacement below
     ``TOLERANCE``, fading to none at ``2 * TOLERANCE``. A hard cut draws
     contour lines through real gradients.
  5. Text band: JPEG ringing beside glyphs is strong enough to count as
     structure, so this band is where the most visible blotches live. The
     field of the nearest large region (``ANCHOR_AREA``) is extended
     smoothly ``BAND_RADIUS`` pixels into it; small gaps between glyphs are
     band too, because their own fields are noisy and copying the single
     nearest value drew streaks and stepped patches. Each band pixel is
     modelled as a blend of background and the nearest ink (two-colour text
     model) and projected onto that blend; only mostly-background pixels
     (``MAX_INK_ALPHA``) are rebuilt, so glyph bodies and their anti-aliasing
     profile are kept. Pixels lighter than the fill are stain.

Untouched pixels are copied from the input bit for bit. Distances are
Euclidean in OpenCV's 8-bit Lab scale, computed in floating point: an 8-bit
Lab round trip alone moves half the pixels of a JPEG by up to 15 levels.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Per-pixel Lab gradient (Scharr / 16) that marks structure. Measured blotch
# gradients on real ChatGPT samples stay below ~5; anti-aliased text edges
# exceed 20.
EDGE_THRESHOLD = 7.0
# Field smoothness. Most blotches are 8-16px but some reach ~25px; sigma 22
# averages those out (sigma 14 left a visible disk in the reference sample)
# while still following the 100-200px vignettes ChatGPT paints inside boxes.
FIELD_SIGMA = 22.0
# Lab distance treated as stain rather than content. Measured blotch p99 is
# ~8 on the reference sample, so 7 with a soft fade to 14 covers it.
TOLERANCE = 7.0
MIN_REGION_AREA = 24
# Flat: FLAT_COVERAGE of the pixels within TOLERANCE of the field and
# FLAT_COVERAGE_WIDE within twice that. Saturated fills carry stronger
# blotches: the reference green "i" badge has 94% within 7 but 99% within 14.
FLAT_COVERAGE = 0.90
FLAT_COVERAGE_WIDE = 0.99
# A designed fill is near-constant: the 98th percentile distance of its field
# from the fill's median colour stays within FILL_RANGE. ChatGPT boxes measure
# 1-2 (vignette included); shaded photo areas exceed it.
FILL_RANGE = 8.0
# Images whose large fills cover less than this share are not flat graphics
# and are returned untouched. ChatGPT infographics measure ~70%; photos from
# examples/ at most ~35%. Cleaning photos flattened cloud texture into patches.
MIN_FILL_SHARE = 0.45
# Ringing beside bold text fuses whole lines into structure; 16px reaches the
# middle of those blocks on 1K ChatGPT output.
BAND_RADIUS = 16
# Only regions this large define a background; smaller accepted regions
# (gaps between letters and lines) take the extension of their nearest anchor,
# computed by normalised convolution with ``EXTEND_SIGMA``.
ANCHOR_AREA = 400
EXTEND_SIGMA = 8.0
# A smaller fill anchors itself when its colour differs from the surrounding
# background by DISTINCT_FILL: badges and circles ("SI", "NO"), whose white
# text leaves only fragments of the fill. Gaps between glyphs share the box
# colour and stay in the band.
DISTINCT_FILL = 20.0
# Two-colour text model in the band. Ink is the colour farthest from the
# background within ``INK_RADIUS``; it must differ by ``INK_MIN`` to define a
# text/background line. Band pixels without ink nearby, or lighter than the
# fill, are background: stains up to ``BAND_TOLERANCE`` are removed, fading to
# none at 1.4x. Pixels farther than ``LINE_TOLERANCE`` from the line are a
# third colour (icons, borders) and are left alone.
INK_RADIUS = 2
INK_MIN = 20.0
MAX_INK_ALPHA = 0.35
LINE_TOLERANCE = 12.0
BAND_TOLERANCE = 20.0
# Mean Lab deviation inside flat regions below which there is nothing to
# clean. A synthetic clean graphic measures 0.0; the reference ChatGPT
# sample measures above 1.
MIN_STAIN_ENERGY = 0.25
# The residual check requires at least this reduction of stain energy.
MAX_RESIDUAL_RATIO = 0.5


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab[..., 0] *= 255.0 / 100.0
    lab[..., 1:] += 128.0
    return lab


def _to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255.0
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


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

    Returns ``(field, covered, deviations, accepted, fill_range)`` where
    ``deviations`` holds each covered pixel's Lab distance to its field and
    ``fill_range`` maps an accepted label to the colour range of its field.
    ``only`` restricts the pass to already-accepted labels, which lets the
    residual check re-measure exactly the regions that were cleaned.
    """
    height, width = labels.shape
    field = lab.copy()
    covered = np.zeros(labels.shape, bool)
    deviations = np.zeros(labels.shape, np.float32)
    accepted = []
    fill_range = {}
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
        if only is None:
            region_error = error[inside]
            if (
                (region_error < TOLERANCE).mean() < FLAT_COVERAGE
                or (region_error < 2 * TOLERANCE).mean() < FLAT_COVERAGE_WIDE
            ):
                continue
            colours = estimate[inside]
            spread = np.sqrt(((colours - np.median(colours, axis=0)) ** 2).sum(1))
            fill_range[label] = float(np.percentile(spread, 98))

        field[y0:y1, x0:x1][inside] = estimate[inside]
        deviations[y0:y1, x0:x1][inside] = error[inside]
        covered[y0:y1, x0:x1] |= inside
        accepted.append(label)
    return field, covered, deviations, accepted, fill_range


def _distinct_fills(field, background, labels, stats, candidates):
    """Small fills whose colour differs from the background around them."""
    distinct = []
    for label in candidates:
        x, y, box_w, box_h, _ = stats[label]
        inside = labels[y:y + box_h, x:x + box_w] == label
        own = field[y:y + box_h, x:x + box_w][inside].mean(axis=0)
        around = background[y:y + box_h, x:x + box_w][inside].mean(axis=0)
        if np.sqrt(((own - around) ** 2).sum()) >= DISTINCT_FILL:
            distinct.append(label)
    return distinct


def _extend_background(field, labels, stats, anchors):
    """Background for pixels near anchor regions, extended from the nearest one."""
    height, width = labels.shape
    anchor_mask = np.isin(labels, anchors)
    # DIST_LABEL_PIXEL numbers the zero pixels in row-major order, which is
    # the order np.nonzero returns them in.
    _, nearest = cv2.distanceTransformWithLabels(
        (~anchor_mask).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(anchor_mask)
    region_of = np.zeros(len(ys) + 1, np.int32)
    region_of[1:] = labels[ys, xs]
    nearest_region = region_of[nearest]
    distance = cv2.distanceTransform((~anchor_mask).astype(np.uint8), cv2.DIST_L2, 3)
    reach = distance <= BAND_RADIUS

    background = field.copy()
    pad = BAND_RADIUS + 1
    for label in anchors:
        x, y, box_w, box_h, _ = stats[label]
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + box_w + pad, width), min(y + box_h + pad, height)
        inside = labels[y0:y1, x0:x1] == label
        targets = (nearest_region[y0:y1, x0:x1] == label) & reach[y0:y1, x0:x1] & ~inside
        if targets.any():
            extended = _masked_blur(field[y0:y1, x0:x1], inside.astype(np.float32), EXTEND_SIGMA)
            background[y0:y1, x0:x1][targets] = extended[targets]
    return background, anchor_mask, distance


def _neighbourhood_ink(lab: np.ndarray, background: np.ndarray):
    """Return the colour farthest from each pixel's background nearby, and that distance."""
    height, width = background.shape[:2]
    r = INK_RADIUS
    padded = cv2.copyMakeBorder(lab, r, r, r, r, cv2.BORDER_REPLICATE)
    best = np.zeros((height, width), np.float32)
    ink = lab.copy()
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            candidate = padded[dy:dy + height, dx:dx + width]
            distance = ((candidate - background) ** 2).sum(2)
            farther = distance > best
            best[farther] = distance[farther]
            ink[farther] = candidate[farther]
    return ink, np.sqrt(best)


def _band_targets(lab, background, band):
    """Two-colour text model for the band; returns ``(target, weight)``."""
    ink, ink_distance = _neighbourhood_ink(lab, background)
    direction = ink - background
    alpha = ((lab - background) * direction).sum(2) / np.maximum((direction ** 2).sum(2), 1e-6)
    clamped = np.clip(alpha, 0.0, 1.0)
    off_line = np.sqrt(((lab - background - clamped[..., None] * direction) ** 2).sum(2))
    # Taper faint ink toward zero so the lightest stains beside glyphs vanish
    # instead of surviving as a 5-10% ink tint.
    tapered = np.where(clamped < 0.12, clamped * clamped / 0.12, clamped)
    target = background + tapered[..., None] * direction

    error = np.sqrt(((lab - background) ** 2).sum(2))
    as_background = band & ((ink_distance < INK_MIN) | (alpha <= 0.0))
    background_weight = np.clip((1.4 * BAND_TOLERANCE - error) / (0.4 * BAND_TOLERANCE), 0.0, 1.0)
    on_line = (
        band & ~as_background & (alpha < MAX_INK_ALPHA) & (off_line < LINE_TOLERANCE)
    )
    target = np.where(as_background[..., None], background, target)
    weight = np.where(as_background, background_weight, on_line.astype(np.float32))
    return target, weight


def _stain_energy(deviations: np.ndarray, covered: np.ndarray) -> float:
    return float(deviations[covered].mean()) if covered.any() else 0.0


def clean_stains(image: Image.Image) -> tuple[Image.Image, dict]:
    """Return ``(result, report)``; ``result`` is ``image`` when untouched."""
    rgb = np.asarray(image.convert("RGB"))
    lab = _to_lab(rgb)
    structure = _structure_mask(lab)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - structure, connectivity=4)
    field, covered, deviations, accepted, fill_range = _region_fields(lab, labels, stats)
    fills = [label for label in accepted if fill_range[label] <= FILL_RANGE]
    large = [label for label in fills if stats[label][4] >= ANCHOR_AREA]
    fill_mask = np.isin(labels, fills)

    report = {
        "covered_fraction": round(float(covered.mean()), 4),
        "fill_share": round(float(stats[large, 4].sum() / labels.size) if large else 0.0, 4),
        "regions": len(fills),
        "stain_energy_before": round(_stain_energy(deviations, fill_mask), 4),
    }
    if (
        report["fill_share"] < MIN_FILL_SHARE
        or report["stain_energy_before"] < MIN_STAIN_ENERGY
    ):
        report.update(changed_fraction=0.0, stain_energy_after=report["stain_energy_before"])
        return image, report

    background, _, _ = _extend_background(field, labels, stats, large)
    small = [label for label in fills if stats[label][4] < ANCHOR_AREA]
    anchors = large + _distinct_fills(field, background, labels, stats, small)
    background, anchored, distance = _extend_background(field, labels, stats, anchors)
    band = ~anchored & (distance <= BAND_RADIUS)

    error = np.sqrt(((lab - background) ** 2).sum(2))
    target, weight = _band_targets(lab, background, band)
    flat_weight = np.clip((2 * TOLERANCE - error) / TOLERANCE, 0.0, 1.0)
    target[anchored] = background[anchored]
    weight[anchored] = flat_weight[anchored]

    changed = weight > 0
    cleaned_rgb = _to_rgb(lab + (target - lab) * weight[..., None])
    output = rgb.copy()
    output[changed] = cleaned_rgb[changed]
    result = Image.fromarray(output)

    _, after_covered, after_deviations, _, _ = _region_fields(
        _to_lab(output), labels, stats, only=fills
    )
    report["changed_fraction"] = round(float(changed.mean()), 4)
    report["stain_energy_after"] = round(_stain_energy(after_deviations, after_covered), 4)
    return result, report
