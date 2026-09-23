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
  2. Regions: connected components of everything else. A region is flat
     only if nearly all of its pixels sit near a robust masked Gaussian
     field (``FLAT_COVERAGE``).
  3. Groups: text splits one fill into fragments (line gaps, letter
     counters), so small flat fragments join a nearby region of the same
     colour (``GROUP_TOLERANCE``, ``GROUP_RADIUS``) and each group gets one
     field. A group is a designed fill only if that field is near-constant
     (``FILL_RANGE``). Smooth photo areas (sky, skin) are shaded and fail
     this test; images with too little fill area (``MIN_FILL_SHARE``,
     measured at ChatGPT's resolution) are returned untouched, so
     photographs pass through.
  4. Soft blend inside fills: full replacement below ``TOLERANCE``, fading
     to none at ``2 * TOLERANCE``. A hard cut draws contour lines through
     real gradients.
  5. Text band: JPEG ringing beside glyphs is strong enough to count as
     structure, so this band is where the most visible blotches live. The
     field of the nearest large group (``ANCHOR_AREA``) on the same side of
     any hard edge (``HARD_EDGE``) is extended smoothly ``BAND_RADIUS``
     pixels into it; isolated gaps between glyphs are band too, because
     their own fields are noisy and copying the single nearest value drew
     streaks and stepped patches. Each band pixel is modelled as a blend of
     background and the nearest ink (two-colour text model) and projected
     onto that blend; only mostly-background pixels (``MAX_INK_ALPHA``) are
     rebuilt, so glyph bodies and their anti-aliasing profile are kept.
     Pixels lighter than the fill are stain.

Untouched pixels are copied from the input bit for bit. Distances are
Euclidean in OpenCV's 8-bit Lab scale, computed in floating point: an 8-bit
Lab round trip alone moves half the pixels of a JPEG by up to 15 levels.
"""

from __future__ import annotations

from typing import NamedTuple

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
# blotches: the reference green "INICIO" pill has 89% within 7 but 99.9%
# within 14.
FLAT_COVERAGE = 0.85
FLAT_COVERAGE_WIDE = 0.99
# Text splits one fill into fragments (line gaps closed by descenders, letter
# counters). Fragments within GROUP_RADIUS of each other whose colours differ
# by at most GROUP_TOLERANCE share one field. On the reference diamond the gap
# between two lines sat 21px from the body, beyond the band, and kept its
# blotches; its colour differed by 5, blotchy counters by up to 8.4.
GROUP_TOLERANCE = 12.0
GROUP_RADIUS = 32
# A designed fill is near-constant: the 98th percentile distance of its field
# from the fill's median colour stays within FILL_RANGE. ChatGPT boxes measure
# 1-2 (vignette included); shaded photo areas exceed it.
FILL_RANGE = 8.0
# Images whose large fills cover less than this share are not flat graphics
# and are returned untouched. ChatGPT infographics measure ~70%; photos from
# examples/ at most 45% (a Gemini thumbnail with a flat dark backdrop).
# Cleaning photos flattened cloud texture into patches.
MIN_FILL_SHARE = 0.50
# Every pixel constant here is calibrated on ChatGPT's native output, 1024 or
# 1536px on the long side. Larger images are measured for the photo guard on
# a copy scaled to that size: at their native 2400px, two Gemini photos from
# examples/ measured 0.46 and 0.49 and would have been flattened.
GUARD_SIZE = 1536
# Ringing beside bold text fuses whole lines into structure; 16px reaches the
# middle of those blocks on 1K ChatGPT output.
BAND_RADIUS = 16
# Gradients above HARD_EDGE are design edges (badge rims, box outlines,
# glyphs); they split the image into rooms. A band pixel takes the background
# of a group in its own room before a nearer one across such an edge: inside
# the reference "SI" badge the white page was nearer than the badge's own
# fragments, and the rim kept its blotches. Blotches inside saturated fills
# reach 13, the badge rim ~57.
HARD_EDGE = 20.0
# Only groups this large define a background; smaller isolated fills (gaps
# between letters) take the extension of their nearest anchor, computed by
# normalised convolution with ``EXTEND_SIGMA``.
ANCHOR_AREA = 400
EXTEND_SIGMA = 8.0
# A smaller group anchors itself when its colour differs from the surrounding
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


def _gradient(lab: np.ndarray) -> np.ndarray:
    gradient = np.zeros(lab.shape[:2], np.float32)
    for channel in range(3):
        gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0) / 16.0
        gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1) / 16.0
        gradient = np.maximum(gradient, np.hypot(gx, gy))
    return gradient


def _structure_mask(gradient: np.ndarray) -> np.ndarray:
    edges = (gradient > EDGE_THRESHOLD).astype(np.uint8)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))


def _masked_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    numerator = cv2.GaussianBlur(values * weight[..., None], (0, 0), sigma)
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    return numerator / np.maximum(denominator, 1e-6)[..., None]


def _bounds(stats, members, pad, shape):
    """Bounding box of a group of labels grown by ``pad``, clipped to ``shape``."""
    x, y, box_w, box_h = (stats[members, i] for i in range(4))
    return (
        max(int(x.min()) - pad, 0),
        max(int(y.min()) - pad, 0),
        min(int((x + box_w).max()) + pad, shape[1]),
        min(int((y + box_h).max()) + pad, shape[0]),
    )


def _flat_regions(lab, labels, stats):
    """Labels of the regions that are flat around a robust smooth field."""
    height, width = labels.shape
    accepted = []
    pad = int(3 * FIELD_SIGMA)
    for label in range(1, len(stats)):
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
        error = np.sqrt(((values - estimate) ** 2).sum(2))[inside]
        if (
            (error < TOLERANCE).mean() >= FLAT_COVERAGE
            and (error < 2 * TOLERANCE).mean() >= FLAT_COVERAGE_WIDE
        ):
            accepted.append(label)
    return accepted


def _group_regions(lab, labels, stats, candidates):
    """Merge fragments of one fill that text split apart; returns label arrays."""
    if not candidates:
        return []
    candidates = np.asarray(candidates)
    count = len(stats)
    colour = np.stack(
        [np.bincount(labels.ravel(), lab[..., c].ravel(), count) for c in range(3)], 1
    ) / np.maximum(stats[:, 4], 1)[:, None]
    area = stats[:, 4].astype(np.float64)

    # Only a fragment below ANCHOR_AREA joins a group, so pairs start from
    # those. Every candidate within GROUP_RADIUS counts, not just the nearest
    # region: flat stroke interiors of bold text sit between a line gap and
    # its box.
    is_candidate = np.zeros(count, bool)
    is_candidate[candidates] = True
    pairs = [np.zeros((0, 2), np.int64)]
    for label in candidates[stats[candidates, 4] < ANCHOR_AREA]:
        x0, y0, x1, y1 = _bounds(stats, [label], GROUP_RADIUS, labels.shape)
        local = labels[y0:y1, x0:x1]
        reach = cv2.distanceTransform((local != label).astype(np.uint8), cv2.DIST_L2, 3)
        others = np.unique(local[reach <= GROUP_RADIUS])
        others = others[is_candidate[others] & (others != label)]
        pairs.append(np.stack([np.full(len(others), label), others], 1))
    pairs = np.unique(np.sort(np.concatenate(pairs), 1), axis=0)
    gaps = np.sqrt(((colour[pairs[:, 0]] - colour[pairs[:, 1]]) ** 2).sum(1))

    # Closest colours merge first, and each merge is checked against the
    # group's running colour so a chain of small steps cannot join distinct
    # fills. Two groups that each hold an ANCHOR_AREA region never merge:
    # large regions carry their own field, and pale boxes sit within
    # GROUP_TOLERANCE of the white page around them.
    parent = np.arange(count)
    anchored = area >= ANCHOR_AREA

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for a, b in pairs[np.argsort(gaps)]:
        root_a, root_b = find(a), find(b)
        if root_a == root_b or (anchored[root_a] and anchored[root_b]):
            continue
        if np.sqrt(((colour[root_a] - colour[root_b]) ** 2).sum()) > GROUP_TOLERANCE:
            continue
        if area[root_a] < area[root_b]:
            root_a, root_b = root_b, root_a
        total = area[root_a] + area[root_b]
        colour[root_a] = (colour[root_a] * area[root_a] + colour[root_b] * area[root_b]) / total
        area[root_a] = total
        anchored[root_a] |= anchored[root_b]
        parent[root_b] = root_a

    roots = np.array([find(label) for label in candidates])
    return [candidates[roots == root] for root in np.unique(roots)]


def _group_fields(lab, labels, stats, groups):
    """Fit one robust smooth field to each group of regions.

    Returns ``(field, covered, deviations, spread)`` where ``deviations``
    holds each covered pixel's Lab distance to its field and ``spread`` the
    colour range of each group's field (98th percentile distance from its
    median). Also re-measures a cleaned image for the residual check.
    """
    field = lab.copy()
    covered = np.zeros(labels.shape, bool)
    deviations = np.zeros(labels.shape, np.float32)
    spread = []
    pad = int(3 * FIELD_SIGMA)
    for members in groups:
        x0, y0, x1, y1 = _bounds(stats, members, pad, labels.shape)
        inside = np.isin(labels[y0:y1, x0:x1], members)
        values = lab[y0:y1, x0:x1]
        weight = inside.astype(np.float32)

        estimate = _masked_blur(values, weight, FIELD_SIGMA)
        error = np.sqrt(((values - estimate) ** 2).sum(2))
        estimate = _masked_blur(values, weight * (error < 2 * TOLERANCE), FIELD_SIGMA)
        error = np.sqrt(((values - estimate) ** 2).sum(2))

        colours = estimate[inside]
        distance = np.sqrt(((colours - np.median(colours, axis=0)) ** 2).sum(1))
        spread.append(float(np.percentile(distance, 98)))
        field[y0:y1, x0:x1][inside] = colours
        deviations[y0:y1, x0:x1][inside] = error[inside]
        covered[y0:y1, x0:x1] |= inside
    return field, covered, deviations, spread


def _distinct_fills(field, background, labels, stats, groups):
    """Small groups whose colour differs from the background around them."""
    distinct = []
    for members in groups:
        x0, y0, x1, y1 = _bounds(stats, members, 0, labels.shape)
        inside = np.isin(labels[y0:y1, x0:x1], members)
        own = field[y0:y1, x0:x1][inside].mean(axis=0)
        around = background[y0:y1, x0:x1][inside].mean(axis=0)
        if np.sqrt(((own - around) ** 2).sum()) >= DISTINCT_FILL:
            distinct.append(members)
    return distinct


def _extend_background(field, labels, stats, anchors, rooms):
    """Background for pixels near anchor groups, extended from the nearest one.

    A group in the pixel's own room (see ``HARD_EDGE``) wins over a nearer
    one across a hard edge; pixels on a hard edge take the nearest group.
    """
    group_of = np.full(len(stats), -1, np.int32)
    for index, members in enumerate(anchors):
        group_of[members] = index
    anchor_mask = group_of[labels] >= 0
    distance = cv2.distanceTransform((~anchor_mask).astype(np.uint8), cv2.DIST_L2, 3)

    background = field.copy()
    best = np.full(labels.shape, np.inf, np.float32)
    for index, members in enumerate(anchors):
        x0, y0, x1, y1 = _bounds(stats, members, BAND_RADIUS + 1, labels.shape)
        inside = group_of[labels[y0:y1, x0:x1]] == index
        reach = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3)
        local_rooms = rooms[y0:y1, x0:x1]
        own_rooms = np.unique(local_rooms[inside])
        across = ~np.isin(local_rooms, own_rooms[own_rooms > 0])
        score = reach + across * (2 * BAND_RADIUS)
        targets = (
            (reach <= BAND_RADIUS) & ~anchor_mask[y0:y1, x0:x1] & (score < best[y0:y1, x0:x1])
        )
        if targets.any():
            extended = _masked_blur(field[y0:y1, x0:x1], inside.astype(np.float32), EXTEND_SIGMA)
            background[y0:y1, x0:x1][targets] = extended[targets]
            best[y0:y1, x0:x1][targets] = score[targets]
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


class _Fills(NamedTuple):
    gradient: np.ndarray
    labels: np.ndarray
    stats: np.ndarray
    field: np.ndarray
    covered: np.ndarray
    deviations: np.ndarray
    fills: list
    large: list


def _find_fills(lab: np.ndarray) -> _Fills:
    gradient = _gradient(lab)
    structure = _structure_mask(gradient)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(1 - structure, connectivity=4)
    groups = _group_regions(lab, labels, stats, _flat_regions(lab, labels, stats))
    field, covered, deviations, spread = _group_fields(lab, labels, stats, groups)
    fills = [members for members, colours in zip(groups, spread) if colours <= FILL_RANGE]
    large = [members for members in fills if stats[members, 4].sum() >= ANCHOR_AREA]
    return _Fills(gradient, labels, stats, field, covered, deviations, fills, large)


def _measure(found: _Fills) -> dict:
    fill_labels = np.concatenate(found.fills) if found.fills else np.zeros(0, np.int32)
    # The photo guard counts large regions, not groups: fragments of a sky
    # joined into one group must not lift a photo over MIN_FILL_SHARE.
    fill_area = found.stats[fill_labels, 4]
    fill_mask = np.isin(found.labels, fill_labels)
    return {
        "covered_fraction": round(float(found.covered.mean()), 4),
        "fill_share": round(float(fill_area[fill_area >= ANCHOR_AREA].sum() / fill_mask.size), 4),
        "regions": len(fill_labels),
        "stain_energy_before": round(_stain_energy(found.deviations, fill_mask), 4),
    }


def clean_stains(image: Image.Image) -> tuple[Image.Image, dict]:
    """Return ``(result, report)``; ``result`` is ``image`` when untouched."""
    rgb = np.asarray(image.convert("RGB"))
    scale = GUARD_SIZE / max(rgb.shape[:2])
    guard = rgb
    if scale < 1:
        guard = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found = _find_fills(_to_lab(guard))
    report = _measure(found)
    if (
        report["fill_share"] < MIN_FILL_SHARE
        or report["stain_energy_before"] < MIN_STAIN_ENERGY
    ):
        report.update(changed_fraction=0.0, stain_energy_after=report["stain_energy_before"])
        return image, report

    lab = _to_lab(rgb)
    if guard is not rgb:
        found = _find_fills(lab)
        report.update(_measure(found), fill_share=report["fill_share"])
    gradient, labels, stats, field, _, _, fills, large = found

    _, rooms = cv2.connectedComponents((gradient <= HARD_EDGE).astype(np.uint8), connectivity=4)
    background, _, _ = _extend_background(field, labels, stats, large, rooms)
    small = [members for members in fills if stats[members, 4].sum() < ANCHOR_AREA]
    anchors = large + _distinct_fills(field, background, labels, stats, small)
    background, anchored, distance = _extend_background(field, labels, stats, anchors, rooms)
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

    _, after_covered, after_deviations, _ = _group_fields(_to_lab(output), labels, stats, fills)
    report["changed_fraction"] = round(float(changed.mean()), 4)
    report["stain_energy_after"] = round(_stain_energy(after_deviations, after_covered), 4)
    return result, report
