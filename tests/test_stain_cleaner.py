"""Tests for the flat-region stain cleaner."""

import cv2
import numpy as np
import pytest
from PIL import Image

from opennomark.stain_cleaner import (
    HARD_EDGE,
    MAX_RESIDUAL_RATIO,
    MIN_FILL_SHARE,
    MIN_STAIN_ENERGY,
    _extend_background,
    _gradient,
    _to_lab,
    clean_stains,
)

BLUE = (20, 100, 175)
GREEN = (22, 140, 50)
WHITE = (252, 252, 252)


def test_blotches_are_flattened_and_validated(stained_graphic):
    stained, clean, text = stained_graphic()
    result, report = clean_stains(stained)

    assert result is not stained
    assert report["stain_energy_before"] > MIN_STAIN_ENERGY
    assert report["stain_energy_after"] <= report["stain_energy_before"] * MAX_RESIDUAL_RATIO

    background = cv2.dilate(text, np.ones((9, 9), np.uint8)) == 0
    reference = np.asarray(clean, np.float32)[background]
    before = np.abs(np.asarray(stained, np.float32)[background] - reference).mean()
    after = np.abs(np.asarray(result, np.float32)[background] - reference).mean()
    assert after < before * 0.5


# Coloured ink after chat-app JPEG recompression: the case where an 8-bit Lab
# round trip altered glyphs by 10 levels and ringing hid stains from cleanup.
REALISTIC = {"ink": (30, 60, 140), "jpeg_quality": 75}


def test_text_pixels_are_untouched(stained_graphic):
    stained, _, text = stained_graphic(**REALISTIC)
    result, _ = clean_stains(stained)

    glyph_core = cv2.erode(text, np.ones((3, 3), np.uint8)) == 255
    assert glyph_core.sum() > 200
    original = np.asarray(stained)[glyph_core]
    cleaned = np.asarray(result)[glyph_core]
    assert np.array_equal(original, cleaned)


def test_stains_hugging_the_text_are_removed(stained_graphic):
    """Ringing beside glyphs counts as structure; the text band must still be cleaned."""
    stained, clean, text = stained_graphic(**REALISTIC)
    result, _ = clean_stains(stained)

    near = cv2.dilate(text, np.ones((9, 9), np.uint8)) > 0
    band = near & (cv2.dilate(text, np.ones((3, 3), np.uint8)) == 0)
    reference = np.asarray(clean, np.float32)[band]
    before = np.abs(np.asarray(stained, np.float32)[band] - reference).mean()
    after = np.abs(np.asarray(result, np.float32)[band] - reference).mean()
    assert after < before * 0.4


def _enclosed_gap(seed):
    """A blue box whose bold "text" encloses a stained line gap.

    The gap is closed at both ends and its middle sits 19px from the box body,
    beyond the text band: the case of the reference diamond, where the gap
    between two lines kept its blotches. Returns ``(image, clean, gap)``.
    """
    clean = np.full((240, 320, 3), WHITE, np.float32)
    clean[40:200, 40:280] = BLUE
    rng = np.random.default_rng(seed)
    coarse = rng.normal(0.0, 1.0, (30, 40, 3)).astype(np.float32)
    stained = clean + cv2.resize(coarse, (320, 240), interpolation=cv2.INTER_CUBIC) * 2.0
    gap = (slice(111, 121), slice(142, 178))
    stained[gap] += 7.0

    ink = np.zeros((240, 320), np.float32)
    ink[97:111, 128:192] = ink[121:135, 128:192] = 1.0
    ink[111:121, 128:142] = ink[111:121, 178:192] = 1.0
    clean = clean * (1 - ink[..., None]) + 255 * ink[..., None]
    stained = stained * (1 - ink[..., None]) + 255 * ink[..., None]
    image = Image.fromarray(np.clip(np.round(stained), 0, 255).astype(np.uint8))
    return image, clean, gap


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_line_gap_enclosed_by_text_is_cleaned(seed):
    stained, clean, gap = _enclosed_gap(seed)
    result, _ = clean_stains(stained)

    before = np.abs(np.asarray(stained, np.float32)[gap] - clean[gap]).mean()
    after = np.abs(np.asarray(result, np.float32)[gap] - clean[gap]).mean()
    assert after < before * 0.5


def test_badge_pixels_take_the_badge_background_across_its_rim():
    """Inside a badge, the white page beyond the rim is nearer than the badge's own fill."""
    image = np.full((100, 100, 3), WHITE, np.uint8)
    cv2.circle(image, (50, 50), 30, GREEN, -1, cv2.LINE_AA)
    lab = _to_lab(image)
    # Anchors: the page outside the rim and a fragment at the badge centre;
    # the ring between them is band, as when white text splits the badge.
    yy, xx = np.mgrid[:100, :100]
    radius = np.hypot(yy - 50, xx - 50)
    anchors_mask = ((radius > 33) | (radius < 12)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(anchors_mask, connectivity=4)
    _, rooms = cv2.connectedComponents((_gradient(lab) <= HARD_EDGE).astype(np.uint8), connectivity=4)
    anchors = [np.array([labels[0, 0]]), np.array([labels[50, 50]])]

    background, _, _ = _extend_background(lab, labels, stats, anchors, rooms)

    near_rim = background[50, 77]  # 3px inside the rim, 15px from the fragment
    assert np.sqrt(((near_rim - lab[50, 50]) ** 2).sum()) < 1.0


def test_clean_graphic_is_returned_unchanged(stained_graphic):
    clean, _, _ = stained_graphic(blotches=False)
    result, report = clean_stains(clean)

    assert result is clean
    assert report["changed_fraction"] == 0.0


@pytest.mark.parametrize("seed", [0, 1])
def test_photographic_texture_is_not_flattened(seed):
    rng = np.random.default_rng(seed)
    texture = Image.fromarray((rng.random((200, 200, 3)) * 255).astype(np.uint8))
    result, report = clean_stains(texture)

    assert result is texture
    assert report["covered_fraction"] < 0.05


@pytest.mark.parametrize(
    "relative",
    [
        # Portrait whose sky and hair were flattened into patches before the
        # fill-share guard existed.
        "doubao/生成超写实时尚人像 (5).png",
        "doubao/生成超写实时尚人像 (2).png",
        "gemini/gemini_sample_1.png",
        "qwen/image_324855086256596.png",
        # Flat dark backdrops; at native 2400px they measured 0.46 and 0.49
        # before the guard ran at ChatGPT's resolution.
        "gemini/Gemini_Generated_Image_t8zaxrt8zaxrt8za.png",
        "gemini/Gemini_Generated_Image_rnr28rnr28rnr28r.png",
    ],
)
def test_real_photos_are_returned_untouched(relative):
    from pathlib import Path

    path = Path(__file__).parent.parent / "examples" / relative
    if not path.exists():
        pytest.skip(f"{relative} not available")
    image = Image.open(path).convert("RGB")
    result, report = clean_stains(image)

    assert result is image
    assert report["fill_share"] < MIN_FILL_SHARE
