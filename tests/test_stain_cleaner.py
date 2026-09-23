"""Tests for the flat-region stain cleaner."""

import cv2
import numpy as np
import pytest
from PIL import Image

from opennomark.stain_cleaner import MAX_RESIDUAL_RATIO, MIN_STAIN_ENERGY, clean_stains


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
