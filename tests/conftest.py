"""Shared fixtures for tests."""

import io
import os
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

FILL = (250, 222, 226)
INK = (20, 20, 20)


def _stained_graphic(blotches=True, seed=0, ink=INK, jpeg_quality=None):
    """A pink card with text, optionally carrying ChatGPT-like blotches.

    ``jpeg_quality`` recompresses the stained card the way chat apps do; that
    adds the ringing beside glyphs where the most visible stains live.
    Returns ``(image, clean_reference, text_mask)``.
    """
    clean = np.full((240, 320, 3), FILL, np.float32)
    stained = clean.copy()
    if blotches:
        rng = np.random.default_rng(seed)
        coarse = rng.normal(0.0, 1.0, (30, 40, 3)).astype(np.float32)
        stained += cv2.resize(coarse, (320, 240), interpolation=cv2.INTER_CUBIC) * 4.0

    text = np.zeros((240, 320), np.uint8)
    cv2.putText(text, "Hola 2026", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4, cv2.LINE_AA)
    cv2.putText(text, "dias del periodo", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    alpha = (text.astype(np.float32) / 255.0)[..., None]
    ink = np.array(ink, np.float32)
    clean = clean * (1 - alpha) + ink * alpha
    stained = stained * (1 - alpha) + ink * alpha
    to_image = lambda array: Image.fromarray(np.clip(np.round(array), 0, 255).astype(np.uint8))
    image = to_image(stained)
    if jpeg_quality:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=jpeg_quality)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return image, to_image(clean), text


@pytest.fixture
def stained_graphic():
    """Factory for a ChatGPT-like stained card and its clean reference."""
    return _stained_graphic


@pytest.fixture
def stained_graphic_path(tmp_path):
    """The stained card saved as a JPEG tagged with AI-provenance EXIF."""
    image, _, _ = _stained_graphic()
    exif = Image.Exif()
    exif[0x0131] = "ChatGPT"
    path = tmp_path / "chatgpt_card.jpg"
    image.save(path, quality=95, exif=exif)
    return str(path)


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image with a fake watermark-like element in the corner."""
    img = Image.new("RGB", (800, 1200), color=(40, 40, 45))
    # Draw a small white rectangle in bottom-right corner to simulate watermark
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([750, 1150, 790, 1190], fill=(200, 200, 200))
    path = str(tmp_path / "test_image.png")
    img.save(path)
    return path


@pytest.fixture
def sample_images_dir(tmp_path):
    """Create a directory with multiple test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    from PIL import ImageDraw
    for i, ext in enumerate(["png", "jpg", "jpeg"]):
        img = Image.new("RGB", (400, 600), color=(30 + i * 20, 30, 40))
        draw = ImageDraw.Draw(img)
        draw.rectangle([350, 550, 390, 590], fill=(180, 180, 180))
        img.save(str(img_dir / f"test_{i}.{ext}"))
    return str(img_dir)


@pytest.fixture
def output_dir(tmp_path):
    """Provide a clean output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)


@pytest.fixture
def real_gemini_image():
    """Return path to a real Gemini test image if available."""
    path = os.path.join(os.path.dirname(__file__), "..", "gemini_images", "Gemini_Generated_Image_ (4).png")
    if os.path.exists(path):
        return os.path.abspath(path)
    pytest.skip("Real Gemini test image not available")


@pytest.fixture
def real_doubao_image():
    """Return path to a real Doubao test image if available."""
    base = os.path.join(os.path.dirname(__file__), "..", "豆包")
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.endswith(".jpeg"):
                return os.path.abspath(os.path.join(base, f))
    pytest.skip("Real Doubao test image not available")
