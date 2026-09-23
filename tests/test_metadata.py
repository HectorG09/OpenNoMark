"""Tests for lossless metadata removal."""

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image, ImageCms, PngImagePlugin

from opennomark.metadata import strip_metadata, strip_metadata_bytes

MARKERS = (b"ChatGPT", b"Made with AI", b"trainedAlgorithmicMedia", b"c2pa", b"OpenAI")
XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF>'
    b'<rdf:Description DigitalSourceType="trainedAlgorithmicMedia"/>'
    b"</rdf:RDF></x:xmpmeta>"
)


@pytest.fixture(scope="module")
def icc():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.fixture(scope="module")
def pixels():
    rng = np.random.default_rng(7)
    return Image.fromarray((rng.random((48, 64, 3)) * 255).astype(np.uint8))


@pytest.fixture(scope="module")
def exif():
    data = Image.Exif()
    data[0x0131] = "ChatGPT"  # Software
    data[0x010E] = "Made with AI"  # ImageDescription
    return data


def _assert_clean(original: bytes, cleaned: bytes, icc=None):
    before = np.asarray(Image.open(io.BytesIO(original)).convert("RGB"))
    after_image = Image.open(io.BytesIO(cleaned))
    assert np.array_equal(before, np.asarray(after_image.convert("RGB")))
    assert not [marker for marker in MARKERS if marker in cleaned]
    if icc is not None:
        assert after_image.info.get("icc_profile") == icc


def test_jpeg_drops_exif_xmp_comment_c2pa_and_trailer(pixels, exif, icc):
    buffer = io.BytesIO()
    pixels.save(buffer, "JPEG", exif=exif, xmp=XMP, comment=b"OpenAI", icc_profile=icc)
    payload = b"JP\x00\x00jumbc2pa.manifest"
    app11 = b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload
    data = buffer.getvalue()
    data = data[:2] + app11 + data[2:] + b"OpenAI gain map trailer"

    cleaned, removed = strip_metadata_bytes(data)

    _assert_clean(data, cleaned, icc)
    assert "APP11 (C2PA/JUMBF)" in removed
    assert "COM" in removed
    assert cleaned.endswith(b"\xff\xd9")


def test_progressive_jpeg_scans_survive(pixels, exif):
    buffer = io.BytesIO()
    pixels.save(buffer, "JPEG", exif=exif, progressive=True)
    cleaned, removed = strip_metadata_bytes(buffer.getvalue())

    _assert_clean(buffer.getvalue(), cleaned)
    assert removed == ["APP1 (EXIF/XMP)"]


def test_png_drops_text_exif_and_c2pa_chunks(pixels, exif, icc):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "Made with AI")
    info.add_itxt("XML:com.adobe.xmp", XMP.decode())
    buffer = io.BytesIO()
    pixels.save(buffer, "PNG", pnginfo=info, exif=exif, icc_profile=icc)
    data = buffer.getvalue()
    cabx = struct.pack(">I", 8) + b"caBXjumbc2pa" + struct.pack(">I", zlib.crc32(b"caBXjumbc2pa"))
    iend = data.rfind(b"IEND") - 4
    data = data[:iend] + cabx + data[iend:]

    cleaned, removed = strip_metadata_bytes(data)

    _assert_clean(data, cleaned, icc)
    assert set(removed) == {"tEXt", "iTXt", "eXIf", "caBX"}


def test_webp_drops_exif_and_xmp_and_clears_flags(pixels, exif, icc):
    buffer = io.BytesIO()
    pixels.save(buffer, "WEBP", lossless=True, exif=exif, xmp=XMP, icc_profile=icc)
    cleaned, removed = strip_metadata_bytes(buffer.getvalue())

    _assert_clean(buffer.getvalue(), cleaned, icc)
    assert set(removed) == {"EXIF", "XMP"}
    assert cleaned[20] & 0x0C == 0  # VP8X EXIF/XMP flags
    assert struct.unpack("<I", cleaned[4:8])[0] == len(cleaned) - 8


def test_clean_file_is_left_byte_identical(tmp_path, pixels):
    path = tmp_path / "clean.png"
    pixels.save(path)
    before = path.read_bytes()

    assert strip_metadata(path) == []
    assert path.read_bytes() == before


def test_strip_metadata_rewrites_file_in_place(tmp_path, pixels, exif):
    path = tmp_path / "tagged.jpg"
    pixels.save(path, exif=exif)

    assert strip_metadata(path) == ["APP1 (EXIF/XMP)"]
    assert b"ChatGPT" not in path.read_bytes()
    assert not list(tmp_path.glob("tmp*"))
