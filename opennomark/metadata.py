"""Lossless metadata removal for JPEG, PNG and WebP files.

Rewrites the container instead of re-encoding pixels, so a saved result is not
compressed a second time. Removes EXIF, XMP, IPTC, comments, C2PA/JUMBF
manifests, PNG text chunks and anything appended after the image. Keeps only
what decoding and colour need: JFIF, ICC profiles, Adobe colour transform,
PNG gamma/chromaticity/transparency and animation chunks.

Removing metadata does not make an image unrecognisable as AI-generated;
detectors that analyse pixel content are unaffected.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile

_JPEG_NAMES = {0xE1: "APP1 (EXIF/XMP)", 0xEB: "APP11 (C2PA/JUMBF)", 0xED: "APP13 (IPTC)", 0xFE: "COM"}

_PNG_KEEP = {
    b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM", b"sRGB",
    b"iCCP", b"sBIT", b"cICP", b"mDCV", b"cLLI", b"bKGD", b"pHYs",
    b"acTL", b"fcTL", b"fdAT",
}

_WEBP_KEEP = {b"VP8 ", b"VP8L", b"VP8X", b"ALPH", b"ANIM", b"ANMF", b"ICCP"}
_VP8X_EXIF = 0x08
_VP8X_XMP = 0x04


def _keep_jpeg_segment(marker: int, payload: bytes) -> bool:
    if marker == 0xE0:
        return payload.startswith(b"JFIF\x00")
    if marker == 0xE2:
        return payload.startswith(b"ICC_PROFILE\x00")
    if marker == 0xEE:
        return payload.startswith(b"Adobe")
    return not (0xE0 <= marker <= 0xEF or marker == 0xFE)


def _scan_end(data: bytes, start: int) -> int:
    """Return the offset of the first real marker after entropy-coded data."""
    pos = start
    while True:
        pos = data.find(b"\xff", pos)
        if pos < 0 or pos + 1 >= len(data):
            return len(data)
        following = data[pos + 1]
        if following == 0x00 or 0xD0 <= following <= 0xD7:
            pos += 2  # byte stuffing or restart marker inside the scan
        elif following == 0xFF:
            pos += 1  # fill byte before a marker
        else:
            return pos


def _strip_jpeg(data: bytes) -> tuple[bytes, list[str]]:
    out = bytearray(data[:2])
    removed = []
    pos = 2
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            raise ValueError("Corrupt JPEG marker stream")
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0xD9:
            out += b"\xff\xd9"
            break  # anything after EOI (MPF images, gain maps, trailers) is dropped
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out += data[pos:pos + 2]
            pos += 2
            continue
        (length,) = struct.unpack(">H", data[pos + 2:pos + 4])
        end = pos + 2 + length
        if marker == 0xDA:
            end = _scan_end(data, end)
            out += data[pos:end]
        elif _keep_jpeg_segment(marker, data[pos + 4:end]):
            out += data[pos:end]
        else:
            removed.append(_JPEG_NAMES.get(marker, f"APP{marker - 0xE0}"))
        pos = end
    return bytes(out), removed


def _strip_png(data: bytes) -> tuple[bytes, list[str]]:
    out = bytearray(data[:8])
    removed = []
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if kind in _PNG_KEEP:
            out += data[pos:end]
        else:
            removed.append(kind.decode("latin-1"))
        pos = end
        if kind == b"IEND":
            break
    return bytes(out), removed


def _strip_webp(data: bytes) -> tuple[bytes, list[str]]:
    body = bytearray(b"WEBP")
    removed = []
    pos = 12
    while pos + 8 <= len(data):
        kind = data[pos:pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        end = pos + 8 + size + (size & 1)
        chunk = bytearray(data[pos:end])
        if kind == b"VP8X":
            chunk[8] &= ~(_VP8X_EXIF | _VP8X_XMP) & 0xFF
        if kind in _WEBP_KEEP:
            body += chunk
        else:
            removed.append(kind.decode("latin-1").strip())
        pos = end
    return b"RIFF" + struct.pack("<I", len(body)) + bytes(body), removed


def strip_metadata_bytes(data: bytes) -> tuple[bytes, list[str]]:
    """Return ``(clean_bytes, removed_block_names)``; unknown formats pass through."""
    if data.startswith(b"\xff\xd8"):
        return _strip_jpeg(data)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _strip_png(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _strip_webp(data)
    return data, []


def strip_metadata(path: str | os.PathLike) -> list[str]:
    """Rewrite ``path`` in place without metadata and return what was removed."""
    with open(path, "rb") as file:
        data = file.read()
    clean, removed = strip_metadata_bytes(data)
    if clean != data:
        directory = os.path.dirname(os.path.abspath(path))
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
            file.write(clean)
        shutil.copymode(path, file.name)
        os.replace(file.name, path)
    return removed
