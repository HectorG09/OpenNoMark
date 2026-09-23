"""Tests for waifu2x settings (no model download)."""

import pytest

from opennomark.enhancer import Waifu2xSettings, _device_ids


@pytest.mark.parametrize(
    "scale, noise, method",
    [
        (1, 0, "noise"),
        (1, 3, "noise"),
        (2, -1, "scale"),
        (2, 1, "noise_scale"),
        (4, -1, "scale4x"),
        (4, 2, "noise_scale4x"),
    ],
)
def test_settings_map_to_nunif_methods(scale, noise, method):
    assert Waifu2xSettings(scale=scale, noise=noise).method == method


def test_defaults_match_the_reference_tuning():
    settings = Waifu2xSettings()
    assert (settings.model, settings.scale, settings.noise) == ("art", 2, 1)
    assert settings.label == "waifu2x_art_noise1_2x"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 1, "noise": -1},
        {"model": "anime"},
        {"scale": 3},
        {"noise": 4},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Waifu2xSettings(**kwargs)


def test_cpu_request_maps_to_nunif_cpu_id():
    assert _device_ids("cpu") == [-1]
    assert _device_ids("mps") == [0]
