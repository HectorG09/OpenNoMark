"""waifu2x denoising and upscaling from the official nagadomi/nunif torch.hub entry.

nunif is the PyTorch successor of nagadomi/waifu2x, by the same author.
torch.hub fetches its code pinned to one commit, and nunif then downloads
its pretrained models (~420MB) into the torch hub cache, which
``portable.configure`` keeps inside the project folder.
"""

import threading
from dataclasses import dataclass

# torch is imported inside the functions that need it, so the API and CLI can
# validate settings without loading it.

# Pinned so a new upstream commit never runs unreviewed. ``skip_validation``
# is required with a pin: torch.hub otherwise asks the GitHub API whether the
# ref is the head of a branch or tag, which stops being true once master moves.
NUNIF_REPO = "nagadomi/nunif:d23721f1b5f0a4c92c3ee1be013180bf298730c5"
# The swin_unet families; all three ship 1x, 2x and 4x weights.
MODEL_TYPES = ("art", "art_scan", "photo")
SCALES = (1, 2, 4)
# -1 disables noise reduction; 0-3 are waifu2x's JPEG noise levels.
NOISE_LEVELS = (-1, 0, 1, 2, 3)


@dataclass(frozen=True)
class Waifu2xSettings:
    # "art", not "art_scan", by default: the scan model reintroduces
    # paper-like vertical striations into flat white areas (column std 0.19
    # vs 0.00 on the reference sample), which users see as leftover stains.
    model: str = "art"
    scale: int = 2
    # Level 1 removes the JPEG ringing left around glyphs of chat-app JPEGs
    # without softening fine lines.
    noise: int = 1

    def __post_init__(self):
        if self.model not in MODEL_TYPES:
            raise ValueError(f"waifu2x model must be one of {', '.join(MODEL_TYPES)}")
        if self.scale not in SCALES:
            raise ValueError("waifu2x scale must be 1, 2 or 4")
        if self.noise not in NOISE_LEVELS:
            raise ValueError("waifu2x noise level must be -1 (off) or 0-3")
        if self.scale == 1 and self.noise < 0:
            raise ValueError("waifu2x needs noise reduction or upscaling; 1x without noise reduction does nothing")

    @property
    def method(self):
        """nunif's method name for this combination."""
        if self.scale == 1:
            return "noise"
        suffix = "scale4x" if self.scale == 4 else "scale"
        return f"noise_{suffix}" if self.noise >= 0 else suffix

    @property
    def label(self):
        noise = f"noise{self.noise}" if self.noise >= 0 else "no_noise"
        return f"waifu2x_{self.model}_{noise}_{self.scale}x"


def _device_ids(device):
    """nunif selects CUDA, then MPS, for any non-negative id; -1 means CPU.

    MPS output matches CPU within 2/255 on the reference sample, so nunif's
    MPS path is safe, unlike LaMa's.
    """
    import torch

    if device is None:
        accelerated = torch.cuda.is_available() or torch.backends.mps.is_available()
        return [0] if accelerated else [-1]
    kind = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    return [-1] if kind == "cpu" else [0]


class Waifu2xEnhancer:
    """Loads one nunif model per (model, method, noise) on first use."""

    def __init__(self, device=None):
        self.device_ids = _device_ids(device)
        self._models = {}
        # Serializes loading and inference: the API runs requests in threads
        # and nunif models are not documented as safe for concurrent calls.
        self._lock = threading.Lock()

    def _model(self, settings):
        key = (settings.model, settings.method, settings.noise)
        if key not in self._models:
            import torch

            self._models[key] = torch.hub.load(
                NUNIF_REPO,
                "waifu2x",
                model_type=settings.model,
                method=settings.method,
                noise_level=settings.noise,
                device_ids=self.device_ids,
                trust_repo=True,
                skip_validation=True,
                verbose=False,
            )
        return self._models[key]

    def enhance(self, image, settings=Waifu2xSettings()):
        with self._lock:
            return self._model(settings).infer(image.convert("RGB")).convert("RGB")
