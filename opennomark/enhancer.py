"""waifu2x denoise + 2x upscaling from the official nagadomi/nunif torch.hub entry.

nunif is the PyTorch successor of nagadomi/waifu2x, by the same author.
torch.hub fetches its code pinned to one commit, and nunif then downloads
its pretrained models (~420MB) into the torch hub cache, which
``portable.configure`` keeps inside the project folder.
"""

import torch

# Pinned so a new upstream commit never runs unreviewed. ``skip_validation``
# is required with a pin: torch.hub otherwise asks the GitHub API whether the
# ref is the head of a branch or tag, which stops being true once master moves.
NUNIF_REPO = "nagadomi/nunif:d23721f1b5f0a4c92c3ee1be013180bf298730c5"
MODEL_TYPE = "art_scan"
# Level 1 removes the JPEG ringing left around glyphs (inputs are usually
# chat-app JPEGs) without softening fine lines; the stain cleaner has already
# flattened the fills. MPS output matches CPU within 2/255 on the reference
# sample, so nunif's MPS path is safe, unlike LaMa's.
NOISE_LEVEL = 1


def _device_ids(device):
    """nunif selects CUDA, then MPS, for any non-negative id; -1 means CPU."""
    if device is None:
        accelerated = torch.cuda.is_available() or torch.backends.mps.is_available()
        return [0] if accelerated else [-1]
    kind = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    return [-1] if kind == "cpu" else [0]


class Waifu2xEnhancer:
    def __init__(self, device=None):
        self.model = torch.hub.load(
            NUNIF_REPO,
            "waifu2x",
            model_type=MODEL_TYPE,
            method="noise_scale2x",
            noise_level=NOISE_LEVEL,
            device_ids=_device_ids(device),
            trust_repo=True,
            skip_validation=True,
            verbose=False,
        )

    def upscale(self, image):
        return self.model.infer(image.convert("RGB")).convert("RGB")
