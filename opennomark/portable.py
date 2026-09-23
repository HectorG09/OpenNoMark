"""Keep model downloads and temporary files inside the project folder.

This fork runs from a removable drive, so nothing it downloads or writes may
land in the user's home directory. When running from a source checkout, the
standard cache variables default to ``<project>/.cache``; torch hub (LaMa,
waifu2x) and Hugging Face (OWLv2, OCR) then resolve their caches there, and
``tempfile`` writes API uploads there. An explicit ``XDG_CACHE_HOME`` still
wins, and an installed wheel keeps the libraries' own defaults.
"""

import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"


def configure() -> None:
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        return
    if "XDG_CACHE_HOME" not in os.environ:
        # torch.hub -> $XDG_CACHE_HOME/torch/hub, Hugging Face -> $XDG_CACHE_HOME/huggingface
        CACHE_DIR.mkdir(exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
    # macOS sets TMPDIR for every login session, so it is not a caller
    # choice and is always redirected.
    temp_dir = CACHE_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)
    # tempfile memoizes its directory on first use; make it re-read TMPDIR.
    tempfile.tempdir = None
