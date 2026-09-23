# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Common commands

Dependency management uses **uv** (not pip). Frontend uses **npm**.

```bash
# Install deps
uv sync                         # core (torch, transformers, opencv, PIL)
uv sync --extra api             # + fastapi/uvicorn for Web UI
uv sync --extra dev             # + pytest

# CLI (entry point declared in pyproject.toml → opennomark.cli:main)
uv run opennomark <files|dirs|globs> -o output/ [--debug] [--device cpu|cuda|mps]
uv run opennomark <inputs> -o output/ --mode stains [--upscale]   # ChatGPT blotches, optional waifu2x after
uv run opennomark <inputs> -o output/ --mode waifu2x --waifu2x-model art|art_scan|photo --waifu2x-scale 1|2|4 --waifu2x-noise -1..3

# API server (singleton pipeline loaded on first request)
uv run uvicorn opennomark.api:app --port 48291

# Frontend (React 19 + Vite 8 + Tailwind 4)
cd frontend && npm install && npm run dev      # dev server on :48292
cd frontend && npm run build                   # builds to frontend/dist (auto-served by api.py if present)
cd frontend && npm run lint

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_pipeline.py -v                     # single file
uv run pytest tests/test_pipeline.py::test_name -v          # single test

# One-shot start (installs deps, runs backend + frontend)
./start.sh        # macOS/Linux
start.bat         # Windows
```

Tests referencing real sample images in `gemini_images/` and `豆包/` auto-skip via `pytest.skip` when those directories are absent — see `tests/conftest.py:45-61`.

## Architecture

### Fused visual-expert pipeline (`opennomark/localizer.py`)

```
image ──┬─► Gemini catalog detector (48/32, 96/64, 96/192)
        ├─► OWLv2 corner prompts ─┐
        ├─► OWLv2 generic prompts├─► conservative cross-expert fusion
        └─► lazy PP-OCRv5 polygons┘              │
                                                 ▼
                                    precise mask(s) → local LaMa
                                                 │
                                                 ▼
                                          residual check
```

The precise Gemini shape mask remains the default when experts overlap. A
non-overlapping calibrated corner signature can replace a weak spatial match
only when it is in the same corner, scores at least as strongly, and lies
closer to the edges. Separately confirmed generic regions can coexist so one
image may contain multiple removal targets.

### Gemini branch (`opennomark/gemini_alpha.py`)

- **Resolution tiers first**: large official outputs try 96/192 and 96/64; preview/1K outputs try 48/32 and legacy 96/64.
- **Two-signal scoring**: spatial NCC and Sobel-edge NCC must both clear the thresholds in `opennomark/assets/gemini_detector.json`.
- **Tight shape mask**: `create_gemini_mask` thresholds the known alpha silhouette, dilates it by 5px, and feathers by 2px. Do not replace it with a full rectangle without re-validating the real corpus.
- The older reverse-alpha helpers remain experimental. Production uses the shape mask with local LaMa because reverse alpha can leave positive/negative diamond residuals on complex backgrounds.

### Open-vocabulary, OCR, and LaMa

**Detector (`opennomark/detector.py`)**: OWLv2 runs established corner prompts and generic watermark prompts in separate passes. Keep the query competition separate: mixing new phrases into the calibrated platform pass regresses the real corpus. `filter_watermarks` uses a `corner_signature` lane plus a stricter `generic_anywhere` lane, permits at most four regions, and blocks automatic removal when candidate count or area budgets overflow.

**Text detector (`opennomark/text_detector.py`)**: PP-OCRv5 mobile detection and recognition are loaded lazily. Strong watermark vocabulary can be accepted anywhere; URLs, handles, and dates require edge placement. OCR polygons refine generic OWLv2 rectangles. Never convert arbitrary recognized scene text into a removal mask, and keep dense/tiled candidates fail-closed.

**Inpainter (`opennomark/inpainter.py`)**: wraps a TorchScript LaMa model.
- **Tight mask defaults (`padding=3, feather=4`) are load-bearing** — see the docstring at line 28. Larger values cause LaMa to bleed across high-contrast structural edges (e.g. paint white fabric over a black panel adjacent to a sparkle). Do not relax these without re-validating on the `examples/` set.
- The model is downloaded on first run from `github.com/enesmsahin/simple-lama-inpainting/releases` to `<torch hub dir>/checkpoints/big-lama.pt` (inside the project, see *Portable caches*).
- Loaded with `map_location="cpu"` because the serialized checkpoint is CUDA — this is what lets it run on Mac.
- After LaMa produces a result, `inpaint()` alpha-blends it back against the original using the feathered mask, so only the masked pixels are modified.

### Stains mode (`opennomark/stain_cleaner.py`)

ChatGPT infographics have no localized mark; their defect is an 8-16px blotch texture inside flat fills, strongest beside dark text. `--mode stains` / `mode=stains` skips the localizer entirely (OWLv2 erased the headline of the reference sample as a "watermark") and runs a deterministic cleaner: structure mask from Lab gradients, connected flat regions, a robust masked-Gaussian field per region and a soft blend toward it. JPEG ringing beside glyphs counts as structure, so the most visible blotches sit in a 16px text band: there the field of the nearest large region (`ANCHOR_AREA`) is extended by normalised convolution (copying the single nearest pixel drew streaks and stepped patches), and each pixel is projected onto a two-colour ink/background model; only mostly-background pixels (`MAX_INK_ALPHA`) are rebuilt, and pixels lighter than the fill count as stain. Lab is computed in floating point on OpenCV's 8-bit scale and untouched pixels are copied bit for bit: an 8-bit Lab round trip alone moved half the pixels of the reference JPEG by up to 15 levels. Only near-constant fills (`FILL_RANGE`) anchor the background, and an image whose large fills cover less than `MIN_FILL_SHARE` of it is returned untouched: without that guard, portraits in `examples/` lost cloud texture and hair detail. `tests/test_stain_cleaner.py` runs real `examples/` photos through the cleaner and requires them unchanged. Small fills whose colour differs from their surroundings (`DISTINCT_FILL`, e.g. green "SI" badges) anchor themselves; gaps between glyphs do not. LaMa over a stain mask was measured and rejected: ~140s on CPU for a 1024x1536 sample and new speckles beside glyphs. Validation re-measures stain energy on the cleaned regions (`MAX_RESIDUAL_RATIO`), mirroring the residual check of the watermark path.

### waifu2x (`opennomark/enhancer.py`)

The official `nagadomi/nunif` torch.hub entry, pinned to one commit (`skip_validation=True` is required with a pinned SHA); models download on first use (~420MB). `Waifu2xSettings(model, scale, noise)` covers the swin_unet `art`/`art_scan`/`photo` families, 1x/2x/4x and noise -1 (off) to 3; 1x without noise reduction is rejected. The default (`art`, 2x, noise 1) is tuned on the reference sample: `art_scan` paints striations into flat white. It runs after stain cleanup (`enhance` / `--upscale`) or alone (`mode=waifu2x`). One nunif model is cached per (model, method, noise), and inference is serialized by a lock. torch is imported lazily so the API and CLI can validate settings without it.

### Metadata (`opennomark/metadata.py`)

Every output passes through `strip_metadata`, including unchanged copies returned by the API. It rewrites the JPEG/PNG/WebP container without re-encoding: EXIF, XMP, IPTC, comments, C2PA/JUMBF, PNG text chunks and data after EOI are dropped; JFIF, ICC, Adobe, and PNG colour chunks are kept. `_save_result` also carries the input's ICC profile (Mac screenshots are Display P3) and saves JPEG at 4:4:4.

### Portable caches (`opennomark/portable.py`)

This fork runs from a USB drive and must not write to the home directory. Importing `opennomark` from a source checkout points `XDG_CACHE_HOME` at `<project>/.cache` (torch hub, Hugging Face) and forces `TMPDIR` to `<project>/.cache/tmp` (macOS always sets its own). The `.app` launcher also sets `UV_CACHE_DIR` and `npm_config_cache` there. `.cache/` is git-ignored.

### Device selection

Both models accept a `device` kwarg (passed through from the `--device` CLI flag) and auto-select when not given, but they select **differently**:

- `WatermarkDetector` (OWLv2): MPS → CUDA → CPU.
- `Waifu2xEnhancer`: CUDA → MPS → CPU (nunif's own selection for `device_ids=[0]`).
- `LamaInpainter`: CUDA → CPU. **MPS requests are silently rewritten to CPU** because LaMa's TorchScript graph uses ops (e.g. FFT variants) that Apple MPS does not support — trying to run on MPS produces garbage, not an error. Do not "fix" this redirect without first confirming the full LaMa op set is MPS-supported.

The checkpoint is CUDA-serialized, so `LamaInpainter` always loads with `map_location="cpu"` then `.to(self.device)` — loading directly to CUDA on a CPU-only machine would fail at unpickle time.

Pipeline does **not** cache models across invocations when used via CLI, but `api.py` holds a module-level `_pipeline` singleton that is lazy-initialized on first `/api/remove` request.

### FastAPI backend (`opennomark/api.py`)

- Uploads and outputs are written to `tempfile.gettempdir()/opennomark_{uploads,outputs}` with an 8-char job id.
- If `frontend/dist` exists, it is mounted at `/` so the single-port deployment works (`uvicorn` only, no vite dev server needed).
- CORS is wide-open (`allow_origins=["*"]`) for the split-port dev setup.
- Split-port development uses backend `48291` and frontend `48292`; keep the Vite proxy and launch scripts aligned when changing either port.

## Directories worth knowing

- `opennomark/assets/` — Gemini alpha maps plus calibrated detector profiles; re-run the real corpus when changing them.
- `scripts/train_gemini_detector.py` — calibration entry point for real Gemini positives and hard negatives.
- `examples/` — the seven-provider real-image acceptance corpus plus canonical README outputs.
- `experiments/` — ablation scripts (`exp_decision.py`, `exp_gain_sweep.py`, `exp_linear_light.py`, `exp_mask_shape.py`, `exp_posterior.py`) that justify the current threshold constants. Consult these before tuning Stage 1 thresholds or mask parameters.
- `gemini_images/`, `豆包/`, `verify/` — larger real-image test sets used by fixtures; not required for unit tests.
- `skills/opennomark/` — portable Agent Skill discovered through the cross-agent skills layout.
