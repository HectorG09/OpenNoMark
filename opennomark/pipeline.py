"""Core pipeline: detect watermarks and remove them."""

import os
import threading

from PIL import Image, ImageChops, ImageFilter

from .enhancer import Waifu2xSettings
from .metadata import strip_metadata
from .stain_cleaner import MAX_RESIDUAL_RATIO, clean_stains

MODES = ("watermark", "stains", "waifu2x")


class WatermarkRemovalPipeline:
    def __init__(self, device=None, verbose=True, load_models=True):
        """``load_models=False`` builds a stains-only pipeline without LaMa/OWLv2."""
        self.verbose = verbose
        self.device = device
        self._enhancer = None
        self._enhancer_lock = threading.Lock()
        if not load_models:
            self.localizer = None
            self.inpainter = None
            return
        from .inpainter import LamaInpainter
        from .localizer import WatermarkLocalizer

        if self.verbose:
            print("Loading inpainting model...")
        self.localizer = WatermarkLocalizer(device=device)
        self.inpainter = LamaInpainter(device=device)
        if self.verbose:
            print("Inpainting model loaded.")

    @property
    def enhancer(self):
        """waifu2x, created on first use; its models download on first run (~420MB)."""
        if self._enhancer is None:
            with self._enhancer_lock:
                if self._enhancer is None:
                    from .enhancer import Waifu2xEnhancer

                    self._enhancer = Waifu2xEnhancer(device=self.device)
        return self._enhancer

    @staticmethod
    def _save_result(result, image_path, output_path):
        """Save in the input's format, then strip every metadata block.

        The input's ICC profile is carried over: Mac screenshots are Display
        P3, and dropping the profile shifts their colours.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with Image.open(image_path) as source:
            icc_profile = source.info.get("icc_profile")
        options = {"icc_profile": icc_profile} if icc_profile else {}
        ext = os.path.splitext(image_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            # 4:4:4 chroma: 4:2:0 subsampling smears colour around text and
            # is what paints blotches into flat fills in the first place.
            result.save(output_path, quality=95, subsampling=0, **options)
        else:
            result.save(output_path, **options)
        strip_metadata(output_path)

    def process(self, image_path, output_path=None, save_debug=False):
        """Process a single image. Returns (result_image, metadata).

        Pipeline:
          1. Ask the unified localizer for evidence-backed regions.  It may use
             a precise shape expert or open-vocabulary proposals, but never a
             filename/provider branch.
          2. Inpaint every accepted region on a local crop with its own mask.
          3. Return one metadata contract for every visual watermark family.
        """
        image = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)
        regions, localization = self.localizer.localize(image)
        public_regions = [region.as_metadata() for region in regions]
        methods_used = []

        metadata = {
            "input": image_path,
            "methods": methods_used,
            "localization": localization,
            "total_detections": localization["total_proposals"],
            "watermarks_found": len(regions),
            "regions": public_regions,
            # Backwards-compatible alias for clients written before the unified
            # region contract.  New code should consume ``regions``.
            "boxes": public_regions,
        }
        automatic_removal_blocked = bool(
            localization.get("safety", {}).get("automatic_removal_blocked")
        )

        if not regions:
            if automatic_removal_blocked:
                metadata["validation"] = {
                    "passed": False,
                    "attempts": 0,
                    "overlapping_residual_regions": [],
                    "reason": "candidate_budget_exceeded",
                }
                metadata["status"] = "partial"
            else:
                metadata["status"] = "no_watermark"
            return image, metadata

        result = image
        for region in regions:
            result = self.inpainter.inpaint_local(result, region.mask)
            method = f"{region.source}_local_lama"
            if method not in methods_used:
                methods_used.append(method)

        residual_regions = self.localizer.localize_residuals(result, regions)
        overlapping = self._overlapping_regions(regions, residual_regions)
        attempts = 1
        if overlapping:
            # A detector box can hug the opaque glyph core and miss a faint
            # antialiasing fringe. Retry once with a slightly grown version of
            # the same evidence-backed mask; never expand to unrelated regions.
            for region in regions:
                if any(self._overlap(region.box, residual.box) >= 0.25 for residual in overlapping):
                    expanded = region.mask.filter(ImageFilter.MaxFilter(9))
                    result = self.inpainter.inpaint_local(result, expanded)
            attempts = 2
            residual_regions = self.localizer.localize_residuals(result, regions)
            overlapping = self._overlapping_regions(regions, residual_regions)

        metadata["methods"] = methods_used
        metadata["validation"] = {
            "passed": not overlapping and not automatic_removal_blocked,
            "attempts": attempts,
            "overlapping_residual_regions": [region.as_metadata() for region in overlapping],
        }
        if automatic_removal_blocked:
            metadata["validation"]["reason"] = "candidate_budget_exceeded"
        metadata["status"] = (
            "cleaned"
            if not overlapping and not automatic_removal_blocked
            else "partial"
        )

        # Save outputs
        if output_path:
            self._save_result(result, image_path, output_path)

            if save_debug:
                debug_dir = os.path.dirname(output_path)
                from PIL import ImageDraw
                debug_img = image.copy()
                draw = ImageDraw.Draw(debug_img)
                combined_mask = Image.new("L", image.size, 0)
                for region in regions:
                    x1, y1, x2, y2 = region.box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    combined_mask = ImageChops.lighter(combined_mask, region.mask)
                debug_img.save(os.path.join(debug_dir, f"debug_{filename}"))
                combined_mask.save(os.path.join(debug_dir, f"mask_{filename}"))

        return result, metadata

    def process_manual(self, image_path, regions, output_path=None):
        """Repair user-confirmed normalized rectangles without auto-detection.

        Manual regions are an explicit user confirmation boundary.  Each box
        is inpainted independently so distant selections do not create one
        oversized LaMa crop or alter pixels between the selected areas.
        """
        image = Image.open(image_path).convert("RGB")
        result = image
        public_regions = []

        for region in regions:
            x1 = int(round(float(region["x"]) * image.width))
            y1 = int(round(float(region["y"]) * image.height))
            x2 = int(round((float(region["x"]) + float(region["width"])) * image.width))
            y2 = int(round((float(region["y"]) + float(region["height"])) * image.height))
            x1, x2 = sorted((max(0, x1), min(image.width, x2)))
            y1, y2 = sorted((max(0, y1), min(image.height, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            box = [x1, y1, x2, y2]
            mask = self.inpainter.create_mask(
                image.size,
                [{"box": box}],
                padding=3,
                feather=4,
            )
            result = self.inpainter.inpaint_local(result, mask)
            public_regions.append(
                {
                    "box": box,
                    "source": "manual_selection",
                    "method": "box_mask",
                    "score": 1.0,
                }
            )

        if not public_regions:
            raise ValueError("No valid manual regions were supplied")

        if output_path:
            self._save_result(result, image_path, output_path)

        return result, {
            "status": "cleaned",
            "watermarks_found": len(public_regions),
            "methods": ["manual_selection_local_lama"],
            "regions": public_regions,
            "validation": {
                "passed": True,
                "manual_confirmation": True,
            },
        }

    def process_stains(self, image_path, output_path=None, waifu2x=None):
        """Clean ChatGPT-style blotches in flat fills, then optionally waifu2x.

        ``waifu2x`` is a ``Waifu2xSettings`` or None. The watermark localizer
        does not run in this mode: on flat infographics OWLv2 proposes
        headline text as a watermark, and the reference ChatGPT sample lost
        its title that way. Validation mirrors ``process``: the stain energy
        of the cleaned regions is re-measured and must drop by
        ``MAX_RESIDUAL_RATIO``.
        """
        image = Image.open(image_path).convert("RGB")
        result, report = clean_stains(image)
        stains_cleaned = result is not image
        methods = ["flat_region_stain_cleanup"] if stains_cleaned else []
        if waifu2x is not None:
            result = self.enhancer.enhance(result, waifu2x)
            methods.append(waifu2x.label)

        passed = (
            not stains_cleaned
            or report["stain_energy_after"]
            <= report["stain_energy_before"] * MAX_RESIDUAL_RATIO
        )
        metadata = {
            "input": image_path,
            "mode": "stains",
            "methods": methods,
            "stains": report,
            "watermarks_found": int(stains_cleaned),
            "regions": [],
            "boxes": [],
            "validation": {"passed": passed, "attempts": 1},
            "status": ("cleaned" if passed else "partial") if methods else "no_watermark",
        }
        if output_path and methods:
            self._save_result(result, image_path, output_path)
        return result, metadata

    def process_waifu2x(self, image_path, output_path=None, settings=Waifu2xSettings()):
        """Run waifu2x alone with the caller's model, scale and noise level."""
        image = Image.open(image_path).convert("RGB")
        result = self.enhancer.enhance(image, settings)
        metadata = {
            "input": image_path,
            "mode": "waifu2x",
            "methods": [settings.label],
            "watermarks_found": 0,
            "regions": [],
            "boxes": [],
            "validation": {"passed": True, "attempts": 1},
            "status": "cleaned",
        }
        if output_path:
            self._save_result(result, image_path, output_path)
        return result, metadata

    @classmethod
    def _overlapping_regions(cls, original_regions, residual_regions):
        return [
            residual
            for residual in residual_regions
            if any(cls._overlap(original.box, residual.box) >= 0.25 for original in original_regions)
        ]

    @staticmethod
    def _overlap(reference, other):
        x1 = max(float(reference[0]), float(other[0]))
        y1 = max(float(reference[1]), float(other[1]))
        x2 = min(float(reference[2]), float(other[2]))
        y2 = min(float(reference[3]), float(other[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        reference_area = max(0.0, float(reference[2]) - float(reference[0])) * max(
            0.0, float(reference[3]) - float(reference[1])
        )
        return intersection / reference_area if reference_area else 0.0

    def process_batch(
        self, image_paths, output_dir, save_debug=False, callback=None,
        mode="watermark", waifu2x=None,
    ):
        """Process multiple images. callback(i, total, metadata) for progress."""
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, path in enumerate(image_paths):
            filename = os.path.basename(path)
            # Normalize output filename
            name, ext = os.path.splitext(filename)
            if not ext or ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = ".png"
            out_name = f"clean_{name}{ext}"
            out_path = os.path.join(output_dir, out_name)

            if mode == "stains":
                _, meta = self.process_stains(path, out_path, waifu2x=waifu2x)
            elif mode == "waifu2x":
                _, meta = self.process_waifu2x(path, out_path, waifu2x or Waifu2xSettings())
            else:
                _, meta = self.process(path, out_path, save_debug=save_debug)
            meta["output"] = out_path
            results.append(meta)

            if callback:
                callback(i + 1, len(image_paths), meta)

        return results
