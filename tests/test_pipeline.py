"""Unit tests for the full pipeline."""

import os
import pytest
from PIL import Image


class TestPipeline:
    """Test the complete watermark removal pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from opennomark.pipeline import WatermarkRemovalPipeline
        return WatermarkRemovalPipeline(device="cpu")

    def test_process_synthetic(self, pipeline, sample_image, output_dir):
        out_path = os.path.join(output_dir, "clean_test.png")
        result_img, meta = pipeline.process(sample_image, out_path)
        assert result_img is not None
        assert result_img.size[0] > 0
        assert meta["status"] in ("cleaned", "no_watermark")
        assert "watermarks_found" in meta

    def test_process_no_output_path(self, pipeline, sample_image):
        result_img, meta = pipeline.process(sample_image)
        assert result_img is not None
        assert meta["status"] in ("cleaned", "no_watermark")

    def test_candidate_budget_overflow_returns_partial_without_edit(self, tmp_path):
        from opennomark.pipeline import WatermarkRemovalPipeline

        class BlockedLocalizer:
            def localize(self, image):
                return [], {
                    "total_proposals": 7,
                    "accepted_regions": 0,
                    "experts": ["open_vocabulary", "ocr_text"],
                    "safety": {
                        "automatic_removal_blocked": True,
                        "overflow": [
                            {"expert": "ocr_text", "reason": "max_regions"}
                        ],
                    },
                }

        source = tmp_path / "tiled.png"
        original = Image.new("RGB", (80, 60), color=(12, 34, 56))
        original.save(source)
        pipeline = WatermarkRemovalPipeline.__new__(WatermarkRemovalPipeline)
        pipeline.verbose = False
        pipeline.device = "cpu"
        pipeline.localizer = BlockedLocalizer()
        pipeline.inpainter = None

        result, metadata = pipeline.process(str(source))

        assert result.tobytes() == original.tobytes()
        assert metadata["status"] == "partial"
        assert metadata["watermarks_found"] == 0
        assert metadata["validation"] == {
            "passed": False,
            "attempts": 0,
            "overlapping_residual_regions": [],
            "reason": "candidate_budget_exceeded",
        }

    def test_process_batch(self, pipeline, sample_images_dir, output_dir):
        import glob
        paths = sorted(
            glob.glob(os.path.join(sample_images_dir, "*.png"))
            + glob.glob(os.path.join(sample_images_dir, "*.jpg"))
            + glob.glob(os.path.join(sample_images_dir, "*.jpeg"))
        )
        assert len(paths) == 3

        progress_calls = []
        def on_progress(i, total, meta):
            progress_calls.append((i, total, meta["status"]))

        results = pipeline.process_batch(paths, output_dir, callback=on_progress)
        assert len(results) == 3
        assert len(progress_calls) == 3
        for r in results:
            assert r["status"] in ("cleaned", "no_watermark")

    def test_process_batch_with_debug(self, pipeline, sample_images_dir, tmp_path):
        import glob
        out = str(tmp_path / "debug_out")
        paths = glob.glob(os.path.join(sample_images_dir, "*.png"))
        results = pipeline.process_batch(paths, out, save_debug=True)
        # Check debug files exist for cleaned images
        for r in results:
            if r["status"] == "cleaned":
                name = os.path.basename(r["input"])
                assert os.path.exists(os.path.join(out, f"debug_{name}"))
                assert os.path.exists(os.path.join(out, f"mask_{name}"))

    def test_e2e_gemini(self, pipeline, real_gemini_image, output_dir):
        """E2E: process a real Gemini image and verify watermark removal."""
        out_path = os.path.join(output_dir, "clean_gemini.png")
        result_img, meta = pipeline.process(real_gemini_image, out_path)
        assert meta["status"] == "cleaned"
        assert meta["watermarks_found"] >= 1
        assert meta["methods"] == ["spatial_template_local_lama"]
        assert meta["total_detections"] >= 1
        assert meta["localization"]["accepted_regions"] == 1
        assert meta["localization"]["experts"] == [
            "spatial_template",
            "open_vocabulary",
        ]
        assert meta["regions"][0]["source"] == "spatial_template"
        assert os.path.exists(out_path)
        # Verify output is a valid image
        check = Image.open(out_path)
        assert check.size[0] > 0

    def test_e2e_doubao(self, pipeline, real_doubao_image, output_dir):
        """E2E: process a real Doubao image and verify watermark removal."""
        out_path = os.path.join(output_dir, "clean_doubao.jpg")
        result_img, meta = pipeline.process(real_doubao_image, out_path)
        assert meta["status"] == "cleaned"
        assert meta["watermarks_found"] >= 1
        assert os.path.exists(out_path)


class TestStainsMode:
    """Stains mode runs without LaMa/OWLv2 and never leaks metadata."""

    @pytest.fixture
    def pipeline(self):
        from opennomark.pipeline import WatermarkRemovalPipeline
        return WatermarkRemovalPipeline(verbose=False, load_models=False)

    def test_stained_card_is_cleaned_without_detection_models(
        self, pipeline, stained_graphic_path, tmp_path
    ):
        from PIL import JpegImagePlugin

        output = tmp_path / "out.jpg"
        _, meta = pipeline.process_stains(stained_graphic_path, str(output))

        assert pipeline.localizer is None and pipeline.inpainter is None
        assert meta["status"] == "cleaned"
        assert meta["methods"] == ["flat_region_stain_cleanup"]
        assert meta["validation"]["passed"]
        saved = Image.open(output)
        assert "exif" not in saved.info
        assert b"ChatGPT" not in output.read_bytes()
        assert JpegImagePlugin.get_sampling(saved) == 0  # 4:4:4

    def test_clean_card_is_reported_without_writing(self, pipeline, stained_graphic, tmp_path):
        source = tmp_path / "clean.png"
        stained_graphic(blotches=False)[0].save(source)
        output = tmp_path / "out.png"

        _, meta = pipeline.process_stains(str(source), str(output))

        assert meta["status"] == "no_watermark"
        assert not output.exists()

    def test_icc_profile_survives_the_save(self, pipeline, stained_graphic, tmp_path):
        from PIL import ImageCms

        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        source = tmp_path / "p3.png"
        stained_graphic()[0].save(source, icc_profile=icc)
        output = tmp_path / "out.png"

        pipeline.process_stains(str(source), str(output))

        assert Image.open(output).info.get("icc_profile") == icc

    class ScaleBy:
        """Stands in for waifu2x: resizes by the requested scale."""

        def __init__(self):
            self.calls = []

        def enhance(self, image, settings):
            self.calls.append(settings)
            return image.resize((image.width * settings.scale, image.height * settings.scale))

    def test_stains_then_waifu2x_uses_the_callers_settings(
        self, pipeline, stained_graphic_path, tmp_path
    ):
        from opennomark.enhancer import Waifu2xSettings

        pipeline._enhancer = self.ScaleBy()
        settings = Waifu2xSettings(model="photo", scale=4, noise=-1)
        result, meta = pipeline.process_stains(
            stained_graphic_path, str(tmp_path / "out.jpg"), waifu2x=settings
        )

        assert pipeline._enhancer.calls == [settings]
        assert result.size == (1280, 960)
        assert meta["methods"] == ["flat_region_stain_cleanup", "waifu2x_photo_no_noise_4x"]

    def test_waifu2x_mode_runs_alone_and_strips_metadata(
        self, pipeline, stained_graphic_path, tmp_path
    ):
        from opennomark.enhancer import Waifu2xSettings

        pipeline._enhancer = self.ScaleBy()
        output = tmp_path / "out.jpg"
        result, meta = pipeline.process_waifu2x(
            stained_graphic_path, str(output), Waifu2xSettings(scale=2, noise=3)
        )

        assert meta["status"] == "cleaned"
        assert meta["methods"] == ["waifu2x_art_noise3_2x"]
        assert result.size == (640, 480)
        assert b"ChatGPT" not in output.read_bytes()
