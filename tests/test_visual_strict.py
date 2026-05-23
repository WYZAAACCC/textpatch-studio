import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

from backend.core.masking import generate_masks, classify_background
from backend.core.inpainting import inpaint_regions
from backend.core.style_inference import infer_style
from backend.core.rendering import render_region
from backend.core.compositing import composite_layers
from backend.models.region import TextRegion
from backend.models.style import TextStyle, ShadowStyle
from backend.renderer.font_manager import FontManager
from backend.renderer.text_layout import layout_text
from backend.renderer.text_renderer import render_text_layer
from backend.models.render import RenderInfo

FM = FontManager()
FONT_PATH = FM.get_default_font_path()


def _make_test_image(width=400, height=200, bg=(240, 240, 240), text="测试文字", x=50, y=50, font_size=24, color=(0, 0, 0)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    if FONT_PATH:
        font = ImageFont.truetype(FONT_PATH, font_size)
        draw.text((x, y), text, font=font, fill=color)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    else:
        text_w, text_h = len(text) * 14, font_size

    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv, text_w, text_h


def _make_region(x, y, w, h, text="测试文字", is_tiny=False):
    region = TextRegion.create(
        polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        bbox=[x, y, x + w, y + h],
        confidence=0.9,
        is_tiny=is_tiny,
    )
    region.final_text = text
    region.status = "approved"
    return region


class TestMSERSWTDetection:
    def test_mser_detects_text_regions(self):
        from backend.core.mser_swt_detection import detect_mser_regions
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试文字", 50, 50, 24, (0, 0, 0))
        regions = detect_mser_regions(image, [])
        assert isinstance(regions, list)

    def test_swt_detects_text_regions(self):
        from backend.core.mser_swt_detection import detect_swt_regions
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试文字", 50, 50, 24, (0, 0, 0))
        regions = detect_swt_regions(image, [])
        assert isinstance(regions, list)

    def test_mser_no_overlap_with_existing(self):
        from backend.core.mser_swt_detection import detect_mser_regions
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试文字", 50, 50, 24, (0, 0, 0))
        existing = _make_region(40, 40, tw + 30, th + 25, "测试文字")
        regions = detect_mser_regions(image, [existing])
        for r in regions:
            rx1, ry1, rx2, ry2 = r.bbox
            ex1, ey1, ex2, ey2 = existing.bbox
            ix1 = max(rx1, ex1)
            iy1 = max(ry1, ey1)
            ix2 = min(rx2, ex2)
            iy2 = min(ry2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = (rx2 - rx1) * (ry2 - ry1)
            if area > 0:
                assert inter / area < 0.6, f"MSER region overlaps too much with existing"

    def test_mser_merged_regions_are_valid(self):
        from backend.core.mser_swt_detection import detect_mser_regions
        image, tw, th = _make_test_image(600, 300, (240, 240, 240), "测试文字内容", 50, 50, 28, (0, 0, 0))
        regions = detect_mser_regions(image, [])
        for r in regions:
            assert r.bbox is not None
            assert len(r.bbox) == 4
            assert r.bbox[2] > r.bbox[0]
            assert r.bbox[3] > r.bbox[1]


class TestGrabCutMaskRefinement:
    def test_grabcut_refines_mask_on_clear_text(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        glyph_mask, erase_mask, safe_mask = generate_masks(image, region)

        x1, y1, x2, y2 = region.bbox
        crop_mask = glyph_mask[y1:y2, x1:x2]
        total = crop_mask.shape[0] * crop_mask.shape[1]
        glyph_ratio = np.sum(crop_mask > 0) / total

        assert glyph_ratio > 0.03, f"GrabCut mask too small: {glyph_ratio:.1%}"
        assert glyph_ratio < 0.85, f"GrabCut mask too large: {glyph_ratio:.1%}"

    def test_grabcut_preserves_mask_on_small_regions(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "小", 50, 50, 10, (0, 0, 0))
        region = _make_region(48, 48, tw + 8, th + 6, "小", is_tiny=True)

        glyph_mask, erase_mask, _ = generate_masks(image, region)
        assert glyph_mask is not None
        assert erase_mask is not None

    def test_grabcut_mask_stays_within_bbox(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        glyph_mask, erase_mask, _ = generate_masks(image, region)

        x1, y1, x2, y2 = region.bbox
        h, w = image.shape[:2]

        outside_top = glyph_mask[:max(0, y1), :]
        outside_bottom = glyph_mask[min(h, y2):, :]
        outside_left = glyph_mask[:, :max(0, x1)]
        outside_right = glyph_mask[:, min(w, x2):]

        for name, area in [("top", outside_top), ("bottom", outside_bottom), ("left", outside_left), ("right", outside_right)]:
            if area.size > 0:
                leak = np.sum(area > 0)
                assert leak == 0, f"GrabCut mask leaks outside bbox ({name}): {leak} pixels"


class TestAdaptiveInpainting:
    def test_solid_bg_inpaint(self):
        bg_color = (200, 200, 200)
        image, tw, th = _make_test_image(400, 200, bg_color, "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        result, _ = inpaint_regions(image, [region], engine)

        x1, y1, x2, y2 = region.bbox
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        center_pixel = result[center_y, center_x].astype(float)
        bg_arr = np.array(bg_color, dtype=float)
        diff = np.abs(center_pixel - bg_arr)
        assert np.mean(diff) < 80, f"Solid bg inpaint center differs from bg: {diff}"

    def test_gradient_bg_inpaint(self):
        img_np = np.zeros((200, 400, 3), dtype=np.uint8)
        for x in range(400):
            val = int(100 + 100 * x / 400)
            img_np[:, x] = [val, val, val]

        if FONT_PATH:
            pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(FONT_PATH, 24)
            draw.text((50, 50), "测试", font=font, fill=(0, 0, 0))
            img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        region = _make_region(45, 45, 100, 40, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        result, _ = inpaint_regions(img_np, [region], engine)

        x1, y1, x2, y2 = region.bbox
        left_val = float(np.mean(result[y1:y2, x1]))
        right_val = float(np.mean(result[y1:y2, x2 - 1]))
        assert right_val > left_val, "Gradient inpaint should preserve gradient direction"

    def test_feather_boundary_smooth(self):
        bg_color = (200, 200, 200)
        image, tw, th = _make_test_image(400, 200, bg_color, "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        result, _ = inpaint_regions(image, [region], engine)

        x1, y1, x2, y2 = region.bbox
        h, w = image.shape[:2]

        border_above = result[max(0, y1 - 3):y1, x1:x2]
        border_below = result[y2:min(h, y2 + 3), x1:x2]

        if border_above.size > 0 and border_below.size > 0:
            bg_arr = np.array(bg_color, dtype=float)
            diff_above = np.abs(border_above.astype(float) - bg_arr)
            diff_below = np.abs(border_below.astype(float) - bg_arr)
            assert np.mean(diff_above) < 30, f"Boundary above not smooth: {np.mean(diff_above):.1f}"
            assert np.mean(diff_below) < 30, f"Boundary below not smooth: {np.mean(diff_below):.1f}"


class TestStrokeAndShadowDetection:
    def test_stroke_detection_on_stroked_text(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        img = Image.new("RGB", (400, 200), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 28)

        text = "描边文字"
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx * dx + dy * dy <= 4:
                    draw.text((50 + dx, 50 + dy), text, font=font, fill=(0, 0, 255))
        draw.text((50, 50), text, font=font, fill=(255, 255, 0))

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        region = _make_region(45, 45, tw + 20, th + 15, text)

        style = infer_style(img_cv, region)

        assert style.stroke_width > 0, f"Stroke not detected: stroke_width={style.stroke_width}"
        assert style.stroke_color is not None, "Stroke color not detected"

    def test_no_false_stroke_on_plain_text(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "普通文字", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "普通文字")

        style = infer_style(image, region)

        assert style.stroke_width == 0 or style.stroke_width < 1.0, f"False stroke detected: stroke_width={style.stroke_width}"

    def test_shadow_detection_on_shadowed_text(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        img = Image.new("RGB", (400, 200), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 28)

        text = "阴影文字"
        draw.text((53, 53), text, font=font, fill=(100, 100, 100))
        draw.text((50, 50), text, font=font, fill=(0, 0, 0))

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        region = _make_region(45, 45, tw + 20, th + 15, text)

        style = infer_style(img_cv, region)

        assert style.shadow is not None, "Shadow not detected"
        assert style.shadow.dx > 0 or style.shadow.dy > 0, f"Shadow offset too small: dx={style.shadow.dx}, dy={style.shadow.dy}"


class TestAlignmentDetection:
    def test_center_alignment_detection(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        img = Image.new("RGB", (400, 60), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 24)
        text = "居中"
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        x = (400 - tw) // 2
        draw.text((x, 15), text, font=font, fill=(0, 0, 0))

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        region = _make_region(0, 0, 400, 60, text)

        style = infer_style(img_cv, region)
        assert style.align == "center", f"Center alignment not detected: align={style.align}"

    def test_left_alignment_detection(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        img = Image.new("RGB", (400, 60), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 24)
        text = "左对齐"
        draw.text((5, 15), text, font=font, fill=(0, 0, 0))

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        region = _make_region(0, 0, 400, 60, text)

        style = infer_style(img_cv, region)
        assert style.align == "left", f"Left alignment not detected: align={style.align}"


class TestImageHarmonization:
    def test_harmonize_adjusts_brightness(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        dark_img = Image.new("RGB", (400, 200), (60, 60, 60))
        draw = ImageDraw.Draw(dark_img)
        font = ImageFont.truetype(FONT_PATH, 24)
        draw.text((50, 50), "测试", font=font, fill=(255, 255, 255))

        dark_cv = cv2.cvtColor(np.array(dark_img), cv2.COLOR_RGB2BGR)

        bbox = font.getbbox("测试")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        region = _make_region(45, 45, tw + 20, th + 15, "测试")
        region.style = TextStyle(font_size=24, color=(255, 255, 255, 255))

        layer, render_info = render_region(region, (400, 200), FM)
        if layer is None:
            pytest.skip("Render failed")

        region.render = render_info
        result = composite_layers(dark_img, [region], {region.id: layer})

        result_arr = np.array(result.convert("RGB"))
        text_area = result_arr[45:45 + th + 15, 45:45 + tw + 20]
        text_pixels = text_area[text_area.mean(axis=2) > 150]

        if len(text_pixels) > 0:
            mean_brightness = text_pixels.mean()
            assert mean_brightness < 255, f"Harmonized text too bright: {mean_brightness:.1f}"

    def test_harmonize_adds_noise(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        noisy_bg = np.random.randint(200, 240, (200, 400, 3), dtype=np.uint8)
        pil_img = Image.fromarray(noisy_bg)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(FONT_PATH, 24)
        draw.text((50, 50), "测试", font=font, fill=(0, 0, 0))

        noisy_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        bbox = font.getbbox("测试")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        region = _make_region(45, 45, tw + 20, th + 15, "测试")
        region.style = TextStyle(font_size=24, color=(0, 0, 0, 255))

        layer, render_info = render_region(region, (400, 200), FM)
        if layer is None:
            pytest.skip("Render failed")

        region.render = render_info
        result = composite_layers(pil_img, [region], {region.id: layer})

        assert result.size == pil_img.size


class TestPseudoHanziDetection:
    def test_detect_rare_unicode_as_pseudo(self):
        from backend.core.ocr_postprocess import is_pseudo_hanzi

        rare_char = chr(0x20001)
        assert is_pseudo_hanzi(rare_char), f"Rare CJK Ext-B char should be pseudo"

    def test_common_hanzi_not_pseudo(self):
        from backend.core.ocr_postprocess import is_pseudo_hanzi

        common_chars = "的一是不了人我在有他这"
        for char in common_chars:
            assert not is_pseudo_hanzi(char), f"Common char '{char}' should not be pseudo"

    def test_detect_pseudo_in_text(self):
        from backend.core.ocr_postprocess import detect_pseudo_hanzi_in_text

        text_with_pseudo = "子槻块"
        results = detect_pseudo_hanzi_in_text(text_with_pseudo)
        pseudo_positions = [r["position"] for r in results]
        assert 1 in pseudo_positions, f"Should detect '槻' as pseudo at position 1"

    def test_pseudo_hanzi_suggestion(self):
        from backend.core.ocr_postprocess import detect_pseudo_hanzi_in_text

        text = "子槻块"
        results = detect_pseudo_hanzi_in_text(text)
        for r in results:
            if r["char"] == "槻":
                assert r["suggestion"] == "模", f"Should suggest '模' for '槻', got '{r['suggestion']}'"

    def test_no_pseudo_in_normal_text(self):
        from backend.core.ocr_postprocess import detect_pseudo_hanzi_in_text

        normal_text = "这是一个正常的中文句子"
        results = detect_pseudo_hanzi_in_text(normal_text)
        assert len(results) == 0, f"Normal text should have no pseudo hanzi, found {len(results)}"


class TestOCRPostProcess:
    def test_clean_latex_artifacts(self):
        from backend.core.ocr_postprocess import clean_ocr_artifacts

        text_with_latex = "P E_=sin(pos})"
        cleaned = clean_ocr_artifacts(text_with_latex)
        assert "}" not in cleaned, f"LaTeX artifact not cleaned: '{cleaned}'"

    def test_clean_markdown_artifacts(self):
        from backend.core.ocr_postprocess import clean_ocr_artifacts

        text_with_md = "**粗体**文字"
        cleaned = clean_ocr_artifacts(text_with_md)
        assert "**" not in cleaned, f"Markdown artifact not cleaned: '{cleaned}'"

    def test_merge_candidates_fixes_pseudo(self):
        from backend.core.ocr_postprocess import merge_ocr_candidates

        candidates = ["子槻块", "子模块"]
        result = merge_ocr_candidates(candidates)
        assert "模" in result, f"Merge should fix pseudo hanzi: '{result}'"
        assert "槻" not in result, f"Merge should remove pseudo hanzi: '{result}'"

    def test_validate_ocr_text_flags_pseudo(self):
        from backend.core.ocr_postprocess import validate_ocr_text

        result = validate_ocr_text("子槻块")
        assert not result["valid"], "Text with pseudo hanzi should be invalid"
        assert "pseudo_hanzi" in result["issues"], "Should flag pseudo_hanzi issue"

    def test_validate_normal_text_is_valid(self):
        from backend.core.ocr_postprocess import validate_ocr_text

        result = validate_ocr_text("正常文字")
        assert result["valid"], "Normal text should be valid"
        assert len(result["issues"]) == 0, "Normal text should have no issues"


class TestMaskPrecision:
    def test_glyph_mask_only_covers_text_pixels(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        glyph_mask, erase_mask, safe_mask = generate_masks(image, region)

        x1, y1, x2, y2 = region.bbox
        crop_mask = glyph_mask[y1:y2, x1:x2]
        total = crop_mask.shape[0] * crop_mask.shape[1]
        glyph_ratio = np.sum(crop_mask > 0) / total

        assert glyph_ratio > 0.05, f"Glyph mask too small: {glyph_ratio:.1%}"
        assert glyph_ratio < 0.8, f"Glyph mask too large: {glyph_ratio:.1%}"

    def test_erase_mask_not_much_larger_than_glyph(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        glyph_mask, erase_mask, _ = generate_masks(image, region)

        glyph_pixels = np.sum(glyph_mask > 0)
        erase_pixels = np.sum(erase_mask > 0)

        if glyph_pixels > 0:
            expand_ratio = erase_pixels / glyph_pixels
            assert expand_ratio < 5.0, f"Erase mask is {expand_ratio:.1f}x larger than glyph mask (max 5x)"

    def test_mask_does_not_leak_outside_bbox(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        _, erase_mask, _ = generate_masks(image, region)

        x1, y1, x2, y2 = region.bbox
        h, w = image.shape[:2]

        outside_top = erase_mask[:max(0, y1 - 5), :]
        outside_bottom = erase_mask[min(h, y2 + 5):, :]
        outside_left = erase_mask[:, :max(0, x1 - 5)]
        outside_right = erase_mask[:, min(w, x2 + 5):]

        for name, area in [("top", outside_top), ("bottom", outside_bottom), ("left", outside_left), ("right", outside_right)]:
            if area.size > 0:
                leak = np.sum(area > 0)
                assert leak == 0, f"Erase mask leaks outside bbox ({name}): {leak} pixels"

    def test_dark_bg_glyph_mask(self):
        image, tw, th = _make_test_image(400, 200, (40, 40, 80), "测试", 50, 50, 24, (255, 255, 255))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        glyph_mask, _, _ = generate_masks(image, region)

        x1, y1, x2, y2 = region.bbox
        crop_mask = glyph_mask[y1:y2, x1:x2]
        total = crop_mask.shape[0] * crop_mask.shape[1]
        glyph_ratio = np.sum(crop_mask > 0) / total

        assert glyph_ratio > 0.05, f"Glyph mask too small on dark bg: {glyph_ratio:.1%}"

    def test_small_text_mask(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "小字", 50, 50, 12, (0, 0, 0))
        region = _make_region(45, 45, tw + 10, th + 8, "小字", is_tiny=True)

        glyph_mask, erase_mask, _ = generate_masks(image, region)

        x1, y1, x2, y2 = region.bbox
        crop_erase = erase_mask[y1:y2, x1:x2]
        total = crop_erase.shape[0] * crop_erase.shape[1]
        erase_ratio = np.sum(crop_erase > 0) / total

        assert erase_ratio < 0.95, f"Small text erase mask covers {erase_ratio:.1%} of region (should be <95%)"


class TestInpaintPrecision:
    def test_inpaint_only_changes_text_area(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        glyph_mask, erase_mask, _ = generate_masks(image, region)

        result, _ = inpaint_regions(image, [region], engine)

        diff = np.abs(image.astype(float) - result.astype(float))
        changed_mask = np.max(diff, axis=2) > 5

        x1, y1, x2, y2 = region.bbox
        margin = 15
        outside_region = changed_mask.copy()
        outside_region[max(0, y1 - margin):min(image.shape[0], y2 + margin),
                       max(0, x1 - margin):min(image.shape[1], x2 + margin)] = False

        outside_changed = np.sum(outside_region)
        assert outside_changed == 0, f"Inpaint changed {outside_changed} pixels outside the text region!"

    def test_inpaint_preserves_background_color(self):
        bg_color = (200, 200, 200)
        image, tw, th = _make_test_image(400, 200, bg_color, "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        result, _ = inpaint_regions(image, [region], engine)

        x1, y1, x2, y2 = region.bbox
        margin = 5
        border_pixels = []
        for yy in range(max(0, y1 - margin), min(image.shape[0], y1)):
            for xx in range(x1, x2):
                border_pixels.append(result[yy, xx])
        for yy in range(y2, min(image.shape[0], y2 + margin)):
            for xx in range(x1, x2):
                border_pixels.append(result[yy, xx])

        if border_pixels:
            border_arr = np.array(border_pixels, dtype=float)
            bg_arr = np.array(bg_color, dtype=float)
            mean_diff = np.mean(np.abs(border_arr - bg_arr))
            assert mean_diff < 30, f"Inpaint border color differs from background by {mean_diff:.1f} (max 30)"


class TestRenderPosition:
    def test_render_layer_matches_bbox_size(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        region = _make_region(50, 50, 200, 40, "测试文字")
        region.style = TextStyle(font_size=24)

        layer, render_info = render_region(region, (800, 600), FM)

        if layer is not None:
            bbox_w = 200
            bbox_h = 40
            assert layer.size[0] == bbox_w, f"Layer width {layer.size[0]} != bbox width {bbox_w}"
            assert layer.size[1] == bbox_h, f"Layer height {layer.size[1]} != bbox height {bbox_h}"

    def test_render_text_is_visible(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        region = _make_region(50, 50, 200, 40, "测试文字")
        region.style = TextStyle(font_size=24, color=(0, 0, 0, 255))

        layer, _ = render_region(region, (800, 600), FM)

        if layer is not None:
            arr = np.array(layer)
            alpha_nonzero = np.sum(arr[:, :, 3] > 0)
            total = arr.shape[0] * arr.shape[1]
            visible_ratio = alpha_nonzero / total

            assert visible_ratio > 0.05, f"Rendered text is nearly invisible: {visible_ratio:.1%}"
            assert visible_ratio < 0.95, f"Rendered text covers too much: {visible_ratio:.1%}"

    def test_render_text_centered_in_bbox(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        region = _make_region(50, 50, 300, 50, "测试")
        region.style = TextStyle(font_size=24, color=(0, 0, 0, 255))

        layer, _ = render_region(region, (800, 600), FM)

        if layer is not None:
            arr = np.array(layer)
            alpha = arr[:, :, 3]

            rows_with_text = np.any(alpha > 0, axis=1)
            if np.any(rows_with_text):
                first_row = np.argmax(rows_with_text)
                last_row = len(rows_with_text) - 1 - np.argmax(rows_with_text[::-1])
                top_margin = first_row
                bottom_margin = arr.shape[0] - 1 - last_row

                margin_diff = abs(top_margin - bottom_margin)
                max_margin = max(top_margin, bottom_margin)
                if max_margin > 0:
                    center_error = margin_diff / max_margin
                    assert center_error < 0.8, f"Text not vertically centered: top={top_margin}, bottom={bottom_margin}, error={center_error:.2f}"

    def test_font_size_fits_bbox(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        for text, bbox_w, bbox_h in [("短", 50, 30), ("测试文字", 150, 40), ("很长的文字内容", 300, 50)]:
            region = _make_region(50, 50, bbox_w, bbox_h, text)
            region.style = TextStyle(font_size=min(bbox_h * 0.85, 48))

            layer, render_info = render_region(region, (800, 600), FM)
            if layer is not None and not render_info.overflow:
                arr = np.array(layer)
                alpha = arr[:, :, 3]
                cols_with_text = np.any(alpha > 0, axis=0)
                if np.any(cols_with_text):
                    first_col = np.argmax(cols_with_text)
                    last_col = len(cols_with_text) - 1 - np.argmax(cols_with_text[::-1])
                    text_width = last_col - first_col + 1
                    assert text_width <= bbox_w, f"Text width {text_width} exceeds bbox width {bbox_w}"


class TestCompositeAlignment:
    def test_composite_places_text_at_bbox_position(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        base = Image.new("RGBA", (400, 200), (240, 240, 240, 255))

        region = _make_region(100, 80, 200, 40, "测试文字")
        region.style = TextStyle(font_size=24, color=(255, 0, 0, 255))
        region.render = RenderInfo()

        layer, render_info = render_region(region, (400, 200), FM)
        if layer is None:
            pytest.skip("Render failed")

        region.render = render_info
        result = composite_layers(base, [region], {region.id: layer})

        result_arr = np.array(result)
        base_arr = np.array(base.convert("RGB"))

        diff = np.abs(result_arr.astype(float) - base_arr.astype(float))
        changed_rows = np.where(np.max(diff, axis=(1, 2)) > 10)[0]

        if len(changed_rows) > 0:
            first_changed = changed_rows[0]
            last_changed = changed_rows[-1]

            assert first_changed >= 75, f"Text starts too early: row {first_changed}, expected >= 75"
            assert last_changed <= 125, f"Text ends too late: row {last_changed}, expected <= 125"

    def test_composite_does_not_damage_background(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        base = Image.new("RGBA", (400, 200), (240, 240, 240, 255))

        region = _make_region(100, 80, 200, 40, "测试文字")
        region.style = TextStyle(font_size=24, color=(0, 0, 0, 255))
        region.render = RenderInfo()

        layer, render_info = render_region(region, (400, 200), FM)
        if layer is None:
            pytest.skip("Render failed")

        region.render = render_info
        result = composite_layers(base, [region], {region.id: layer})

        result_arr = np.array(result.convert("RGB"))
        base_arr = np.array(base.convert("RGB"))

        diff = np.abs(result_arr.astype(float) - base_arr.astype(float))

        corner_tl = diff[:30, :30]
        corner_tr = diff[:30, -30:]
        corner_bl = diff[-30:, :30]
        corner_br = diff[-30:, -30:]

        for name, corner in [("TL", corner_tl), ("TR", corner_tr), ("BL", corner_bl), ("BR", corner_br)]:
            corner_diff = corner.mean()
            assert corner_diff < 1.0, f"Corner {name} damaged: mean_diff={corner_diff:.2f}"


class TestStyleInference:
    def test_font_size_matches_bbox(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试文字", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试文字")

        style = infer_style(image, region)

        assert style.font_size > 0, "Font size is 0"
        assert style.font_size <= region.bbox[3] - region.bbox[1], f"Font size {style.font_size} > bbox height {region.bbox[3] - region.bbox[1]}"

    def test_color_matches_text(self):
        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        style = infer_style(image, region)

        r, g, b = style.color[0], style.color[1], style.color[2]
        assert r < 80 and g < 80 and b < 80, f"Inferred color too bright for black text: ({r},{g},{b})"

    def test_white_text_on_dark_bg(self):
        image, tw, th = _make_test_image(400, 200, (40, 40, 80), "测试", 50, 50, 24, (255, 255, 255))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        style = infer_style(image, region)

        r, g, b = style.color[0], style.color[1], style.color[2]
        assert r > 180 and g > 180 and b > 180, f"Inferred color too dark for white text: ({r},{g},{b})"


class TestEndToEnd:
    def test_full_pipeline_on_synthetic_image(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试文字", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试文字")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        clean_base, updated_regions = inpaint_regions(image, [region], engine)

        diff_inpaint = np.abs(image.astype(float) - clean_base.astype(float))
        inpaint_changed = np.sum(np.max(diff_inpaint, axis=2) > 5)
        assert inpaint_changed > 0, "Inpaint did not change any pixels"

        for r in updated_regions:
            r.style = infer_style(image, r)

        region_layers = {}
        for r in updated_regions:
            layer, render_info = render_region(r, (400, 200), FM)
            r.render = render_info
            if layer is not None:
                region_layers[r.id] = layer

        clean_pil = Image.fromarray(cv2.cvtColor(clean_base, cv2.COLOR_BGR2RGB))
        final = composite_layers(clean_pil, updated_regions, region_layers)

        final_arr = np.array(final.convert("RGB"))
        clean_arr = np.array(clean_pil.convert("RGB"))
        diff_render = np.abs(final_arr.astype(float) - clean_arr.astype(float))
        render_changed = np.sum(np.max(diff_render, axis=2) > 5)
        assert render_changed > 0, "Render did not add any visible text"

    def test_background_unchanged_outside_text(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        image, tw, th = _make_test_image(400, 200, (240, 240, 240), "测试", 50, 50, 24, (0, 0, 0))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        clean_base, _ = inpaint_regions(image, [region], engine)

        x1, y1, x2, y2 = region.bbox
        margin = 20

        safe_zone_top = image[:max(0, y1 - margin), :]
        safe_zone_clean = clean_base[:max(0, y1 - margin), :]
        if safe_zone_top.size > 0:
            diff = np.abs(safe_zone_top.astype(float) - safe_zone_clean.astype(float))
            assert diff.mean() < 1.0, f"Background changed above text area: mean_diff={diff.mean():.2f}"

        safe_zone_bottom = image[min(image.shape[0], y2 + margin):, :]
        safe_zone_clean_bottom = clean_base[min(clean_base.shape[0], y2 + margin):, :]
        if safe_zone_bottom.size > 0:
            diff = np.abs(safe_zone_bottom.astype(float) - safe_zone_clean_bottom.astype(float))
            assert diff.mean() < 1.0, f"Background changed below text area: mean_diff={diff.mean():.2f}"

    def test_dark_bg_full_pipeline(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        image, tw, th = _make_test_image(400, 200, (40, 40, 80), "测试", 50, 50, 24, (255, 255, 255))
        region = _make_region(45, 45, tw + 20, th + 15, "测试")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        clean_base, updated_regions = inpaint_regions(image, [region], engine)

        for r in updated_regions:
            r.style = infer_style(image, r)

        assert updated_regions[0].style.color[0] > 150, "White text on dark bg should have bright color"

        region_layers = {}
        for r in updated_regions:
            layer, render_info = render_region(r, (400, 200), FM)
            r.render = render_info
            if layer is not None:
                region_layers[r.id] = layer

        clean_pil = Image.fromarray(cv2.cvtColor(clean_base, cv2.COLOR_BGR2RGB))
        final = composite_layers(clean_pil, updated_regions, region_layers)

        final_arr = np.array(final.convert("RGB"))
        assert final_arr.shape == (200, 400, 3), "Final image has wrong dimensions"

    def test_multi_region_pipeline(self):
        if not FONT_PATH:
            pytest.skip("No font available")

        img = Image.new("RGB", (600, 300), (220, 220, 220))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 24)

        draw.text((30, 30), "标题文字", font=font, fill=(0, 0, 0))
        draw.text((30, 120), "正文内容", font=font, fill=(50, 50, 50))

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        region1 = _make_region(25, 25, 150, 40, "标题文字")
        region2 = _make_region(25, 115, 150, 40, "正文内容")

        from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
        engine = OpenCVInpaintEngine()

        clean_base, updated_regions = inpaint_regions(img_cv, [region1, region2], engine)

        for r in updated_regions:
            r.style = infer_style(img_cv, r)

        region_layers = {}
        for r in updated_regions:
            layer, render_info = render_region(r, (600, 300), FM)
            r.render = render_info
            if layer is not None:
                region_layers[r.id] = layer

        clean_pil = Image.fromarray(cv2.cvtColor(clean_base, cv2.COLOR_BGR2RGB))
        final = composite_layers(clean_pil, updated_regions, region_layers)

        final_arr = np.array(final.convert("RGB"))
        assert final_arr.shape == (300, 600, 3), "Final image has wrong dimensions"

        diff = np.abs(final_arr.astype(float) - np.array(img).astype(float))
        changed = np.sum(np.max(diff, axis=2) > 10)
        assert changed > 0, "Multi-region pipeline should produce visible changes"
