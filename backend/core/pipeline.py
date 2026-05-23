from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.config import AppConfig, app_config
from backend.models.project import Project
from backend.models.region import TextRegion
from backend.models.style import TextStyle
from backend.models.render import RenderInfo
from backend.storage.project_store import ProjectStore
from backend.storage.file_store import FileStore
from backend.ocr_adapters.paddle import PaddleOCREngine
from backend.ocr_adapters.rapid import RapidOCREngine
from backend.llm_adapters.deepseek import DeepSeekClient
from backend.llm_adapters.mock import MockLLMClient
from backend.inpaint_adapters.opencv_inpaint import OpenCVInpaintEngine
from backend.renderer.font_manager import FontManager
from backend.core.detection import detect_text_regions
from backend.core.grouping import group_text_regions
from backend.core.ocr_engine import run_ocr
from backend.core.correction import correct_regions
from backend.core.risk_rules import detect_risk_flags
from backend.renderer.formula_renderer import classify_as_formula
from backend.core.masking import generate_masks
from backend.core.inpainting import inpaint_regions
from backend.core.style_inference import infer_style
from backend.core.rendering import render_region
from backend.core.compositing import composite_layers
from backend.core.exporter import export_project

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or app_config
        self.project_store = ProjectStore(self.config.storage.data_dir)
        self.file_store = FileStore(self.config.storage.data_dir)
        self.font_manager = FontManager(self.config.render.font_dirs)

        self._ocr_engine = None
        self._llm_client = None
        self._inpaint_engine = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            if self.config.ocr.api_url and self.config.ocr.token:
                logger.info("Using PaddleOCR HTTP API")
                self._ocr_engine = PaddleOCREngine(
                    api_url=self.config.ocr.api_url,
                    token=self.config.ocr.token,
                )
            elif RapidOCREngine.is_available():
                logger.info("PaddleOCR not configured, using local RapidOCR engine")
                self._ocr_engine = RapidOCREngine()
            else:
                raise RuntimeError(
                    "No OCR engine available.\n\n"
                    "Options:\n"
                    "  1. Configure PaddleOCR API: set TEXTPATCH_OCR_API_URL and "
                    "TEXTPATCH_OCR_TOKEN environment variables\n"
                    "  2. Install local OCR: pip install rapidocr-onnxruntime"
                )
        return self._ocr_engine

    @property
    def llm_client(self):
        if self._llm_client is None:
            if self.config.llm.api_key:
                self._llm_client = DeepSeekClient(
                    api_key=self.config.llm.api_key,
                    api_base=self.config.llm.api_base,
                    model=self.config.llm.model,
                    timeout=self.config.llm.timeout,
                    max_retries=self.config.llm.max_retries,
                    temperature=self.config.llm.temperature,
                    top_p=self.config.llm.top_p,
                )
            else:
                logger.warning("No LLM API key, using mock client")
                self._llm_client = MockLLMClient()
        return self._llm_client

    @property
    def inpaint_engine(self):
        if self._inpaint_engine is None:
            self._inpaint_engine = OpenCVInpaintEngine(
                method=self.config.inpaint.method,
                radius=self.config.inpaint.radius,
            )
        return self._inpaint_engine

    def create_project(self, name: str, image_path: Path) -> Project:
        img = Image.open(str(image_path))
        w, h = img.size

        project = Project.create(name=name, image_path="", width=w, height=h)

        self.project_store.save(project)

        original_path = self.file_store.save_original(
            project.id, image_path, image_path.name
        )
        project.original_image_path = str(original_path)

        self.project_store.save(project)
        return project

    def detect(self, project_id: str, detect_small_text: bool = True,
               scales: list = None, language: str = "zh-CN") -> Project:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        original_path = self.file_store.get_original(project_id)
        if not original_path:
            raise FileNotFoundError(f"Original image not found for project {project_id}")

        image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError(f"Failed to read image: {original_path}")

        regions = detect_text_regions(
            image,
            ocr_engine=self.ocr_engine,
            scales=scales,
            detect_small_text=detect_small_text,
            language=language,
        )

        grouped = group_text_regions(regions)

        for region in grouped:
            region.risk_flags = detect_risk_flags(region.final_text)
            is_formula, reason = classify_as_formula(region.final_text)
            if is_formula:
                region.is_formula = True
                region.latex_source = region.final_text

        project.regions = grouped
        self.project_store.save(project)
        return project

    def ocr(self, project_id: str) -> Project:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        original_path = self.file_store.get_original(project_id)
        if not original_path:
            raise FileNotFoundError(f"Original image not found")

        image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError(f"Failed to read image")

        project.regions = run_ocr(image, project.regions, self.ocr_engine)

        for region in project.regions:
            region.risk_flags = detect_risk_flags(region.final_text)
            is_formula, reason = classify_as_formula(region.final_text)
            if is_formula:
                region.is_formula = True
                region.latex_source = region.final_text

        self.project_store.save(project)
        return project

    def correct(self, project_id: str, auto_accept: bool = False,
                progress_callback=None) -> Project:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        project.regions = correct_regions(
            project.regions, self.llm_client, auto_accept,
            max_workers=self.config.llm.max_workers,
            progress_callback=progress_callback,
        )

        for region in project.regions:
            region.risk_flags = detect_risk_flags(region.final_text)
            # Ensure formula classification for regions LLM didn't flag
            if not region.is_formula:
                is_formula, reason = classify_as_formula(region.final_text)
                if is_formula:
                    region.is_formula = True
                    region.latex_source = region.final_text

        self.project_store.save(project)
        return project

    def inpaint(self, project_id: str, region_ids: list = None,
                simple_mode: bool = False) -> Project:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        original_path = self.file_store.get_original(project_id)
        if not original_path:
            raise FileNotFoundError(f"Original image not found")

        image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError(f"Failed to read image")

        regions_to_inpaint = project.regions
        if region_ids:
            regions_to_inpaint = [r for r in project.regions if r.id in region_ids]

        # Quality gate: skip low-confidence noise regions to avoid damaging the image.
        # MSER/SWT/tiny-text detectors produce many false positives on image
        # texture. Only keep regions that are either high-confidence (>= 0.5) or
        # have substantial OCR text (>= 5 chars). Everything else is noise.
        noise_sources = {
            "tiny_text_detection", "mser_detection", "mser_merged",
            "swt_detection", "swt_merged",
        }
        filtered = []
        skipped = 0
        for r in regions_to_inpaint:
            if r.source in noise_sources and r.confidence < 0.5:
                text_len = len(r.final_text.strip()) if r.final_text else 0
                if text_len < 5:
                    skipped += 1
                    continue
            filtered.append(r)
        if skipped:
            logger.info(f"Skipped {skipped} low-confidence noise regions from inpainting")

        clean_base, updated_regions = inpaint_regions(
            image, filtered, self.inpaint_engine, simple_mode=simple_mode
        )

        for i, region in enumerate(project.regions):
            for updated in updated_regions:
                if region.id == updated.id:
                    project.regions[i] = updated
                    break

        clean_base_pil = Image.fromarray(cv2.cvtColor(clean_base, cv2.COLOR_BGR2RGB))
        clean_path = self.file_store.save_clean_base(project_id, clean_base_pil)
        project.clean_base_path = str(clean_path)

        self.project_store.save(project)
        return project

    def render(self, project_id: str) -> Project:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        clean_path = self.file_store.get_clean_base(project_id)
        original_path = self.file_store.get_original(project_id)

        if clean_path:
            base_image = Image.open(str(clean_path))
        elif original_path:
            base_image = Image.open(str(original_path))
        else:
            raise FileNotFoundError("No base image available")

        region_layers = {}

        for region in project.regions:
            if region.status not in ("approved", "needs_review", "llm_corrected", "inpainted"):
                continue

            if not region.final_text:
                continue

            # Skip noise regions (same gate as inpaint step)
            noise_sources = {
                "tiny_text_detection", "mser_detection", "mser_merged",
                "swt_detection", "swt_merged",
            }
            if region.source in noise_sources and region.confidence < 0.5:
                text_len = len(region.final_text.strip())
                if text_len < 5:
                    continue

            if region.style is None:
                if original_path:
                    image_np = cv2.imread(str(original_path))
                else:
                    image_np = None
                if image_np is not None:
                    region.style = infer_style(image_np, region)
                else:
                    region.style = TextStyle()

            layer, render_info = render_region(
                region, base_image.size, self.font_manager
            )

            region.render = render_info

            if layer is not None:
                region_layers[region.id] = layer

                layer_path = self.file_store.save_region_image(
                    project_id, region.id, "render.png", layer
                )
                region.render.rendered_layer_path = str(layer_path)

            if render_info.overflow:
                region.status = "overflow"
            else:
                region.status = "rendered"

        final_image = composite_layers(base_image, project.regions, region_layers)
        final_path = self.file_store.save_final(project_id, final_image)
        project.final_image_path = str(final_path)

        self.project_store.save(project)
        return project

    def export_project(self, project_id: str, format: str = "png",
                       quality: int = 100, include_project_file: bool = True) -> Path:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        return export_project(
            project, self.project_store, self.file_store,
            format=format, quality=quality,
            include_project_file=include_project_file,
        )

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.project_store.load(project_id)

    def update_region(self, project_id: str, region_id: str,
                      final_text: str = None, style: dict = None,
                      status: str = None) -> Project:
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        for region in project.regions:
            if region.id == region_id:
                if final_text is not None:
                    region.final_text = final_text
                if style is not None:
                    if region.style is None:
                        region.style = TextStyle()
                    for key, value in style.items():
                        if hasattr(region.style, key):
                            if key == "color" and isinstance(value, list):
                                value = tuple(value)
                            elif key == "stroke_color" and isinstance(value, list):
                                value = tuple(value)
                            setattr(region.style, key, value)
                if status is not None:
                    region.status = status
                break

        self.project_store.save(project)
        return project

    def restore_region(self, project_id: str, x: int, y: int,
                       width: int, height: int) -> Project:
        """Restore original pixels within a bounding box on the clean image."""
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        original_path = self.file_store.get_original(project_id)
        clean_path = self.file_store.get_clean_base(project_id)
        if not original_path:
            raise FileNotFoundError("Original image not found")

        # Use clean_base if exists, otherwise original
        target_path = clean_path or original_path
        original_img = cv2.imread(str(original_path))
        target_img = cv2.imread(str(target_path))

        if original_img is None or target_img is None:
            raise ValueError("Failed to read images")

        h, w = target_img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + width), min(h, y + height)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid restore region")

        # Copy original pixels into target
        target_img[y1:y2, x1:x2] = original_img[y1:y2, x1:x2]

        clean_base_pil = Image.fromarray(
            cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        )
        clean_path_new = self.file_store.save_clean_base(project_id, clean_base_pil)
        project.clean_base_path = str(clean_path_new)

        self.project_store.save(project)
        return project

    def detect_region(self, project_id: str, x: int, y: int,
                      width: int, height: int, simple_mode: bool = False) -> Project:
        """Detect, OCR, and inpaint text within a bounding box.

        Used by the frontend's undetected-text post-processing box.
        """
        project = self.project_store.load(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        clean_path = self.file_store.get_clean_base(project_id)
        original_path = self.file_store.get_original(project_id)
        source_path = clean_path or original_path
        if not original_path or not source_path:
            raise FileNotFoundError("Image not found")

        image = cv2.imread(str(source_path))
        original_image = cv2.imread(str(original_path))
        if image is None:
            raise ValueError("Failed to read image")

        h, w = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + width), min(h, y + height)

        if x2 - x1 < 10 or y2 - y1 < 10:
            raise ValueError("Region too small for detection")

        # Crop the region
        crop = image[y1:y2, x1:x2]
        crop_orig = original_image[y1:y2, x1:x2]

        # Save crop as temp image for detection
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, crop)
            tmp_path = tmp.name

        try:
            # Detect text in crop
            regions = detect_text_regions(crop, self.ocr_engine, language="zh-CN")
            if not regions:
                return project

            # Group and OCR
            regions = group_text_regions(regions)
            regions = run_ocr(crop, regions, self.ocr_engine)

            # Mark approved so inpaint_regions will process them
            for r in regions:
                r.status = "approved"

            # Adjust coordinates from crop-local to full-image
            for r in regions:
                if r.bbox:
                    r.bbox[0] += x1
                    r.bbox[1] += y1
                    r.bbox[2] += x1
                    r.bbox[3] += y1
                if r.polygon:
                    r.polygon = [[px + x1, py + y1] for px, py in r.polygon]

            # Inpaint on the full image
            clean_base, updated_regions = inpaint_regions(
                image, regions, self.inpaint_engine, simple_mode=simple_mode
            )

            # Save clean base
            clean_base_pil = Image.fromarray(
                cv2.cvtColor(clean_base, cv2.COLOR_BGR2RGB)
            )
            clean_path_new = self.file_store.save_clean_base(
                project_id, clean_base_pil
            )
            project.clean_base_path = str(clean_path_new)

            # Add new regions to project
            for r in updated_regions:
                r.status = "approved"
            project.regions.extend(updated_regions)

        finally:
            try:
                import os
                os.unlink(tmp_path)
            except Exception:
                pass

        self.project_store.save(project)
        return project

    def list_projects(self) -> list[dict]:
        return self.project_store.list_projects()
