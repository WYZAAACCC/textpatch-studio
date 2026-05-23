from __future__ import annotations
import base64
import logging
import re
from typing import Optional

import requests

from backend.ocr_adapters.base import OCREngine, OCRDetectionResult, OCRRecognizeResult

logger = logging.getLogger(__name__)


class PaddleOCREngine(OCREngine):
    def __init__(self, api_url: str, token: str, **kwargs):
        self.api_url = api_url
        self.token = token
        self.use_doc_orientation_classify = kwargs.get("use_doc_orientation_classify", False)
        self.use_doc_unwarping = kwargs.get("use_doc_unwarping", False)
        self.use_chart_recognition = kwargs.get("use_chart_recognition", False)

    def _call_api(self, image_bytes: bytes, file_type: int = 1, max_retries: int = 3) -> dict:
        import time
        file_data = base64.b64encode(image_bytes).decode("ascii")

        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json",
        }

        payload = {
            "file": file_data,
            "fileType": file_type,
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useChartRecognition": self.use_chart_recognition,
        }

        for attempt in range(max_retries):
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = min(2 ** attempt, 8)
                logger.warning(f"OCR API rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"OCR API failed: {response.status_code}")

        raise RuntimeError(f"OCR API failed after {max_retries} retries")

    def detect(self, image_bytes: bytes, file_type: int = 1) -> list[OCRDetectionResult]:
        result = self._call_api(image_bytes, file_type)
        return self._parse_detection_results(result)

    def recognize(self, image_bytes: bytes, file_type: int = 1) -> list[OCRRecognizeResult]:
        result = self._call_api(image_bytes, file_type)
        return self._parse_recognition_results(result)

    def detect_and_recognize(
        self, image_bytes: bytes, file_type: int = 1
    ) -> list[OCRDetectionResult]:
        result = self._call_api(image_bytes, file_type)
        return self._parse_detection_results(result)

    def _parse_detection_results(self, api_result: dict) -> list[OCRDetectionResult]:
        results = []
        parsed_result = api_result.get("result", api_result)
        layout_results = parsed_result.get("layoutParsingResults", [])

        if not layout_results:
            return results

        for page_result in layout_results:
            pruned = page_result.get("prunedResult", {})
            parsing_res_list = pruned.get("parsing_res_list", [])

            if parsing_res_list:
                for block in parsing_res_list:
                    block_label = block.get("block_label", "")
                    block_content = block.get("block_content", "")
                    block_bbox = block.get("block_bbox", [])
                    block_polygon = block.get("block_polygon_points", [])

                    if not block_content.strip():
                        continue

                    if block_label == "image":
                        continue

                    text = self._clean_text(block_content)

                    if not text.strip():
                        continue

                    if block_bbox and len(block_bbox) >= 4:
                        x1, y1, x2, y2 = block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3]
                        polygon = block_polygon if len(block_polygon) >= 4 else [
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                        ]
                    else:
                        continue

                    confidence = 0.95 if block_label in ("paragraph_title", "figure_title") else 0.85

                    results.append(
                        OCRDetectionResult(
                            text=text,
                            confidence=confidence,
                            bbox=[x1, y1, x2, y2],
                            polygon=polygon,
                            angle=0.0,
                        )
                    )
                continue

            ocr_res = page_result.get("ocrResults", [])
            for ocr_item in ocr_res:
                if isinstance(ocr_item, dict):
                    text = ocr_item.get("text", "")
                    confidence = ocr_item.get("confidence", 0.0)
                    bbox = ocr_item.get("bbox", [])
                    polygon = ocr_item.get("polygon", [])
                    angle = ocr_item.get("angle", 0.0)
                    if text:
                        results.append(
                            OCRDetectionResult(
                                text=text,
                                confidence=confidence,
                                bbox=bbox,
                                polygon=polygon if polygon else bbox,
                                angle=angle,
                            )
                        )

            if not ocr_res:
                markdown_text = ""
                md_data = page_result.get("markdown", {})
                if isinstance(md_data, dict):
                    markdown_text = md_data.get("text", "")

                if markdown_text:
                    regions = self._extract_regions_from_markdown(markdown_text)
                    results.extend(regions)

        return results

    def _looks_like_formula(self, text: str) -> bool:
        """Check if text contains LaTeX formula notation that should be preserved."""
        if not text:
            return False
        # Strong signals: LaTeX math commands
        if re.search(
            r'\\frac|\\sqrt|\\sum|\\int|\\prod|\\alpha|\\beta|\\gamma|'
            r'\\delta|\\theta|\\sigma|\\phi|\\omega|\\partial|\\nabla|'
            r'\\infty|\\times|\\cdot|\\pm|\\leq|\\geq|\\neq|\\approx|'
            r'\\equiv|\\left|\\right|\\mathrm|\\mathbf|\\mathcal|\\mathbb',
            text
        ):
            return True
        # Math delimiters
        if re.search(r'\$\$.*?\$\$', text):
            return True
        if re.search(r'\$[^$]+\$', text):
            return True
        # Subscript/superscript in LaTeX notation
        if re.search(r'[_^]\{', text):
            return True
        # Greek Unicode
        if re.search(r'[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ]', text):
            return True
        return False

    def _clean_text(self, text: str) -> str:
        is_formula = self._looks_like_formula(text)
        if is_formula:
            # Only strip delimiters, preserve all LaTeX content
            text = re.sub(r'\$\$', '', text)
            text = re.sub(r'\$', '', text)
            text = re.sub(r'☐\s*', '', text)
            text = re.sub(r'\s{2,}', ' ', text)
            return text.strip()
        # Non-formula: aggressive cleaning
        text = re.sub(r'\$\$', '', text)
        text = re.sub(r'\$', '', text)
        text = re.sub(r'\\frac\{[^}]*\}\{[^}]*\}', '', text)
        text = re.sub(r'\\sqrt\{[^}]*\}', '', text)
        text = re.sub(r'\\left\(', '(', text)
        text = re.sub(r'\\right\)', ')', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'☐\s*', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _extract_regions_from_markdown(
        self, markdown_text: str, image_size: tuple = None
    ) -> list[OCRDetectionResult]:
        results = []
        if not markdown_text:
            return results

        lines = markdown_text.strip().split("\n")
        current_y = 0
        prev_heading_level = 0

        for line in lines:
            line = line.strip()
            if not line:
                current_y += 5
                continue

            heading_match = re.match(r'^(#{1,6})\s+', line)
            heading_level = len(heading_match.group(1)) if heading_match else 0

            text = self._clean_markdown_line(line)
            if not text:
                continue

            if heading_level > 0:
                if heading_level <= 2:
                    line_height = 36
                    font_size_est = 28
                elif heading_level <= 4:
                    line_height = 28
                    font_size_est = 22
                else:
                    line_height = 22
                    font_size_est = 16
                if prev_heading_level > 0 and heading_level != prev_heading_level:
                    current_y += 8
            else:
                line_height = 22
                font_size_est = 16

            prev_heading_level = heading_level

            char_width = font_size_est * 0.7
            estimated_width = len(text) * char_width

            bbox = [0, current_y, int(estimated_width), current_y + line_height]

            results.append(
                OCRDetectionResult(
                    text=text,
                    confidence=0.7,
                    bbox=bbox,
                    polygon=[
                        [0, current_y],
                        [int(estimated_width), current_y],
                        [int(estimated_width), current_y + line_height],
                        [0, current_y + line_height],
                    ],
                    angle=0.0,
                )
            )

            current_y += line_height + 4

        return results

    def _clean_markdown_line(self, line: str) -> str:
        is_formula = self._looks_like_formula(line)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", line)
        line = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", line)
        line = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"`{1,3}[^`]*`{1,3}", "", line)
        line = re.sub(r"---+", "", line)
        if is_formula:
            # Preserve LaTeX: only strip delimiters, keep commands intact
            line = re.sub(r"\$\$", "", line)
            line = re.sub(r"\$", "", line)
        else:
            line = re.sub(r"\$\$", "", line)
            line = re.sub(r"\$", "", line)
            line = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", line)
            line = re.sub(r"\\[a-zA-Z]+", "", line)
            line = re.sub(r"\{[^}]*\}", "", line)
            line = re.sub(r"\}", "", line)
        line = re.sub(r"\|", " ", line)
        line = re.sub(r"☐\s*", "", line)
        line = re.sub(r"P\s*E\s*_?\s*=\s*sin\s*\(", "PE = sin(", line)
        line = re.sub(r"P\s*E\s*_?\s*=\s*\(", "PE = (", line)
        line = re.sub(r"pos\s*\)", "pos)", line)
        line = re.sub(r"\s{2,}", " ", line)
        return line.strip()

    def _parse_recognition_results(self, api_result: dict) -> list[OCRRecognizeResult]:
        results = []
        parsed_result = api_result.get("result", api_result)
        layout_results = parsed_result.get("layoutParsingResults", [])

        for page_result in layout_results:
            pruned = page_result.get("prunedResult", {})
            parsing_res_list = pruned.get("parsing_res_list", [])

            if parsing_res_list:
                for block in parsing_res_list:
                    block_label = block.get("block_label", "")
                    block_content = block.get("block_content", "")

                    if not block_content.strip():
                        continue
                    if block_label == "image":
                        continue

                    text = self._clean_text(block_content)
                    if text.strip():
                        results.append(
                            OCRRecognizeResult(
                                text=text,
                                confidence=0.85,
                                source="paddle_layout",
                            )
                        )
                continue

            ocr_res = page_result.get("ocrResults", [])
            for ocr_item in ocr_res:
                if isinstance(ocr_item, dict):
                    text = ocr_item.get("text", "")
                    confidence = ocr_item.get("confidence", 0.0)
                    if text:
                        results.append(
                            OCRRecognizeResult(
                                text=text,
                                confidence=confidence,
                                source="paddle_ocr",
                            )
                        )

            if not ocr_res:
                markdown_text = ""
                md_data = page_result.get("markdown", {})
                if isinstance(md_data, dict):
                    markdown_text = md_data.get("text", "")

                if markdown_text:
                    lines = markdown_text.strip().split("\n")
                    for line in lines:
                        text = self._clean_markdown_line(line)
                        if text:
                            results.append(
                                OCRRecognizeResult(
                                    text=text,
                                    confidence=0.7,
                                    source="paddle_layout",
                                )
                            )

        return results
