from __future__ import annotations
import logging
from typing import Optional

import numpy as np

from backend.models.region import TextRegion

logger = logging.getLogger(__name__)


def group_text_regions(regions: list[TextRegion]) -> list[TextRegion]:
    if not regions:
        return []

    lines = _group_into_lines(regions)
    result = []

    for line in lines:
        if len(line) == 1:
            result.append(line[0])
        else:
            line.sort(key=lambda r: r.bbox[0] if r.bbox else 0)

            words = _split_into_words(line)

            for word in words:
                if len(word) == 1:
                    result.append(word[0])
                else:
                    result.append(_merge_regions(word))

    return result


def _group_into_lines(regions: list[TextRegion]) -> list[list[TextRegion]]:
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]) if r.bbox else (0, 0))

    lines = []
    current_line = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        if not region.bbox or not current_line[-1].bbox:
            current_line.append(region)
            continue

        prev = current_line[-1]
        y_center_curr = (region.bbox[1] + region.bbox[3]) / 2
        y_center_prev = (prev.bbox[1] + prev.bbox[3]) / 2
        prev_height = prev.bbox[3] - prev.bbox[1]
        curr_height = region.bbox[3] - region.bbox[1]

        avg_height = (prev_height + curr_height) / 2
        if avg_height <= 0:
            avg_height = max(prev_height, curr_height, 1)

        if abs(y_center_curr - y_center_prev) < avg_height * 0.5:
            angle_diff = abs(region.angle - prev.angle)
            if angle_diff < 8:
                current_line.append(region)
                continue

        lines.append(current_line)
        current_line = [region]

    lines.append(current_line)
    return lines


def _split_into_words(line: list[TextRegion]) -> list[list[TextRegion]]:
    if len(line) <= 1:
        return [line]

    gaps = []
    for i in range(len(line) - 1):
        if line[i].bbox and line[i + 1].bbox:
            gap = line[i + 1].bbox[0] - line[i].bbox[2]
            gaps.append(gap)
        else:
            gaps.append(0)

    if not gaps:
        return [line]

    heights = [r.bbox[3] - r.bbox[1] for r in line if r.bbox]
    avg_height = np.median(heights) if heights else 20

    gap_threshold = avg_height * 1.5

    groups = []
    current_group = [line[0]]

    for i in range(1, len(line)):
        if i - 1 < len(gaps) and gaps[i - 1] > gap_threshold:
            groups.append(current_group)
            current_group = [line[i]]
        else:
            current_group.append(line[i])

    groups.append(current_group)
    return groups


def _merge_regions(regions: list[TextRegion]) -> TextRegion:
    bboxes = [r.bbox for r in regions if r.bbox]
    if not bboxes:
        return regions[0]

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)

    merged = TextRegion.create(
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=[x1, y1, x2, y2],
        angle=sum(r.angle for r in regions) / len(regions),
        source="grouped",
        confidence=max(r.confidence for r in regions),
        is_tiny=any(r.is_tiny for r in regions),
    )

    texts = [r.final_text for r in regions if r.final_text]
    merged.final_text = " ".join(texts)

    all_risks = []
    for r in regions:
        all_risks.extend(r.risk_flags)
    merged.risk_flags = list(set(all_risks))

    return merged
