from __future__ import annotations
import re
from typing import Optional


HIGH_RISK_PATTERNS = {
    "number": r"\d",
    "currency": r"[￥¥$€]\s*\d+",
    "percent": r"\d+(\.\d+)?%",
    "date": r"\d{4}[-/.年]\d{1,2}([-/.月]\d{1,2})?",
    "time": r"\d{1,2}:\d{2}",
    "phone_cn": r"1[3-9]\d{9}",
    "url": r"https?://|www\.",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
    "model": r"[A-Za-z]+[-_]?\d+[A-Za-z0-9-]*",
}


def detect_risk_flags(text: str) -> list[str]:
    if not text:
        return []

    flags = []
    for name, pattern in HIGH_RISK_PATTERNS.items():
        if re.search(pattern, text):
            flags.append(name)

    return flags


def has_high_risk_pattern(text: str) -> bool:
    return len(detect_risk_flags(text)) > 0


def should_auto_accept(
    original_text: str,
    corrected_text: str,
    ocr_confidence: float,
    llm_confidence: float,
    llm_needs_human: bool,
    bbox: Optional[list] = None,
) -> bool:
    if ocr_confidence < 0.75:
        return False
    if llm_confidence < 0.92:
        return False
    if has_high_risk_pattern(original_text):
        return False
    if has_high_risk_pattern(corrected_text):
        return False
    if edit_distance_ratio(original_text, corrected_text) > 0.25:
        return False
    if len(corrected_text) <= 1:
        return False
    if llm_needs_human:
        return False

    if bbox and len(bbox) >= 4:
        region_height = bbox[3] - bbox[1]
        if region_height < 14:
            return False

    return True


def should_force_review(
    original_text: str,
    corrected_text: str,
    ocr_confidence: float,
    llm_confidence: float,
    llm_needs_human: bool,
    bbox: Optional[list] = None,
) -> bool:
    if has_high_risk_pattern(original_text):
        return True
    if has_high_risk_pattern(corrected_text):
        return True
    if ocr_confidence < 0.75:
        return True
    if llm_confidence < 0.92:
        return True
    if edit_distance_ratio(original_text, corrected_text) > 0.25:
        return True
    if len(original_text) <= 2 and original_text != corrected_text:
        return True
    if bbox and len(bbox) >= 4:
        region_height = bbox[3] - bbox[1]
        if region_height < 14:
            return True
    if llm_needs_human:
        return True

    return False


def is_forbidden_change(original: str, corrected: str) -> bool:
    if extract_numbers(original) != extract_numbers(corrected):
        return True
    if extract_urls(original) != extract_urls(corrected):
        return True
    if extract_emails(original) != extract_emails(corrected):
        return True
    if extract_currency(original) != extract_currency(corrected):
        return True
    return False


def edit_distance_ratio(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    if not s1:
        return 1.0
    if not s2:
        return 1.0

    dist = _levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    return dist / max_len if max_len > 0 else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://\S+|www\.\S+", text)


def extract_emails(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)


def extract_currency(text: str) -> list[str]:
    return re.findall(r"[￥¥$€]\s*\d+(?:\.\d+)?", text)
