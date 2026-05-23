from __future__ import annotations
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_COMMON_HANZI_RANGES = [
    (0x4E00, 0x9FFF),
]

_RARE_RANGES = [
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
]

_HIGH_FREQ_HANZI = set(
    "的一是不了人我在有他这中大来上个国到说们为子和你地出会也时要就可以对生能而那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军三已老关点正新十无力它与长把机十者次进市什口直场政手向问向体明步把物则"
    "一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾佰仟"
    "天地人和大小多少高低长短远近快慢好坏新旧明暗冷热轻重"
    "东南西北上下左右前后内外中正反真假善恶美丑"
    "金木水火土风雨雷电日月星辰山川河流"
    "春夏秋冬年月日时分秒周"
    "红橙黄绿青蓝紫黑白灰"
    "爱恨情仇喜怒哀乐悲欢离合"
    "衣食住行吃喝玩乐读写听说"
    "学校公司家庭社会国家世界"
    "电电脑手机网络信息数据"
    "发展经济科学技术文化教育"
    "问题方法答案结果过程"
    "模块区域检测识别处理系统功能配置参数设置"
)

_PSEUDO_HANZI_TABLE = {
    "槻": "模",
    "砵": "体",
    "盫": "盖",
    "埵": "堆",
    "盨": "须",
    "奫": "深",
    "磻": "磷",
    "穇": "稳",
    "翙": "翔",
    "翯": "翎",
    "翾": "翼",
    "翣": "扇",
    "翥": "著",
    "翢": "周",
    "翛": "消",
    "翝": "宏",
    "翞": "强",
    "翨": "翅",
    "翪": "总",
    "翫": "玩",
    "翬": "辉",
    "翭": "羽",
    "翮": "合",
    "翰": "散",
    "翱": "傲",
    "翲": "飘",
    "翳": "意",
    "翴": "连",
    "翵": "从",
    "翶": "傲",
    "翷": "临",
    "翸": "飞",
    "翹": "起",
    "翺": "傲",
    "翻": "飞",
    "翼": "飞",
    "翽": "辉",
    "翾": "飞",
    "翿": "到",
    "耂": "老",
    "耇": "考",
    "耈": "考",
    "耉": "考",
    "耊": "老",
}

_LATEX_PATTERNS = [
    re.compile(r'\\[a-zA-Z]+\{[^}]*\}'),
    re.compile(r'\\[a-zA-Z]+'),
    re.compile(r'\$[^$]+\$'),
    re.compile(r'_{[^}]*}'),
    re.compile(r'\^{[^}]*}'),
    re.compile(r'\\frac\{[^}]*\}\{[^}]*\}'),
    re.compile(r'\\sqrt\{[^}]*\}'),
    re.compile(r'\\sum|\\prod|\\int|\\lim'),
    re.compile(r'[{}\\]'),
]

_MARKDOWN_PATTERNS = [
    re.compile(r'^#{1,6}\s'),
    re.compile(r'\*\*[^*]+\*\*'),
    re.compile(r'__[^_]+__'),
    re.compile(r'\[[^\]]+\]\([^)]+\)'),
    re.compile(r'^[-*+]\s'),
    re.compile(r'^\d+\.\s'),
    re.compile(r'`[^`]+`'),
    re.compile(r'```'),
]

_COMMON_CONFUSABLE = {
    "槻": "模",
    "砵": "体",
    "盫": "盖",
    "埵": "堆",
    "盨": "须",
    "奫": "深",
    "磻": "磷",
    "穇": "稳",
    "翙": "翔",
    "翯": "翎",
    "翾": "翼",
    "翣": "扇",
    "翥": "著",
    "翢": "周",
    "翛": "消",
    "翝": "宏",
    "翞": "强",
    "翨": "翅",
    "翪": "总",
    "翫": "玩",
    "翬": "辉",
    "翭": "羽",
    "翮": "合",
    "翯": "白",
    "翰": "散",
    "翱": "傲",
    "翲": "飘",
    "翳": "意",
    "翴": "连",
    "翵": "从",
    "翶": "傲",
    "翷": "临",
    "翸": "飞",
    "翹": "起",
    "翺": "傲",
    "翻": "飞",
    "翼": "飞",
    "翽": "辉",
    "翾": "飞",
    "翿": "到",
    "耂": "老",
    "耇": "考",
    "耈": "考",
    "耉": "考",
    "耊": "老",
}


def is_pseudo_hanzi(char: str) -> bool:
    if len(char) != 1:
        return False

    if char in _PSEUDO_HANZI_TABLE:
        return True

    code = ord(char)

    try:
        name = unicodedata.name(char, "")
    except ValueError:
        return True

    if "CJK" not in name and "HIRAGANA" not in name and "KATAKANA" not in name:
        return False

    for rare_start, rare_end in _RARE_RANGES:
        if rare_start <= code <= rare_end:
            return True

    for common_start, common_end in _COMMON_HANZI_RANGES:
        if common_start <= code <= common_end:
            if char in _HIGH_FREQ_HANZI:
                return False

            stroke_count = _estimate_stroke_count(char)
            if stroke_count > 25:
                return True

            return False

    return False


def detect_pseudo_hanzi_in_text(text: str) -> list[dict]:
    results = []

    for i, char in enumerate(text):
        if is_pseudo_hanzi(char):
            suggestion = _PSEUDO_HANZI_TABLE.get(char, _COMMON_CONFUSABLE.get(char, ""))
            results.append({
                "position": i,
                "char": char,
                "unicode": f"U+{ord(char):04X}",
                "suggestion": suggestion,
                "confidence": 0.9 if suggestion else 0.6,
            })

    return results


def clean_ocr_artifacts(text: str) -> str:
    if not text:
        return text

    result = text

    for pattern in _LATEX_PATTERNS:
        result = pattern.sub('', result)

    for pattern in _MARKDOWN_PATTERNS:
        result = pattern.sub('', result)

    result = re.sub(r'\s{2,}', ' ', result)
    result = result.strip()

    # Preserve all printable ASCII, CJK, Greek, and math notation.
    # The original only kept \w + CJK ranges, which stripped basic
    # ASCII punctuation (=+-,./()[]{} etc.) needed for formulas.
    allowed = (
        r'\w\s'                         # word chars + whitespace
        r' +!\"#\$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~'  # ASCII punctuation
        r'\u4e00-\u9fff'                         # CJK unified ideographs
        r'\u3000-\u303f'                         # CJK punctuation
        r'\uff00-\uffef'                         # Halfwidth/fullwidth forms
        r'\u0370-\u03ff'                         # Greek and Coptic
        r'\u1f00-\u1fff'                         # Greek extended
        r'\u2200-\u22ff'                         # Mathematical operators
        r'\u2300-\u23ff'                         # Miscellaneous technical
        r'\u25a0-\u25ff'                         # Geometric shapes
        r'\u27c0-\u27ef'                         # Miscellaneous math symbols A
        r'\u2980-\u29ff'                         # Miscellaneous math symbols B
        r'\u2a00-\u2aff'                         # Supplemental math operators
        r'\u2070-\u209f'                         # Superscripts and subscripts
        r'\u00b0-\u00ff'                         # Latin-1 supplement (\u00b7 \u00d7 \u00b0 \u00b1)
        r'\u2000-\u206f'                         # General punctuation
        r'\u2190-\u21ff'                         # Arrows
        r'\u0300-\u036f'                         # Combining diacritical marks
    )
    result = re.sub(f'[^{allowed}]', '', result)

    return result


def merge_ocr_candidates(candidates: list[str], confidences: list[float] = None) -> str:
    if not candidates:
        return ""

    if len(candidates) == 1:
        return candidates[0]

    if confidences and len(confidences) == len(candidates):
        best_idx = max(range(len(candidates)), key=lambda i: confidences[i])
        base = candidates[best_idx]
    else:
        base = max(candidates, key=len)

    pseudo_chars = detect_pseudo_hanzi_in_text(base)
    if not pseudo_chars:
        return base

    result = list(base)
    for pc in pseudo_chars:
        pos = pc["position"]
        if pos >= len(result):
            continue

        if pc["suggestion"]:
            result[pos] = pc["suggestion"]
        else:
            for candidate in candidates:
                if pos < len(candidate) and not is_pseudo_hanzi(candidate[pos]):
                    result[pos] = candidate[pos]
                    break

    return ''.join(result)


def validate_ocr_text(text: str) -> dict:
    if not text:
        return {"valid": False, "issues": ["empty_text"], "cleaned": ""}

    issues = []
    cleaned = text

    pseudo_chars = detect_pseudo_hanzi_in_text(text)
    if pseudo_chars:
        issues.append("pseudo_hanzi")
        for pc in pseudo_chars:
            if pc["suggestion"]:
                cleaned = cleaned[:pc["position"]] + pc["suggestion"] + cleaned[pc["position"] + 1:]

    cleaned_before = cleaned
    cleaned = clean_ocr_artifacts(cleaned)
    if cleaned != cleaned_before:
        issues.append("ocr_artifacts")

    if len(cleaned) < len(text) * 0.3:
        issues.append("excessive_cleanup")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "cleaned": cleaned,
        "pseudo_chars": pseudo_chars,
    }


def _estimate_stroke_count(char: str) -> int:
    code = ord(char)

    if 0x4E00 <= code <= 0x9FFF:
        offset = code - 0x4E00
        return min(25, max(1, (offset % 20) + 1))

    return 15
