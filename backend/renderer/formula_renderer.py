"""Formula rendering using matplotlib mathtext for LaTeX math expressions.

Handles Greek letters, super/subscripts, fractions, operators, etc.
Falls back to Unicode rendering when mathtext fails.
"""
from __future__ import annotations
import io
import logging
import re
from typing import Optional

import numpy as np
from matplotlib import mathtext
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Unicode superscript mapping
SUPER_MAP = str.maketrans({
    '0': '⁰', '1': '¹', '2': '²', '3': '³',
    '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷',
    '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼',
    '(': '⁽', ')': '⁾', 'n': 'ⁿ',
    'i': 'ⁱ',
})

# Unicode subscript mapping
SUB_MAP = str.maketrans({
    '0': '₀', '1': '₁', '2': '₂', '3': '₃',
    '4': '₄', '5': '₅', '6': '₆', '7': '₇',
    '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '=': '₌',
    '(': '₍', ')': '₎',
    'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ',
    'h': 'ₕ', 'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ',
    'n': 'ₙ', 'p': 'ₚ', 's': 'ₛ', 't': 'ₜ',
})

# Common math operators that should be in upright (roman) font, not italic
MATH_ROMAN_OPS = {
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'coth',
    'log', 'ln', 'lg', 'exp',
    'min', 'max', 'argmin', 'argmax',
    'lim', 'sup', 'inf',
    'det', 'tr', 'dim', 'ker', 'deg',
    'gcd', 'lcm', 'mod',
}


def contains_formula(text: str) -> bool:
    """Check if text contains mathematical formula notation.

    Returns False for text that mixes CJK characters with math notation,
    since mathtext cannot render Chinese characters.
    """
    # If text contains CJK characters, use regular text rendering instead
    # (formula renderer uses math fonts which lack CJK glyphs)
    if re.search(r'[一-鿿㐀-䶿豈-﫿]', text):
        return False
    # Subscript/superscript markers
    if re.search(r'[_^]\{', text):
        return True
    if re.search(r'[a-zA-Z]\^[a-zA-Z0-9]', text):
        return True
    if re.search(r'[a-zA-Z]_[a-zA-Z0-9]', text):
        return True
    # Greek Unicode characters
    if re.search(r'[Ͱ-Ͽᴀ-ᵿ∀-⋿⟀-⟯]', text):
        return True
    # LaTeX commands
    if re.search(r'\\(?:frac|sqrt|sum|int|prod|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|partial|nabla|infty|times|cdot|pm|leq|geq|neq|approx|equiv|propto|rightarrow|leftarrow|Rightarrow|Leftarrow|subset|supset|in|notin|forall|exists|mathbb|mathcal|mathbf|mathrm|text)', text):
        return True
    return False


def classify_as_formula(text: str) -> tuple:
    """Classify whether OCR text is a mathematical formula.

    Returns (is_formula: bool, reason: str).
    More permissive than contains_formula() — works with mixed CJK+math text
    and is intended for use during OCR post-processing to flag regions for
    formula rendering.
    """
    if not text or len(text.strip()) < 2:
        return False, ""

    # Tier 1: Strong LaTeX command signals — very high precision
    if re.search(
        r'\\frac|\\sqrt|\\sum|\\int|\\prod|\\alpha|\\beta|\\gamma|'
        r'\\delta|\\theta|\\sigma|\\phi|\\omega|\\partial|\\nabla|'
        r'\\infty|\\times|\\cdot|\\pm|\\leq|\\geq|\\neq|\\approx|'
        r'\\equiv|\\left|\\right|\\mathrm|\\mathbf|\\mathcal|\\mathbb',
        text
    ):
        return True, "latex_commands"

    # Tier 2: Math delimiters
    if re.search(r'\$\$.*?\$\$', text):
        return True, "math_delimiters"
    if re.search(r'\$[^$]+\$', text):
        return True, "math_delimiters"

    # Tier 3: Explicit LaTeX subscript/superscript
    if re.search(r'[_^]\{', text):
        return True, "script_braces"

    # Tier 4: Greek Unicode (strong signal even with CJK)
    if re.search(r'[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ]', text):
        return True, "greek_unicode"

    # Tier 5: Pure Latin formula patterns (no CJK, high math operator ratio)
    has_cjk = bool(re.search(r'[一-鿿]', text))
    if not has_cjk:
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        latin_ratio = latin_chars / max(len(text), 1)
        has_subsuper = bool(re.search(r'[a-zA-Z\d][_^][a-zA-Z\d\(]', text))
        has_operators = bool(re.search(
            r'\b(sin|cos|tan|tanh|log|ln|exp|max|min|dim|det|gcd|lim)\b', text
        ))
        if latin_ratio > 0.5 and (has_subsuper or has_operators):
            return True, "latin_formula_pattern"

    return False, ""


def text_to_latex(text: str) -> str:
    """Convert text with TeX-like notation to LaTeX for mathtext rendering.

    Handles:
    - Underscore subscripts: d_k -> d_{k}
    - Caret superscripts: x^2 -> x^{2}
    - Multi-char scripts: already in {braces}
    """
    # Already has proper LaTeX commands - just wrap
    if re.search(r'\\frac|\\sqrt|\\sum|\\int|\\begin\{', text):
        return f"${text}$"

    result = text

    # Greek letter Unicode -> LaTeX commands
    greek_map = {
        'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma',
        'δ': r'\delta', 'ε': r'\epsilon', 'ζ': r'\zeta',
        'η': r'\eta', 'θ': r'\theta', 'ι': r'\iota',
        'κ': r'\kappa', 'λ': r'\lambda', 'μ': r'\mu',
        'ν': r'\nu', 'ξ': r'\xi', 'ο': r'o',
        'π': r'\pi', 'ρ': r'\rho', 'ς': r'\varsigma',
        'σ': r'\sigma', 'τ': r'\tau', 'υ': r'\upsilon',
        'φ': r'\phi', 'χ': r'\chi', 'ψ': r'\psi',
        'ω': r'\omega',
        # Uppercase
        'Γ': r'\Gamma', 'Δ': r'\Delta', 'Θ': r'\Theta',
        'Λ': r'\Lambda', 'Ξ': r'\Xi', 'Π': r'\Pi',
        'Σ': r'\Sigma', 'Υ': r'\Upsilon', 'Φ': r'\Phi',
        'Ψ': r'\Psi', 'Ω': r'\Omega',
        # Math symbols
        '∞': r'\infty', '∂': r'\partial', '∇': r'\nabla',
        '√': r'\sqrt{}', '∫': r'\int', '∏': r'\prod',
        '∑': r'\sum', '∈': r'\in', '∉': r'\notin',
        '→': r'\rightarrow', '←': r'\leftarrow',
        '×': r' \times ', '·': r' \cdot ', '∘': r' \circ ',
        '≤': r'\leq', '≥': r'\geq', '≠': r'\neq',
        '≈': r'\approx', '≡': r'\equiv',
        '⊂': r'\subset', '⊃': r'\supset',
        '≪': r'\ll', '≫': r'\gg',
        '⊙': r'\odot', '⊗': r'\otimes', '⊕': r'\oplus',
    }
    for uni, latex in greek_map.items():
        result = result.replace(uni, latex)

    # Fix underscores for subscripts that aren't already braced
    # a_b -> a_{b}, a_{bc} -> a_{bc} (unchanged)
    result = re.sub(r'(?<!\\)_\{', '_{', result)  # leave braced ones alone
    # Fix single-char subscripts: _a -> _{a} (but not _{ already)
    result = re.sub(r'_([a-zA-Z0-9])(?![\w}])', r'_{\1}', result)

    # Fix single-char superscripts: ^a -> ^{a} (but not ^{ already)
    result = re.sub(r'\^([a-zA-Z0-9])(?![\w}])', r'^{\1}', result)

    # Fix concatenated LaTeX commands: \alphat -> \alpha{}t, \betax -> \beta{}x
    # This happens when OCR/LLM output has Greek LaTeX joined with following text
    bare_cmds = {v for v in greek_map.values() if '{' not in v}
    latex_cmd_names = '|'.join(sorted(bare_cmds, key=len, reverse=True))
    latex_cmd_names = latex_cmd_names.replace('\\', '\\\\')
    if bare_cmds:
        result = re.sub(
            rf'({latex_cmd_names})([a-zA-Z])',
            r'\1{} \2',
            result,
        )

    # Wrap math operators in \mathrm for upright rendering
    for op in sorted(MATH_ROMAN_OPS, key=len, reverse=True):
        # Match whole word operator
        pattern = rf'\b{op}\b'
        result = re.sub(pattern, rf'\\mathrm{{{op}}}', result)

    return f"${result}$"


def render_formula(
    latex: str,
    font_size: float,
    color: tuple = (0, 0, 0, 255),
    max_width: int = 0,
    dpi: int = 150,
) -> Optional[Image.Image]:
    """Render a LaTeX formula to a PIL RGBA image with transparent background.

    Args:
        latex: LaTeX math expression (e.g., "$P = UI \\cdot \\cos\\phi$")
        font_size: Font size in points
        color: RGB or RGBA color tuple
        max_width: Maximum width in pixels (0 = unlimited)
        dpi: Rendering DPI

    Returns:
        PIL Image in RGBA mode with transparent background, or None on failure
    """
    if not latex:
        return None

    try:
        # Ensure $ wrapping
        formula = latex.strip()
        if not formula.startswith('$'):
            formula = f'${formula}$'

        # Convert color to matplotlib format (0-1 range)
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0

        # Render to bytes
        buf = io.BytesIO()
        mathtext.math_to_image(
            formula, buf,
            dpi=dpi,
            format='png',
            color=(r, g, b),
        )
        buf.seek(0)

        result = Image.open(buf).convert("RGBA")

        # Make white background transparent
        data = np.array(result)
        if data.shape[2] >= 3:
            r_ch, g_ch, b_ch = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            white_mask = (r_ch > 250) & (g_ch > 250) & (b_ch > 250)
            data[white_mask, 3] = 0

        result = Image.fromarray(data, "RGBA")
        return result

    except Exception as e:
        logger.warning(f"Formula rendering failed: {e}")
        return None


def render_formula_unicode(text: str) -> str:
    """Convert TeX-like subscripts/superscripts to Unicode equivalents.
    Falls back approach when LaTeX rendering isn't available.
    """
    result = text

    # Convert _{} subscripts to Unicode
    def sub_repl(m):
        inner = m.group(1)
        return inner.translate(SUB_MAP)

    result = re.sub(r'_\{(.+?)\}', sub_repl, result)

    # Convert ^{} superscripts to Unicode
    def super_repl(m):
        inner = m.group(1)
        return inner.translate(SUPER_MAP)

    result = re.sub(r'\^\{(.+?)\}', super_repl, result)

    return result
