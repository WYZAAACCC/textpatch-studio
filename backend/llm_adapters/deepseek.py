from __future__ import annotations
import json
import logging
import re
from typing import Optional

import requests

from backend.llm_adapters.base import LLMClient, TextCorrectionRequest, TextCorrectionResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个中文图片文字 OCR 校对助手，专门处理AI生成图片中的文字问题。

AI生成图片中经常出现以下问题：
- 伪汉字：看起来像汉字但实际不存在的字符，如"奫砵盫埵盨"等
- OCR误识别：形近字被错误识别，如"磁"被误识为"码"、"槻"被误识为"模"、"盘"被误识为"叠"
- 乱码小字：AI生成的小字区域经常是无意义的随机汉字
- 数学公式中的符号错误：希腊字母被误识为形近汉字或符号，上下标丢失

你的任务是根据 OCR 结果、上下文和常见中文表达，校正这些错误。

你必须遵守：

1. 优先校正伪汉字和乱码：如果OCR结果包含罕见汉字、无意义字符组合，应尝试替换为上下文合理的常见词。
2. 校正形近字错误：如"掩磁"→"掩码"、"子槻块"→"子模块"、"堆盘"→"堆叠"。
3. **数学公式保护**：如果OCR结果包含明显数学符号，请保持其正确形式：
   - 希腊字母：保持α, β, γ, δ, ε, θ, σ, φ, ω, Σ, Π, Ω等不改变
   - 数学算子：sin, cos, tan, tanh, log, ln, exp, max, min 等保持原样
   - 上下标：d_k, x^2, 10^{{-3}} 等使用 LaTeX 记号（下划线_表下标，^表上标）
   - 对于纯公式区域，corrected_text中可保留LaTeX风格记号
4. 不要创作新文案，不要改写风格，不要润色，不要扩写，不要翻译。
5. 不要补充图片中不存在的文字。
6. 数字、价格、日期、时间、电话、网址、邮箱、品牌名、型号、地址、法律/医疗/金融声明必须谨慎处理，不得自动修改。
7. 如果无法确定校正结果，必须标记 needs_human=true，但仍然给出你最好的猜测。
8. 即使 needs_human=true，也请在 corrected_text 中给出你的最佳校正建议，而不是返回原文。
9. 输出必须是严格 JSON，不要输出 Markdown，不要输出解释性正文。"""

USER_PROMPT_TEMPLATE = """请校正以下AI生成图片中文字区域的 OCR 结果。

这是AI生成的图片，文字区域可能包含伪汉字、乱码或OCR误识别。如果区域中包含数学公式，请注意保全公式符号的正确性。

场景说明：
{scene_hint}

OCR 主结果：
{ocr_best}

OCR 候选：
{ocr_candidates}

相邻文字：
{neighbor_texts}

风险标签：
{risk_flags}

校正要点：
- 如果OCR结果包含罕见汉字或无意义字符组合，这很可能是伪汉字或乱码，请根据上下文替换为合理的常见词。
- 如果OCR结果中有形近字错误（如"磁"→"码"、"槻"→"模"、"盘"→"叠"），请校正。
- **数学公式区域检测与处理**：
  - 判断当前区域是否为数学公式（包含希腊字母、数学符号、LaTeX命令、上下标结构等）
  - 如果是公式区域：is_formula 设为 true，并在 latex 字段输出标准LaTeX代码
  - 希腊字母（αβγδσθφωΣΠΩ等）不要改成汉字，在LaTeX中用 \\alpha、\\beta 等表示
  - 函数名（sin,cos,tan,tanh,log,exp,max,min等）保持原样
  - 下标用下划线标记：如d_k, h_t, C_out, x_{{t-1}}
  - 上标用^标记：如x^2, e^{{-3}}, 10^{{-6}}
  - 分式用 \\frac{{分子}}{{分母}}，根号用 \\sqrt{{}}
  - 点积用 \\cdot，Hadamard积用 \\odot，卷积用 *
  - **如果不是公式区域**：is_formula 设为 false，latex 设为空字符串
- 不允许创造新文案、扩写、润色、翻译。
- 不允许自动修改数字、价格、日期、网址、电话、邮箱、品牌名、型号。
- 即使不确定，也请在 corrected_text 中给出你最好的校正建议。
- 如果有不确定字符，请在 uncertain_chars 中列出。

请只输出如下 JSON：

{{
  "corrected_text": "校正后的文字（公式则为纯LaTeX代码）",
  "confidence": 0.0,
  "correction_type": "unchanged|ocr_noise_removed|typo_fixed|format_fixed|uncertain",
  "is_formula": false,
  "latex": "",
  "changed_chars": [
    {{
      "from_char": "原字符",
      "to_char": "新字符",
      "reason": "修改原因"
    }}
  ],
  "uncertain_chars": [],
  "needs_human": true
}}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "corrected_text",
        "confidence",
        "correction_type",
        "changed_chars",
        "uncertain_chars",
        "needs_human",
    ],
    "properties": {
        "corrected_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "correction_type": {
            "type": "string",
            "enum": [
                "unchanged",
                "ocr_noise_removed",
                "typo_fixed",
                "format_fixed",
                "uncertain",
            ],
        },
        "changed_chars": {"type": "array"},
        "uncertain_chars": {"type": "array"},
        "needs_human": {"type": "boolean"},
    },
}


class DeepSeekClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        timeout: int = 60,
        max_retries: int = 3,
        temperature: float = 0.0,
        top_p: float = 0.1,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p

    def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        candidates_str = json.dumps(
            [{"text": c.text if hasattr(c, "text") else str(c), "confidence": c.confidence if hasattr(c, "confidence") else 0, "source": c.source if hasattr(c, "source") else ""} for c in request.ocr_candidates],
            ensure_ascii=False,
        )
        neighbor_str = " | ".join(request.neighbor_texts) if request.neighbor_texts else "无"
        risk_str = ", ".join(request.risk_flags) if request.risk_flags else "无"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            scene_hint=request.scene_hint or "通用图片",
            ocr_best=request.ocr_best,
            ocr_candidates=candidates_str,
            neighbor_texts=neighbor_str,
            risk_flags=risk_str,
        )

        for attempt in range(self.max_retries):
            try:
                result = self._call_api(user_prompt)
                return self._parse_response(result)
            except Exception as e:
                logger.warning(f"DeepSeek API attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return TextCorrectionResponse(
                        corrected_text=request.ocr_best,
                        confidence=0.0,
                        correction_type="uncertain",
                        changed_chars=[],
                        uncertain_chars=[],
                        needs_human=True,
                        is_formula=False,
                        latex="",
                        raw_response={"error": str(e)},
                    )

    def _call_api(self, user_prompt: str) -> dict:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        return {"content": content, "raw": data}

    def _parse_response(self, api_result: dict) -> TextCorrectionResponse:
        content = api_result["content"]
        raw = api_result.get("raw", {})

        try:
            parsed = self._extract_json(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON response: {content}")
            raise ValueError(f"Invalid JSON from LLM: {content[:200]}")

        self._validate_schema(parsed)

        changed_chars = []
        for cc in parsed.get("changed_chars", []):
            changed_chars.append({
                "from_char": cc.get("from_char", ""),
                "to_char": cc.get("to_char", ""),
                "reason": cc.get("reason", ""),
            })

        return TextCorrectionResponse(
            corrected_text=parsed.get("corrected_text", ""),
            confidence=float(parsed.get("confidence", 0.0)),
            correction_type=parsed.get("correction_type", "uncertain"),
            changed_chars=changed_chars,
            uncertain_chars=parsed.get("uncertain_chars", []),
            needs_human=bool(parsed.get("needs_human", True)),
            is_formula=bool(parsed.get("is_formula", False)),
            latex=str(parsed.get("latex", "")),
            raw_response=raw,
        )

    def _extract_json(self, content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip()

        # Find the outermost JSON object using brace matching
        start = content.find('{')
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", content, 0)

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(content)):
            ch = content[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    json_str = content[start:i+1]
                    return json.loads(json_str)

        raise json.JSONDecodeError("Unmatched braces", content, start)

    def _validate_schema(self, data: dict) -> None:
        required_fields = ["corrected_text", "confidence", "correction_type", "changed_chars", "uncertain_chars", "needs_human"]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")

        valid_types = {"unchanged", "ocr_noise_removed", "typo_fixed", "format_fixed", "uncertain"}
        raw_type = data.get("correction_type", "")
        # Normalize: LLM sometimes returns pipe-separated values like "typo_fixed|format_fixed"
        for part in raw_type.replace("|", " ").split():
            if part in valid_types:
                data["correction_type"] = part
                break
        if data.get("correction_type") not in valid_types:
            data["correction_type"] = "uncertain"
