"""历史 DirectVisual 提示模板；仅用公开问题、选项及回答格式。"""
from typing import Mapping, Any

def require_str(row, key):
    value = row[key]
    if not isinstance(value, str) or not value:
        raise ValueError(key)
    return value

def build_direct_native_prompt(*, question_payload: Mapping[str, Any], answer: Mapping[str, Any], answer_type: str) -> str:
    """构造 direct-Qwen baseline 的 native-answer prompt。

    这里刻意不提“injected 3D evidence”，因为这个 baseline 不看我们的 3D/flow 信息。
    它只看视频帧和题目，然后按原始 answer.text 学会自回归输出答案。
    """

    question = require_str(question_payload, "text")
    qtype = str(question_payload.get("canonical_question_type") or "unknown")
    reference = str(question_payload.get("reference_condition") or "unknown")
    situation = str(question_payload.get("situation") or "").strip()
    lines = [
        "You are a spatial reasoning assistant.",
        "Use the video frames to answer the question.",
        f"Question type: {qtype}",
        f"Reference condition: {reference}",
    ]
    if situation:
        lines.append(f"Situation: {situation}")
    lines.append(f"Question: {question}")
    if answer_type == "multiple_choice":
        options = answer.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError("multiple_choice prompt requires answer.options")
        lines.append("Options:")
        for option in options:
            if not isinstance(option, Mapping):
                raise ValueError(f"option must be mapping, got {option!r}")
            letter = str(option.get("letter") or "").strip()
            text = str(option.get("text") or "").strip()
            if not letter or not text:
                raise ValueError(f"invalid option payload: {option}")
            lines.append(f"{letter}. {text}")
        lines.append("Answer with the option letter and the option text, e.g. 'A. chair'.")
    elif answer_type in {"numeric", "count"}:
        unit = str(answer.get("numeric_unit") or "").strip()
        if unit:
            lines.append(f"Answer as a concise number in {unit}.")
        else:
            lines.append("Answer as a concise number.")
    elif answer_type == "yes_no":
        lines.append("Answer yes or no.")
    else:
        lines.append("Answer with a concise text phrase.")
    lines.append("Answer:")
    return "\n".join(lines)
