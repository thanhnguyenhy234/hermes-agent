from __future__ import annotations

from typing import Optional


def _clean_piece(value: Optional[str]) -> str:
    return str(value or "").strip()


def format_model_reply_label(model: Optional[str], provider: Optional[str] = None) -> str:
    parts: list[str] = []
    model_name = _clean_piece(model)
    provider_name = _clean_piece(provider)
    if model_name:
        parts.append(f"Model: {model_name}")
    if provider_name:
        parts.append(f"Provider: {provider_name}")
    return " | ".join(parts)


def build_model_reply_suffix(model: Optional[str], provider: Optional[str] = None) -> str:
    label = format_model_reply_label(model, provider)
    return f"\n\n[{label}]" if label else ""


def append_model_reply_suffix(text: str, model: Optional[str] = None, provider: Optional[str] = None) -> str:
    stripped = str(text or "").rstrip()
    if not stripped:
        return stripped
    suffix = build_model_reply_suffix(model, provider)
    if not suffix:
        return stripped
    tail = stripped[-max(200, len(suffix) + 20):]
    if suffix in tail or "[Model:" in tail:
        return stripped
    return stripped + suffix


def build_model_markdown_lines(model: Optional[str], provider: Optional[str] = None) -> list[str]:
    lines: list[str] = []
    model_name = _clean_piece(model)
    provider_name = _clean_piece(provider)
    if model_name:
        lines.append(f"**Model:** {model_name}")
    if provider_name:
        lines.append(f"**Provider:** {provider_name}")
    return lines
