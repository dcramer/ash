"""Helpers for Telegram user-visible provenance attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ash.tools.base import ToolResult

_WEB_SOURCE_TOOLS = {"web_search", "web_fetch", "browser"}


def _normalize_domain(value: str) -> str | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if not host:
        return None
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _format_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _domains_from_web_search(result: ToolResult) -> list[str]:
    raw = result.metadata.get("domains")
    if not isinstance(raw, list):
        return []
    domains: list[str] = []
    for item in raw:
        if isinstance(item, str):
            normalized = _normalize_domain(item)
            if normalized:
                domains.append(normalized)
    return _dedupe_ordered(domains)


def _domains_from_web_fetch(
    result: ToolResult, tool_input: dict[str, object]
) -> list[str]:
    candidates: list[str] = []
    for key in ("final_url", "url"):
        value = result.metadata.get(key)
        if isinstance(value, str):
            candidates.append(value)
    input_url = tool_input.get("url")
    if isinstance(input_url, str):
        candidates.append(input_url)
    domains = [_normalize_domain(candidate) for candidate in candidates]
    return _dedupe_ordered([domain for domain in domains if domain])


def _domains_from_browser(
    result: ToolResult, tool_input: dict[str, object]
) -> list[str]:
    candidates: list[str] = []
    for key in ("domain", "page_url", "url"):
        value = result.metadata.get(key)
        if isinstance(value, str):
            candidates.append(value)
    input_url = tool_input.get("url")
    if isinstance(input_url, str):
        candidates.append(input_url)
    domains = [_normalize_domain(candidate) for candidate in candidates]
    return _dedupe_ordered([domain for domain in domains if domain])


def extract_domains(
    tool_name: str,
    tool_input: dict[str, object],
    result: ToolResult,
) -> list[str]:
    if tool_name == "web_search":
        return _domains_from_web_search(result)
    if tool_name == "web_fetch":
        return _domains_from_web_fetch(result, tool_input)
    if tool_name == "browser":
        return _domains_from_browser(result, tool_input)
    return []


def extract_skill_name(tool_name: str, tool_input: dict[str, object]) -> str | None:
    if tool_name != "use_skill":
        return None
    skill_name = tool_input.get("skill")
    if not isinstance(skill_name, str):
        return None
    value = skill_name.strip()
    return value or None


@dataclass(slots=True)
class ProvenanceState:
    domains: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    def add_from_tool(
        self,
        tool_name: str,
        tool_input: dict[str, object],
        result: ToolResult,
    ) -> None:
        if result.is_error:
            return
        if tool_name in _WEB_SOURCE_TOOLS:
            self.domains = _dedupe_ordered(
                self.domains + extract_domains(tool_name, tool_input, result)
            )
        skill_name = extract_skill_name(tool_name, tool_input)
        if skill_name:
            self.skills = _dedupe_ordered(self.skills + [skill_name])

    def render_inline(self, max_domains: int = 3) -> str | None:
        top_domains = self.domains[:max_domains]
        skill_refs = [
            name if name.startswith("/") else f"/{name}" for name in self.skills
        ]
        if top_domains and skill_refs:
            return f"I checked {_format_list(top_domains)} and used {_format_list(skill_refs)}."
        if top_domains:
            return f"I checked {_format_list(top_domains)}."
        if skill_refs:
            return f"I used {_format_list(skill_refs)}."
        return None


def build_provenance_clause_from_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> str | None:
    """Build inline provenance text from serialized tool call records."""
    state = ProvenanceState()
    for call in tool_calls:
        tool_name = call.get("name")
        tool_input = call.get("input")
        if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
            continue
        result = ToolResult(
            content=str(call.get("result", "")),
            is_error=bool(call.get("is_error", False)),
            metadata=call.get("metadata", {})
            if isinstance(call.get("metadata"), dict)
            else {},
        )
        state.add_from_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
        )
    return state.render_inline(max_domains=3)
