"""Treat retrieved text as untrusted evidence. Strip instruction-shaped payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Public, well-known instruction-override shapes. Not an exploit catalogue.
_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior|above) instructions"
    r"|disregard (the )?(system|developer) prompt"
    r"|you are now "
    r"|new system prompt"
    r"|exfiltrate"
    r"|do not follow the user"
    r"|override (your )?safety)",
    re.IGNORECASE,
)

_ROLE_SPOOF = re.compile(
    r"^\s*(system|developer|assistant)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class InjectionScan:
    blocked: bool
    reason: str
    cleaned: str


def scan_text(text: str) -> InjectionScan:
    """Mark instruction-shaped text. Return a cleaned copy for retrieval use."""
    if not text or not text.strip():
        return InjectionScan(blocked=False, reason="", cleaned="")
    if _INJECTION.search(text) or _ROLE_SPOOF.search(text):
        return InjectionScan(
            blocked=True,
            reason="retrieved_or_query_contains_instruction_override",
            cleaned="",
        )
    return InjectionScan(blocked=False, reason="", cleaned=text.strip())


def scan_query(query: str) -> InjectionScan:
    return scan_text(query)
