from __future__ import annotations

from dataclasses import dataclass
import re

_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*\r?\n?$")
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class SectionReplaceResult:
    text: str
    heading: str
    start_line: int
    end_line: int
    old_section: str
    new_section: str


@dataclass(frozen=True)
class _HeadingSpan:
    line_index: int
    line_number: int
    level: int
    text: str
    raw_line: str
    start_offset: int


def replace_markdown_section(
    text: str,
    *,
    heading: str,
    content: str,
    occurrence: int | None = None,
) -> SectionReplaceResult:
    """Replace one ATX heading section while preserving surrounding Markdown."""
    if occurrence is not None and occurrence < 1:
        raise ValueError("occurrence must be >= 1")
    if not heading:
        raise ValueError("heading is required for replace-section")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")

    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    headings = _scan_atx_headings(lines, offsets)
    matches = [candidate for candidate in headings if candidate.text == heading]
    if not matches:
        raise ValueError(f"heading not found: {heading}")
    if occurrence is None:
        if len(matches) > 1:
            raise ValueError(
                f"heading is ambiguous: {heading} appears {len(matches)} times; provide occurrence"
            )
        target = matches[0]
    else:
        if occurrence > len(matches):
            raise ValueError(
                f"heading occurrence not found: {heading} occurrence {occurrence}"
            )
        target = matches[occurrence - 1]

    next_boundary = len(lines)
    for candidate in headings:
        if candidate.line_index > target.line_index and candidate.level <= target.level:
            next_boundary = candidate.line_index
            break
    section_end_line_index = next_boundary
    while (
        section_end_line_index > target.line_index + 1
        and not lines[section_end_line_index - 1].strip()
    ):
        section_end_line_index -= 1

    start_offset = offsets[target.line_index]
    end_offset = (
        offsets[section_end_line_index]
        if section_end_line_index < len(offsets)
        else len(text)
    )
    old_section = text[start_offset:end_offset].rstrip("\r\n")
    new_section = _build_replacement_section(target, content)
    final = text[:start_offset] + new_section + text[end_offset:]
    final = final.rstrip("\r\n") + "\n"
    return SectionReplaceResult(
        text=final,
        heading=target.text,
        start_line=target.line_number,
        end_line=target.line_number
        + max(0, section_end_line_index - target.line_index - 1),
        old_section=old_section,
        new_section=new_section.rstrip("\r\n"),
    )


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    current = 0
    for line in lines:
        offsets.append(current)
        current += len(line)
    offsets.append(current)
    return offsets


def _scan_atx_headings(
    lines: list[str], offsets: list[int]
) -> list[_HeadingSpan]:
    headings: list[_HeadingSpan] = []
    start_index = _body_start_line_index(lines)
    in_fence = False
    fence_char = ""
    fence_len = 0
    for index, line in enumerate(lines[start_index:], start=start_index):
        fence = _FENCE_LINE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0
                continue
        if in_fence:
            continue
        parsed = _parse_atx_heading_line(line)
        if parsed is None:
            continue
        level, normalized = parsed
        headings.append(
            _HeadingSpan(index, index + 1, level, normalized, line, offsets[index])
        )
    return headings


def _body_start_line_index(lines: list[str]) -> int:
    if not lines or lines[0].strip() != "---":
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return index + 1
    return 0


def _parse_atx_heading_line(line: str) -> tuple[int, str] | None:
    match = _HEADING_LINE_RE.match(line)
    if not match:
        return None
    text = _normalize_heading_text(match.group(2))
    if not text:
        return None
    return len(match.group(1)), text


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"[ \t]+#+[ \t]*$", "", text.strip()).strip()


def _build_replacement_section(target: _HeadingSpan, content: str) -> str:
    stripped = content.strip()
    first_line = next((line for line in stripped.splitlines() if line.strip()), "")
    first_heading = _parse_atx_heading_line(first_line)
    if first_heading is None:
        return target.raw_line.rstrip("\r\n") + "\n" + stripped + "\n"

    level, normalized = first_heading
    if normalized != target.text:
        raise ValueError(
            f"replacement heading must match target heading: {target.text}"
        )
    if level != target.level:
        raise ValueError(
            f"replacement heading level must match target level: {target.level}"
        )
    replacement_lines = stripped.splitlines(keepends=True)
    offsets = _line_offsets(replacement_lines)
    nested_headings = _scan_atx_headings(replacement_lines, offsets)
    for candidate in nested_headings[1:]:
        if candidate.level <= target.level:
            raise ValueError(
                "full-section replacement must not contain an additional same-or-higher heading"
            )
    return stripped + "\n"
