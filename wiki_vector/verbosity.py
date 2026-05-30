from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import math
import re
from pathlib import Path
from typing import Any, Sequence

from .markdown import parse_markdown
from .readability import ReadabilityAnalyzer, merge_readability_analyses

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+|[가-힣]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^\)]+\)")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "have", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "use", "with", "without",
    "when", "where", "which", "while", "will", "should", "could", "would", "not", "only", "than", "then",
    "too", "very", "more", "most", "less", "also", "etc", "etc.", "및", "그리고", "그러나", "하지만", "또는", "에서", "으로", "하다", "한다", "하는", "있다", "없는", "위해",
}

@dataclass(frozen=True)
class VerbosityProfile:
    name: str = "wiki-default-v1"
    warning_line_count: int = 200
    hard_line_count: int = 300
    warning_section_lines: int = 80
    hard_section_lines: int = 120
    min_links_for_long_page: int = 2
    avg_sentence_words_threshold: float = 28.0
    long_sentence_words: int = 30
    long_sentence_ratio_threshold: float = 0.25
    repeated_5gram_ratio_threshold: float = 0.08
    near_duplicate_sentence_ratio_threshold: float = 0.10
    unique_content_word_ratio_floor: float = 0.36
    lexical_density_floor: float = 0.42
    gzip_ratio_floor: float = 0.35
    heading_density_lines: int = 80
    high_word_count: int = 3000

@dataclass(frozen=True)
class VerbosityReason:
    code: str
    value: float | int | str
    threshold: float | int | str
    weight: float
    message: str
    start_line: int | None = None
    end_line: int | None = None
    heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class VerboseSection:
    heading: str
    level: int
    start_line: int
    end_line: int
    line_count: int
    score: float
    reasons: list[VerbosityReason]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class VerbosityResult:
    path: str
    is_verbose: bool
    score: float
    severity: str
    profile: str
    metrics: dict[str, float | int | list[dict[str, Any]]]
    reasons: list[VerbosityReason]
    suggestions: list[str]
    sections: list[VerboseSection]
    semantic: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

_WEIGHTS = {
    "line_count_high": 0.22,
    "line_count_warning": 0.10,
    "word_count_high": 0.10,
    "section_too_long": 0.20,
    "section_warning": 0.08,
    "understructured_long_page": 0.14,
    "long_page_low_links": 0.08,
    "wall_of_text": 0.14,
    "avg_sentence_too_long": 0.08,
    "long_sentence_ratio_high": 0.08,
    "sentence_extreme": 0.08,
    "repeated_5grams": 0.12,
    "near_duplicate_sentences": 0.12,
    "low_unique_content_ratio": 0.08,
    "low_lexical_density": 0.06,
    "high_compressibility": 0.08,
    "rewrite_not_compact": 0.08,
    "rewrite_too_destructive": 0.08,
}


def analyze_verbosity(
    path: str,
    text: str,
    *,
    profile: VerbosityProfile | None = None,
    include_code: bool = False,
    compare_to: str | None = None,
    readability_analyzers: Sequence[ReadabilityAnalyzer] | None = None,
) -> VerbosityResult:
    profile = profile or VerbosityProfile()
    doc = parse_markdown(Path(path), text)
    body = doc.body
    full_lines = text.splitlines()
    body_lines = body.splitlines()
    prose = body if include_code else _strip_code_blocks(body)
    prose_no_md = _strip_markdown_noise(prose)
    tokens = _tokens(prose_no_md)
    content_tokens = [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]
    sentences = _sentences(prose_no_md)
    sentence_lengths = [len(_tokens(s)) for s in sentences if _tokens(s)]

    sections_raw = _sections(doc.frontmatter_lines, body)
    section_infos: list[VerboseSection] = []
    reasons: list[VerbosityReason] = []

    heading_count = len(sections_raw)
    line_count = len(full_lines)
    body_line_count = len(body_lines)
    section_line_counts = [s["line_count"] for s in sections_raw]
    max_section_lines = max(section_line_counts or [body_line_count or line_count])
    avg_section_lines = sum(section_line_counts) / max(len(section_line_counts), 1)
    bullet_line_ratio = _line_ratio(body_lines, r"^\s*(?:[-*+] |\d+[.)] )")
    table_line_ratio = sum(1 for line in body_lines if "|" in line or re.search(r"^\s*\|?\s*:?-{3,}:?", line)) / max(len(body_lines), 1)
    code_line_ratio = _code_line_ratio(body)
    wikilink_count = len(_WIKILINK_RE.findall(body))
    markdown_link_count = len(_MD_LINK_RE.findall(body))
    repeated_5, top_5 = _repeated_ngram_ratio(tokens, 5)
    repeated_8, _ = _repeated_ngram_ratio(tokens, 8)
    near_dup = _near_duplicate_sentence_ratio(sentences)
    heading_rep = _heading_repetition_ratio(sections_raw)
    gzip_ratio = _gzip_ratio(prose_no_md)
    unique_token_ratio = len(set(tokens)) / max(len(tokens), 1)
    unique_content_word_ratio = len(set(content_tokens)) / max(len(content_tokens), 1)
    lexical_density = len(content_tokens) / max(len(tokens), 1)
    stopword_ratio = sum(1 for t in tokens if t in _STOPWORDS) / max(len(tokens), 1)
    avg_sentence_words = sum(sentence_lengths) / max(len(sentence_lengths), 1)
    max_sentence_words = max(sentence_lengths or [0])
    long_sentence_count = sum(1 for n in sentence_lengths if n >= profile.long_sentence_words)
    long_sentence_ratio = long_sentence_count / max(len(sentence_lengths), 1)
    avg_word_chars = sum(len(t) for t in tokens) / max(len(tokens), 1)
    ari_score = 4.71 * (sum(len(t) for t in tokens) / max(len(tokens), 1)) + 0.5 * (len(tokens) / max(len(sentence_lengths), 1)) - 21.43

    metrics: dict[str, float | int | list[dict[str, Any]]] = {
        "line_count": line_count,
        "body_line_count": body_line_count,
        "word_count": len(tokens),
        "char_count": len(text),
        "heading_count": heading_count,
        "max_section_lines": max_section_lines,
        "avg_section_lines": round(avg_section_lines, 3),
        "lines_per_heading": round(body_line_count / max(heading_count, 1), 3),
        "bullet_line_ratio": round(bullet_line_ratio, 6),
        "table_line_ratio": round(table_line_ratio, 6),
        "code_line_ratio": round(code_line_ratio, 6),
        "wikilink_count": wikilink_count,
        "markdown_link_count": markdown_link_count,
        "sentence_count": len(sentence_lengths),
        "avg_sentence_words": round(avg_sentence_words, 3),
        "max_sentence_words": max_sentence_words,
        "long_sentence_count": long_sentence_count,
        "long_sentence_ratio": round(long_sentence_ratio, 6),
        "avg_word_chars": round(avg_word_chars, 3),
        "ari_score": round(ari_score, 3),
        "repeated_5gram_ratio": round(repeated_5, 6),
        "repeated_8gram_ratio": round(repeated_8, 6),
        "top_repeated_phrases": top_5,
        "near_duplicate_sentence_ratio": round(near_dup, 6),
        "heading_repetition_ratio": round(heading_rep, 6),
        "gzip_ratio": round(gzip_ratio, 6),
        "unique_token_ratio": round(unique_token_ratio, 6),
        "unique_content_word_ratio": round(unique_content_word_ratio, 6),
        "lexical_density": round(lexical_density, 6),
        "stopword_ratio": round(stopword_ratio, 6),
    }

    def add(code: str, value: float | int | str, threshold: float | int | str, message: str, *, start_line=None, end_line=None, heading=None, weight: float | None = None) -> None:
        reasons.append(VerbosityReason(code, _round_value(value), threshold, weight if weight is not None else _WEIGHTS[code], message, start_line, end_line, heading))

    if line_count >= profile.hard_line_count:
        add("line_count_high", line_count, profile.hard_line_count, f"Page exceeds {profile.hard_line_count} lines.")
    elif line_count >= profile.warning_line_count:
        add("line_count_warning", line_count, profile.warning_line_count, f"Page exceeds {profile.warning_line_count} lines.")
    if len(tokens) >= profile.high_word_count:
        add("word_count_high", len(tokens), profile.high_word_count, f"Page has {len(tokens)} prose tokens.")
    if max_section_lines >= profile.hard_section_lines:
        longest = max(sections_raw, key=lambda s: s["line_count"], default=None)
        add("section_too_long", max_section_lines, profile.hard_section_lines, "A section is too long to scan comfortably.", start_line=longest and longest["start_line"], end_line=longest and longest["end_line"], heading=longest and longest["heading"])
    elif max_section_lines >= profile.warning_section_lines:
        longest = max(sections_raw, key=lambda s: s["line_count"], default=None)
        add("section_warning", max_section_lines, profile.warning_section_lines, "A section is approaching split threshold.", start_line=longest and longest["start_line"], end_line=longest and longest["end_line"], heading=longest and longest["heading"])
    if line_count >= profile.warning_line_count and heading_count < (line_count / profile.heading_density_lines):
        add("understructured_long_page", heading_count, f">= {line_count / profile.heading_density_lines:.1f} headings", "Long page has too few headings for navigation.")
    if line_count >= profile.warning_line_count and wikilink_count + markdown_link_count < profile.min_links_for_long_page:
        add("long_page_low_links", wikilink_count + markdown_link_count, profile.min_links_for_long_page, "Long page has too few links for hub/deep-dive navigation.")
    if line_count >= 120 and bullet_line_ratio + table_line_ratio < 0.08 and heading_count <= 2:
        add("wall_of_text", round(bullet_line_ratio + table_line_ratio, 6), ">= 0.08 list/table ratio or >2 headings", "Page looks like a wall of prose.")
    if avg_sentence_words >= profile.avg_sentence_words_threshold:
        add("avg_sentence_too_long", avg_sentence_words, profile.avg_sentence_words_threshold, "Average sentence is too long.")
    if long_sentence_ratio >= profile.long_sentence_ratio_threshold:
        add("long_sentence_ratio_high", long_sentence_ratio, profile.long_sentence_ratio_threshold, "Too many long sentences.")
    if max_sentence_words >= 80:
        add("sentence_extreme", max_sentence_words, 80, "At least one sentence is extremely long.")
    if repeated_5 >= profile.repeated_5gram_ratio_threshold:
        add("repeated_5grams", repeated_5, profile.repeated_5gram_ratio_threshold, "Repeated 5-grams suggest redundant phrasing.")
    if near_dup >= profile.near_duplicate_sentence_ratio_threshold:
        add("near_duplicate_sentences", near_dup, profile.near_duplicate_sentence_ratio_threshold, "Near-duplicate sentences suggest repeated claims.")
    if len(tokens) >= 500 and unique_content_word_ratio < profile.unique_content_word_ratio_floor:
        add("low_unique_content_ratio", unique_content_word_ratio, profile.unique_content_word_ratio_floor, "Low unique content-word ratio suggests low information density.")
    if len(tokens) >= 500 and lexical_density < profile.lexical_density_floor:
        add("low_lexical_density", lexical_density, profile.lexical_density_floor, "Low lexical density suggests excess connective prose.")
    if len(tokens) >= 500 and gzip_ratio < profile.gzip_ratio_floor:
        add("high_compressibility", gzip_ratio, profile.gzip_ratio_floor, "Text is highly compressible, suggesting repetition.")

    for sec in sections_raw:
        sec_result = _analyze_section(sec, profile, include_code=include_code)
        if sec_result.reasons:
            section_infos.append(sec_result)

    if compare_to is not None:
        cmp_metrics, cmp_reasons = _comparison_metrics(text, compare_to)
        metrics.update(cmp_metrics)
        reasons.extend(cmp_reasons)

    semantic = None
    if readability_analyzers:
        analyses = [
            analyzer.analyze(path=path, text=text, metrics=metrics, sections=sections_raw, compare_to=compare_to)
            for analyzer in readability_analyzers
        ]
        semantic = merge_readability_analyses(analyses)

    score = min(1.0, sum(r.weight for r in reasons))
    if line_count >= profile.warning_line_count and heading_count >= line_count / 60 and max_section_lines < profile.warning_section_lines:
        score = max(0.0, score - 0.08)
    if bullet_line_ratio + table_line_ratio >= 0.25 and repeated_5 < profile.repeated_5gram_ratio_threshold:
        score = max(0.0, score - 0.05)
    if line_count >= profile.hard_line_count or max_section_lines >= profile.hard_section_lines:
        score = max(score, 0.65)
    score = round(min(1.0, score), 6)
    hard = any(r.code in {"line_count_high", "section_too_long"} for r in reasons)
    if score >= 0.65 or hard:
        severity = "high"
    elif score >= 0.35 or reasons:
        severity = "warning"
    else:
        severity = "ok"
    return VerbosityResult(
        path=path,
        is_verbose=severity in {"warning", "high"},
        score=score,
        severity=severity,
        profile=profile.name,
        metrics=metrics,
        reasons=reasons,
        suggestions=_suggestions(reasons, sections_raw, metrics),
        sections=sorted(section_infos, key=lambda s: s.score, reverse=True),
        semantic=semantic,
    )


def _sections(frontmatter_lines: int, body: str) -> list[dict[str, Any]]:
    stripped = body.strip()
    if not stripped:
        return []
    matches = list(_HEADING_RE.finditer(stripped))
    if not matches:
        line_count = _line_count(stripped)
        return [{"heading": "(document)", "level": 0, "text": stripped, "start_line": frontmatter_lines + 1, "end_line": frontmatter_lines + line_count, "line_count": line_count}]
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stripped)
        sec_text = stripped[start:end].strip()
        start_line = frontmatter_lines + _line_number_at(stripped, start)
        end_line = frontmatter_lines + _line_number_at(stripped, max(end - 1, start))
        out.append({"heading": m.group(2).strip(), "level": len(m.group(1)), "text": sec_text, "start_line": start_line, "end_line": end_line, "line_count": max(1, end_line - start_line + 1)})
    return out


def _analyze_section(sec: dict[str, Any], profile: VerbosityProfile, *, include_code: bool) -> VerboseSection:
    text = sec["text"] if include_code else _strip_code_blocks(sec["text"])
    tokens = _tokens(_strip_markdown_noise(text))
    sentences = _sentences(text)
    slens = [len(_tokens(s)) for s in sentences if _tokens(s)]
    repeated_5, _ = _repeated_ngram_ratio(tokens, 5)
    content = [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]
    unique_content = len(set(content)) / max(len(content), 1)
    links = len(_WIKILINK_RE.findall(text)) + len(_MD_LINK_RE.findall(text))
    reasons: list[VerbosityReason] = []
    if sec["line_count"] >= profile.hard_section_lines:
        reasons.append(VerbosityReason("section_too_long", sec["line_count"], profile.hard_section_lines, _WEIGHTS["section_too_long"], "Section exceeds hard line threshold.", sec["start_line"], sec["end_line"], sec["heading"]))
    elif sec["line_count"] >= profile.warning_section_lines:
        reasons.append(VerbosityReason("section_warning", sec["line_count"], profile.warning_section_lines, _WEIGHTS["section_warning"], "Section exceeds warning line threshold.", sec["start_line"], sec["end_line"], sec["heading"]))
    if slens and sum(slens) / len(slens) >= profile.avg_sentence_words_threshold:
        reasons.append(VerbosityReason("avg_sentence_too_long", round(sum(slens) / len(slens), 3), profile.avg_sentence_words_threshold, _WEIGHTS["avg_sentence_too_long"], "Section average sentence is too long.", sec["start_line"], sec["end_line"], sec["heading"]))
    if repeated_5 >= profile.repeated_5gram_ratio_threshold:
        reasons.append(VerbosityReason("repeated_5grams", round(repeated_5, 6), profile.repeated_5gram_ratio_threshold, _WEIGHTS["repeated_5grams"], "Section has repeated 5-grams.", sec["start_line"], sec["end_line"], sec["heading"]))
    if len(tokens) >= 150 and unique_content < profile.unique_content_word_ratio_floor:
        reasons.append(VerbosityReason("low_unique_content_ratio", round(unique_content, 6), profile.unique_content_word_ratio_floor, _WEIGHTS["low_unique_content_ratio"], "Section has low unique content-word ratio.", sec["start_line"], sec["end_line"], sec["heading"]))
    if sec["line_count"] >= profile.warning_section_lines and links == 0:
        reasons.append(VerbosityReason("long_page_low_links", 0, 1, _WEIGHTS["long_page_low_links"], "Long section has no links.", sec["start_line"], sec["end_line"], sec["heading"]))
    return VerboseSection(sec["heading"], sec["level"], sec["start_line"], sec["end_line"], sec["line_count"], round(min(1.0, sum(r.weight for r in reasons)), 6), reasons)


def _strip_code_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.startswith("    ") or line.startswith("\t"):
            continue
        out.append(line)
    return "\n".join(out)


def _strip_markdown_noise(text: str) -> str:
    text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*+] |\d+[.)] )", "", text, flags=re.MULTILINE)
    return text


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return []
    parts = _SENTENCE_RE.split(compact)
    if len(parts) == 1:
        # fallback: markdown bullet/newline-ish chunks become sentence-ish spans
        parts = re.split(r"(?:\n+|;\s+)", text)
    return [p.strip() for p in parts if p.strip()]


def _repeated_ngram_ratio(tokens: list[str], n: int) -> tuple[float, list[dict[str, Any]]]:
    if len(tokens) < n:
        return 0.0, []
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    total = max(len(grams), 1)
    top = []
    for gram, c in counts.most_common(5):
        if c <= 1:
            continue
        if all(t in _STOPWORDS for t in gram):
            continue
        top.append({"phrase": " ".join(gram), "count": c})
    return repeated / total, top


def _near_duplicate_sentence_ratio(sentences: list[str]) -> float:
    toks = [set(t for t in _tokens(s) if t not in _STOPWORDS) for s in sentences]
    toks = [t for t in toks if len(t) >= 4]
    if len(toks) < 2:
        return 0.0
    checks = dup = 0
    window = 40
    for i, a in enumerate(toks):
        for b in toks[i+1:i+1+window]:
            checks += 1
            sim = len(a & b) / max(len(a | b), 1)
            if sim >= 0.82:
                dup += 1
    return dup / max(checks, 1)


def _heading_repetition_ratio(sections: list[dict[str, Any]]) -> float:
    if not sections:
        return 0.0
    hits = 0
    total = 0
    for sec in sections:
        htoks = [t for t in _tokens(sec["heading"]) if t not in _STOPWORDS]
        body = Counter(_tokens(sec["text"]))
        for t in htoks:
            total += 1
            if body[t] >= 6:
                hits += 1
    return hits / max(total, 1)


def _gzip_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 1.0
    return len(gzip.compress(data)) / max(len(data), 1)


def _line_ratio(lines: list[str], pattern: str) -> float:
    rx = re.compile(pattern)
    return sum(1 for line in lines if rx.search(line)) / max(len(lines), 1)


def _code_line_ratio(body: str) -> float:
    lines = body.splitlines()
    in_fence = False
    code = 0
    for line in lines:
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_fence = not in_fence
            code += 1
        elif in_fence or line.startswith("    ") or line.startswith("\t"):
            code += 1
    return code / max(len(lines), 1)


def _comparison_metrics(source_text: str, rewrite_text: str) -> tuple[dict[str, float | int], list[VerbosityReason]]:
    src_body = parse_markdown(Path("source.md"), source_text).body
    rew_body = parse_markdown(Path("rewrite.md"), rewrite_text).body
    src_tokens = _tokens(_strip_markdown_noise(_strip_code_blocks(src_body)))
    rew_tokens = _tokens(_strip_markdown_noise(_strip_code_blocks(rew_body)))
    src_set = set(src_tokens)
    rew_set = set(rew_tokens)
    src_links = set(_WIKILINK_RE.findall(src_body))
    rew_links = set(_WIKILINK_RE.findall(rew_body))
    src_heads = {m.group(2).strip().lower() for m in _HEADING_RE.finditer(src_body)}
    rew_heads = {m.group(2).strip().lower() for m in _HEADING_RE.finditer(rew_body)}
    compression_ratio = len(rew_tokens) / max(len(src_tokens), 1)
    heading_preservation = len(src_heads & rew_heads) / max(len(src_heads), 1)
    link_preservation = len(src_links & rew_links) / max(len(src_links), 1) if src_links else 1.0
    metrics = {
        "source_word_count": len(src_tokens),
        "rewrite_word_count": len(rew_tokens),
        "compression_ratio": round(compression_ratio, 6),
        "kept_unigrams_ratio": round(len(src_set & rew_set) / max(len(src_set), 1), 6),
        "deleted_unigrams_ratio": round(len(src_set - rew_set) / max(len(src_set), 1), 6),
        "added_unigrams_ratio": round(len(rew_set - src_set) / max(len(rew_set), 1), 6),
        "heading_preservation_ratio": round(heading_preservation, 6),
        "wikilink_preservation_ratio": round(link_preservation, 6),
    }
    reasons: list[VerbosityReason] = []
    if compression_ratio > 0.75:
        reasons.append(VerbosityReason("rewrite_not_compact", round(compression_ratio, 6), 0.75, _WEIGHTS["rewrite_not_compact"], "Compared rewrite is not substantially more compact."))
    if heading_preservation < 0.5 or link_preservation < 0.5:
        reasons.append(VerbosityReason("rewrite_too_destructive", f"headings={heading_preservation:.3f}, links={link_preservation:.3f}", ">= 0.5", _WEIGHTS["rewrite_too_destructive"], "Compared rewrite may drop too much structure or links."))
    return metrics, reasons


def _suggestions(reasons: list[VerbosityReason], sections: list[dict[str, Any]], metrics: dict[str, Any]) -> list[str]:
    codes = {r.code for r in reasons}
    suggestions: list[str] = []
    def add(s: str) -> None:
        if s not in suggestions:
            suggestions.append(s)
    if "line_count_high" in codes and metrics.get("heading_count", 0) >= 4:
        add("split_by_heading")
    if "line_count_high" in codes and _has_chronology_headings(sections):
        add("move_chronology_to_archive")
    if "understructured_long_page" in codes or "wall_of_text" in codes:
        add("add_headings")
    if "long_page_low_links" in codes:
        add("add_wikilinks")
    if "section_too_long" in codes:
        add("extract_deep_dive_page")
    verbose_sections = sum(1 for s in sections if s["line_count"] >= 80)
    if verbose_sections >= 2 or ("line_count_high" in codes and metrics.get("heading_count", 0) >= 4):
        add("create_hub_page")
    if "repeated_5grams" in codes or "near_duplicate_sentences" in codes:
        add("deduplicate_repeated_claims")
    if "avg_sentence_too_long" in codes or "long_sentence_ratio_high" in codes or "sentence_extreme" in codes:
        add("shorten_sentences")
    if "low_unique_content_ratio" in codes or "low_lexical_density" in codes:
        add("replace_prose_with_table_or_bullets")
    if "rewrite_not_compact" not in codes and "rewrite_too_destructive" not in codes and "compression_ratio" in metrics:
        add("replace_with_compact_index_and_archive_original")
    return suggestions


def _has_chronology_headings(sections: list[dict[str, Any]]) -> bool:
    pat = re.compile(r"\b(?:20\d{2}|history|chronology|timeline|log|session|update|implementation history)\b", re.I)
    return sum(1 for s in sections if pat.search(s["heading"])) >= 2


def _line_count(text: str) -> int:
    return text.count("\n") + 1 if text else 0


def _line_number_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _round_value(value: float | int | str) -> float | int | str:
    if isinstance(value, float):
        return round(value, 6)
    return value
