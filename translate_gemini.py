"""Parallel translation script powered by Gemini 2.5 Flash."""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from tqdm import tqdm

# --- POWER CONFIGURATION ---
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SOURCE_LANG = "English"
DEFAULT_TARGET_LANG = "Latin American Spanish"

# Keep medium-size batches for speed
MAX_BUDGET_BYTES = 4500

# Number of batches that will be translated concurrently.
# Eight workers are fast and safe for paid accounts.
DEFAULT_MAX_RETRIES = 5
BACKOFF_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0

# Use the compact prompt by default to reduce tokens without losing core rules.
DEFAULT_COMPACT_PROMPT = True
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_QUALITY_RETRIES = 2
STRICT_NO_ENGLISH_RESIDUE = True

# Request timeout (seconds) for each API call. Prevents workers from hanging forever.
DEFAULT_API_TIMEOUT = 120

# How often the on-disk cache is flushed during parallel translation.
# Writing after every batch is wasteful with many workers; this debounces to
# either every N batches or every M seconds, whichever happens first.
CACHE_FLUSH_EVERY_N_BATCHES = 10
CACHE_FLUSH_EVERY_SECONDS = 15.0

PLACEHOLDER_RE = re.compile(r"(%\d+\$[sdif]|%[sdif]|\\n|\\t|\\r)")
PROTECT_TOKEN_RE = re.compile(r"__PROTECT_\d+__")
QUALITY_TOKEN_RE = re.compile(r"__TOK\d+__")
DEFAULT_SKIP_SYMBOL_CONTAINS = ["folder", "path", "dir", "directory"]
DEFAULT_PROTECTED_TERMS = ["Age of Empires III: Wars of Liberty", "My Games"]
DEFAULT_ACRONYM_TERMS = [
    "XP",
    "HP",
    "MP",
    "DPS",
    "AOE",
    "UI",
    "HUD",
    "AI",
    "NPC",
    "FPS",
    "CPU",
    "GPU",
    "APM",
]
DEFAULT_ACRONYM_REGEX = re.compile(
    r"(?<!__)\b(?:"
    + "|".join(DEFAULT_ACRONYM_TERMS)
    + r")(?:\d+)?\b(?![a-z])"
)
DEFAULT_PROTECTED_REGEX = [
    DEFAULT_ACRONYM_REGEX,
    re.compile(r"\bMy\s+Games\b", re.IGNORECASE),
]

# ALL-CAPS tokens that should be allowed to translate (e.g., English number words).
# These sometimes appear in legacy/localized strings and should NOT be treated as acronyms.
DEFAULT_ACRONYM_EXCLUDE = [
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ZERO",
    "TEAM",
]

ENGLISH_RESIDUE_STOPWORDS = {
    "the",
    "of",
    "to",
    "through",
    "enter",
    "address",
    "host",
    "connect",
    "original",
    "version",
    "new",
    "world",
    "trade",
    "center",
}
ENGLISH_RESIDUE_PHRASES = {
    "of the",
    "new world trade center",
}
STRICT_QUALITY_RULES = (
    "STRICT QUALITY RULE\n"
    "Do not leave ANY English articles/prepositions (the/of/to/through/enter/address/host/connect/original/version) "
    "in the output. Translate them to Spanish.\n"
    "Keep names/acronyms and protected tokens unchanged."
)


def target_is_spanish(target_lang: str) -> bool:
    tl = (target_lang or "").lower()
    return ("spanish" in tl) or ("español" in tl) or ("espanol" in tl)

def _strip_quality_tokens(text: str) -> str:
    cleaned = PROTECT_TOKEN_RE.sub(" ", text)
    cleaned = QUALITY_TOKEN_RE.sub(" ", cleaned)
    cleaned = PLACEHOLDER_RE.sub(" ", cleaned)
    return cleaned


def has_english_residue(src: str, out: str, target_lang: str) -> bool:
    if not target_is_spanish(target_lang):
        return False

    cleaned_out = _strip_quality_tokens(out or "")
    lowered = cleaned_out.lower().strip()
    if not lowered:
        return False

    if lowered.startswith("the "):
        return True

    if re.search(r"\bof\s+the\b", lowered):
        return True

    for phrase in ENGLISH_RESIDUE_PHRASES:
        if phrase in lowered:
            return True

    tokens = re.findall(r"\b[a-zA-Z]+\b", cleaned_out)
    for token in tokens:
        if token.lower() in ENGLISH_RESIDUE_STOPWORDS:
            return True

    return False


def _team_casing_repl(match: "re.Match[str]") -> str:
    """Preserve the casing of the matched English 'team' on the Spanish 'equipo'."""
    word = match.group(0)
    if word.isupper():
        return "EQUIPO"
    if word.islower():
        return "equipo"
    return "Equipo"


@dataclass(frozen=True)
class GlossaryEntry:
    """A single Spanish-target terminology rule, the source of truth for BOTH layers.

    - ``prompt_hint`` guides Gemini up front (preventive, "soft").
    - ``output_fixes`` deterministically corrects the translation afterwards (hard guarantee).

    ``output_fixes`` only run when ``source_trigger`` matches the ENGLISH original, so we never
    touch unrelated strings. To add a term, append one entry here — both layers pick it up.
    """

    name: str
    source_trigger: "re.Pattern[str]"
    prompt_hint: str
    # Tuple of (compiled pattern, replacement); replacement is a str or a callable (re.sub style).
    output_fixes: Tuple[Tuple["re.Pattern[str]", object], ...]


SPANISH_GLOSSARY: List[GlossaryEntry] = [
    GlossaryEntry(
        name="home-city",
        source_trigger=re.compile(r"\bHome\s+Cit(?:y|ies)\b", re.IGNORECASE),
        prompt_hint=(
            "- Translate 'Home City' as 'Metrópoli'.\n"
            "- Translate 'Home Cities' as 'Metrópolis'.\n"
            "- If 'Home City' appears inside a longer sentence, still render it as 'Metrópoli/Metrópolis'.\n"
        ),
        output_fixes=(
            # Leftover English occurrences.
            (re.compile(r"\bHome\s+Cities\b", re.IGNORECASE), "Metrópolis"),
            (re.compile(r"\bHome\s+City\b", re.IGNORECASE), "Metrópoli"),
            # The common (but unwanted in WoL Spanish) translation 'ciudad natal' / 'ciudades natales'.
            # Pick singular/plural from the Spanish form itself.
            (re.compile(r"\bciudades\s+natales\b", re.IGNORECASE), "Metrópolis"),
            (re.compile(r"\bciudad\s+natal\b", re.IGNORECASE), "Metrópoli"),
        ),
    ),
    GlossaryEntry(
        name="team",
        source_trigger=re.compile(r"\bteam\b", re.IGNORECASE),
        prompt_hint="- Translate 'team' as 'equipo' (keep the casing of the original word).\n",
        output_fixes=((re.compile(r"\bteam\b", re.IGNORECASE), _team_casing_repl),),
    ),
    GlossaryEntry(
        name="game-ages",
        # Activate whenever the source mentions an 'Age' (named epoch or generic 'Age up').
        # This only gates WHICH strings get post-processed; the output_fixes below are themselves
        # tightly anchored, so a broad trigger here is safe (it never rewrites a bare 'Era').
        source_trigger=re.compile(r"\bAges?\b", re.IGNORECASE),
        prompt_hint=(
            "- Translate the game-epoch names with 'Edad' (NEVER 'Era'): "
            "'Enlightenment Age'→'Edad de la Ilustración', 'National Age'→'Edad Nacional', "
            "'Capital Age'→'Edad Capital', 'Industrial Age'→'Edad Industrial', "
            "'Imperial Age'→'Edad Imperial', 'Golden Age'→'Edad de Oro', 'Stone Age'→'Edad de Piedra'.\n"
            "- When 'Age' means a game epoch (e.g. 'Age up', 'advance to the Age', 'reach the Age'), "
            "translate it as 'Edad', not 'Era'. Do NOT translate the title 'Age of Empires'.\n"
        ),
        output_fixes=(
            # Named ages: accept either 'Era' or 'Edad' from the model and force 'Edad <X>'.
            # Anchored on the second word, so 'ciudad capital' / a bare verb 'era' never match.
            (re.compile(r"\b(?:Era|Edad)\s+(Nacional|Capital|Industrial|Imperial)\b", re.IGNORECASE),
             lambda m: "Edad " + m.group(1).capitalize()),
            (re.compile(r"\b(?:Era|Edad)\s+de\s+la\s+(?:Ilustración|Iluminación)\b", re.IGNORECASE),
             "Edad de la Ilustración"),
            (re.compile(r"\b(?:Era|Edad)\s+de\s+(Oro|Piedra)\b", re.IGNORECASE),
             lambda m: "Edad de " + m.group(1).capitalize()),
            # Generic epoch sense after an advance verb: '...avanzar a la Era' -> '...Edad'.
            # Only the trailing 'Era' (the noun, never the verb 'era' here) is rewritten.
            (re.compile(r"((?:avanz|alcanz|sub|lleg)\w*\s+(?:de\s+|a\s+(?:la\s+)?))Era\b", re.IGNORECASE),
             lambda m: m.group(1) + "Edad"),
        ),
    ),
]


def terminology_overrides_for_target(target_lang: str) -> str:
    """Extra instructions appended to the prompt, only when needed.

    Built from ``SPANISH_GLOSSARY`` so the prompt and the post-process fixes share one source
    of truth. Keep this language-conditional so the script remains global (multi-language).
    """
    if target_is_spanish(target_lang):
        hints = "".join(entry.prompt_hint for entry in SPANISH_GLOSSARY)
        return "TERMINOLOGY OVERRIDES (apply ONLY when target language is Spanish)\n" + hints
    return ""


def apply_postprocess_overrides(original_text: str, translated_text: str, target_lang: str) -> str:
    """Last-mile fixes that must be *conditional on the target language*.

    Driven by ``SPANISH_GLOSSARY``: each entry's ``output_fixes`` run only when its
    ``source_trigger`` matches the English original. This prevents Spanish-specific decisions
    from leaking into other targets like Portuguese, and keeps the fixes scoped to relevant strings.
    """
    if not target_is_spanish(target_lang):
        return translated_text

    out = translated_text
    for entry in SPANISH_GLOSSARY:
        if not entry.source_trigger.search(original_text):
            continue
        for pattern, replacement in entry.output_fixes:
            out = pattern.sub(replacement, out)

    return out


@dataclass(frozen=True)


class PromptConfig:
    """Holds prompt templates for translation requests."""

    compact_template: str
    detailed_template: str

    def build(
        self,
        batch: Sequence[str],
        source_lang: str,
        target_lang: str,
        compact: bool,
        extra_rules: str = "",
    ) -> str:
        template = self.compact_template if compact else self.detailed_template
        prompt = template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            input_list=json.dumps(batch, ensure_ascii=False),
        )
        overrides = terminology_overrides_for_target(target_lang)
        if overrides:
            prompt = prompt + "\n\n" + overrides
        if extra_rules:
            prompt = prompt + "\n\n" + extra_rules
        return prompt


DEFAULT_PROMPT_CONFIG = PromptConfig(
    compact_template=(
        "You are a professional video game localization specialist. "
        "Translate the provided list from {source_lang} to {target_lang} "
        "for a historical video game set between 1789 and 1916 "
        "(Age of Empires III: Wars of Liberty). "
        "Use historically appropriate terminology from the late 18th to early 20th century, "
        "avoid modern slang, and keep the language clear and playable. "
        "DO NOT modernize or embellish the text. "
        "Keep all placeholders (__TOK#, %s, %1$s, %d, \n, \t) unchanged and in the same position. "
        "Treat any __PROTECT_x__ tokens as immutable placeholders. "
        "Treat common gaming acronyms (XP, HP, MP, DPS, AOE, UI, etc.) as non-translatable; "
        "they must remain exactly the same even when adjacent to numbers or symbols. Do NOT treat English number words like ONE/TWO/THREE as acronyms; translate them normally when used as words. "
        "Translate emphasized ALL-CAPS words (e.g., YOU, THEY, THESE) into the target language and keep them in ALL-CAPS, unless they are in the acronym list. "
        "If a string contains escaped newlines (\\n) or bullet characters (•), keep them exactly as written (do not convert \\n to real newlines). "
        "Do NOT merge, split, rephrase, or reorder strings. "
        "Ensure identical source strings receive identical translations. "
        "Return ONLY a valid JSON array of translated strings, "
        "with the exact same number of elements and order as the input. "
        "If a string is empty or contains only placeholders, return it unchanged. "
        "If any rule cannot be followed, return the original string unchanged. "
        "Input list: {input_list}"
    ),
    detailed_template=f"""
    You are an expert video game localization specialist with experience in historical settings.

    TASK
    Translate the following strings from {{source_lang}} to {{target_lang}} for
    “Age of Empires III: Wars of Liberty”, a historical strategy game set between 1789 and 1916.

    ERA & STYLE
    - Historical scope: Napoleonic Wars, Industrial Revolution, World War I.
    - Use accurate military and civilian terminology appropriate to the late 18th, 19th, and early 20th centuries.
    - Avoid modern slang, contemporary expressions, or anachronistic terms.
    - Do NOT use archaic or literary language; the translation must remain clear, concise, and suitable for gameplay.
    - Maintain a neutral, professional tone appropriate for UI and in-game text.

    CONSISTENCY
    - If the same source string appears multiple times, translate it exactly the same way each time.
    - Keep sentences concise; do not add explanations or extra words.

    TECHNICAL RULES (STRICT)
    1. Do NOT translate, modify, reorder, or remove placeholders such as:
       __TOK#, %s, %1$s, %d, \n, \t, and __PROTECT_x__ tokens.
    1.1 Treat common gaming acronyms (XP, HP, MP, DPS, AOE, UI, etc.) as immutable terminology. Do NOT translate or change their character order, even when they appear next to numbers or symbols.
    1.2 Do NOT treat English number words like ONE/TWO/THREE as acronyms; translate them normally when used as words.
    1.3 Translate emphasized ALL-CAPS words (e.g., YOU, THEY, THESE) into the target language and keep them in ALL-CAPS, unless they are in the acronym list.
    2. Preserve literal escape sequences: keep \\n and similar sequences as-is (do NOT convert them to real newlines).
       Maintain bullet characters (•) and surrounding spacing exactly.
    3. Do NOT merge, split, expand, or rephrase strings.
    4. Preserve the original order and number of strings.
    5. Output ONLY a valid JSON array of strings.
    6. The output array MUST have the exact same length and order as the input array.
    7. If a string is empty or contains only placeholders, return it unchanged.
    8. If any rule cannot be followed or the translation is uncertain, return the original string unchanged.

    Input List:
    {{input_list}}
    """,
)


@dataclass(frozen=True)


class DocumentFormat:
    encoding: str
    newline: str
    xml_declaration: bool
    bom: Optional[bytes]


@dataclass(frozen=True)


class TranslationTarget:
    element: ET.Element
    text: str
    symbol: Optional[str]
    skip: bool
    reason: Optional[str] = None
    loc_id: Optional[str] = None


@dataclass


class TranslationStats:
    total_strings: int
    cache_used: int = 0
    api_translated: int = 0
    cache_empty_skipped: int = 0


@dataclass(frozen=True)


class SkipRules:
    symbol_exact: Sequence[str]
    symbol_contains: Sequence[str]
    symbol_regex: Sequence[re.Pattern[str]]
    text_regex: Sequence[re.Pattern[str]]
    enable_path_heuristic: bool = True


def setup_gemini(api_key: str, timeout_seconds: int = DEFAULT_API_TIMEOUT) -> genai.Client:
    """Create a Google GenAI client (google-genai SDK).

    Uses http_options to set a per-request timeout (in milliseconds) so that a stuck
    connection cannot hang a worker forever.
    """
    try:
        http_options = types.HttpOptions(timeout=timeout_seconds * 1000)
        return genai.Client(api_key=api_key, http_options=http_options)
    except (TypeError, AttributeError):
        # Older SDK versions may not support http_options; fall back silently.
        logging.debug("google-genai SDK does not support http_options timeout; using defaults.")
        return genai.Client(api_key=api_key)


def protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}
    idx = 0
    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        key = f"__TOK{idx}__" 
        token_map[key] = match.group(0)
        idx += 1
        return key
    return PLACEHOLDER_RE.sub(repl, text), token_map


def unprotect_tokens(text: str, token_map: Dict[str, str]) -> str:
    for key, value in token_map.items():
        text = text.replace(key, value)
    return text


def protect_phrases(
    text: str,
    phrases: Sequence[str],
    regex_patterns: Sequence[re.Pattern[str]],
    regex_exclude: Optional[Sequence[str]] = None,
) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}
    protected = text
    idx = 0
    exclude_set = {t.upper() for t in (regex_exclude or [])}

    for phrase in phrases:
        if not phrase:
            continue
        while phrase in protected:
            token = f"__PROTECT_{idx}__"
            protected = protected.replace(phrase, token, 1)
            token_map[token] = phrase
            idx += 1

    for pattern in regex_patterns:
        def repl(match: re.Match[str]) -> str:
            nonlocal idx
            token_text = match.group(0)
            if token_text.upper() in exclude_set:
                return token_text
            token = f"__PROTECT_{idx}__"
            token_map[token] = token_text
            idx += 1
            return token

        protected = pattern.sub(repl, protected)

    return protected, token_map


def restore_protected_terms(
    text: str,
    token_map: Dict[str, str],
    original_text: str,
) -> str:
    restored = text
    for token, phrase in token_map.items():
        restored = restored.replace(token, phrase)

    for phrase in token_map.values():
        orig_count = original_text.count(phrase)
        if orig_count and restored.count(phrase) < orig_count:
            logging.warning(
                "Protected phrase missing or altered; restoring from source text."
            )
            return original_text

    if "__PROTECT_" in restored:
        unexpected_tokens = [
            token for token in PROTECT_TOKEN_RE.findall(restored)
            if token not in original_text
        ]
        if unexpected_tokens:
            logging.warning(
                "Unexpected protect tokens found in translation; removing: %s",
                ", ".join(sorted(set(unexpected_tokens))),
            )
            restored = PROTECT_TOKEN_RE.sub("", restored)
            restored = re.sub(r" {2,}", " ", restored).strip()

    return restored


def enforce_acronym_integrity(
    original_text: str,
    candidate_text: str,
    acronym_regex: Optional[re.Pattern[str]] = DEFAULT_ACRONYM_REGEX,
    exclude: Optional[Sequence[str]] = None,
) -> str:
    """Ensure gaming-style acronyms stay exactly as in the source.

    If any acronym detected in the source is missing or altered in the candidate,
    return the original source string to avoid leaking a bad translation.
    """

    exclude_set = {t.upper() for t in (exclude or [])}

    matches = list(acronym_regex.finditer(original_text)) if acronym_regex else []
    if not matches:
        return candidate_text

    for match in matches:
        token = match.group(0)
        if token.upper() in exclude_set:
            continue
        expected = original_text.count(token)
        actual = candidate_text.count(token)
        if actual < expected:
            logging.warning("Acronym '%s' missing or altered; restoring source text.", token)
            return original_text

    return candidate_text


def restore_all_tokens(
    text: str,
    placeholder_map: Dict[str, str],
    protected_map: Dict[str, str],
    original_text: str,
) -> str:
    restored = unprotect_tokens(text, placeholder_map)
    restored = restore_protected_terms(restored, protected_map, original_text)
    return restored


def _normalize_protection(
    protected_terms: Optional[Sequence[str]] = None,
    protected_regex: Optional[Sequence[re.Pattern[str]]] = None,
    acronym_exclude: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[re.Pattern[str]], List[str]]:
    """Merge user-supplied protection settings with the built-in defaults.

    This is the single source of truth used both by translate_strings and by the
    standalone cache-key helper, so the protected text (and therefore the cache key)
    is computed identically no matter who asks. DEFAULT_PROTECTED_TERMS is always
    prepended (protect_phrases is idempotent for already-protected text, so callers
    that already added the defaults — main()/the GUI — get the same key, while callers
    that pass nothing — protected_cache_key()/the compare tab — now match them too).
    """
    protected_terms = list(DEFAULT_PROTECTED_TERMS) + (list(protected_terms) if protected_terms else [])
    protected_regex = list(DEFAULT_PROTECTED_REGEX) + (list(protected_regex) if protected_regex else [])
    acronym_exclude = list(DEFAULT_ACRONYM_EXCLUDE) + (list(acronym_exclude) if acronym_exclude else [])
    return protected_terms, protected_regex, acronym_exclude


def protect_for_cache(
    text: str,
    protected_terms: Sequence[str],
    protected_regex: Sequence[re.Pattern[str]],
    acronym_exclude: Sequence[str],
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Apply phrase + token protection exactly as translate_strings does.

    Returns (protected_text, token_map, phrase_map). The protected_text is the cache key.
    Callers must pass already-normalized lists (see _normalize_protection).
    """
    phrase_protected, phrase_map = protect_phrases(
        text,
        protected_terms,
        protected_regex,
        regex_exclude=acronym_exclude,
    )
    protected_text, token_map = protect_tokens(phrase_protected)
    return protected_text, token_map, phrase_map


def protected_cache_key(
    text: str,
    protected_terms: Optional[Sequence[str]] = None,
    protected_regex: Optional[Sequence[re.Pattern[str]]] = None,
    acronym_exclude: Optional[Sequence[str]] = None,
) -> str:
    """The exact cache key translate_strings would use for `text`.

    Normalizes the protection settings (prepending the built-in defaults) just like
    translate_strings, so external callers (merge seeding, GUI manual edits) write to
    the same keys the engine reads.
    """
    terms, regex, exclude = _normalize_protection(protected_terms, protected_regex, acronym_exclude)
    key, _token_map, _phrase_map = protect_for_cache(text, terms, regex, exclude)
    return key


# Only printf-style format specifiers are load-bearing for the game engine; an
# altered/missing %s or %1$s can crash it. Escaped whitespace (\n/\t/\r) is allowed
# to move around (translations legitimately reorder it), so it is NOT compared here.
# IGNORECASE so an old translation's %S/%D is treated as equivalent to %s/%d.
_FORMAT_SPECIFIER_RE = re.compile(r"%\d+\$[sdif]|%[sdif]", re.IGNORECASE)


def placeholders_compatible(new_source: str, candidate_translation: str) -> bool:
    """True if `candidate_translation` carries the same set of %-format specifiers as `new_source`.

    Used before reusing an old translation against a new source string, so we never
    reuse a translation whose placeholders no longer line up with the (possibly changed)
    source. Comparison is case-insensitive (%S == %s) since old WoL translations vary case.
    """
    def specs(text: str) -> List[str]:
        return sorted(m.lower() for m in _FORMAT_SPECIFIER_RE.findall(text or ""))
    return specs(new_source) == specs(candidate_translation)


def is_all_caps_source(text: str) -> bool:
    if not text:
        return False
    cleaned = QUALITY_TOKEN_RE.sub("", text)
    cleaned = PROTECT_TOKEN_RE.sub("", cleaned)
    cleaned = PLACEHOLDER_RE.sub("", cleaned)
    letters = [ch for ch in cleaned if ch.isalpha()]
    if not letters:
        return False
    return all(ch.isupper() for ch in letters)


def apply_source_casing(source: str, translated: str) -> str:
    if is_all_caps_source(source):
        return translated.upper()
    return translated


def compile_regex_list(patterns: Optional[Sequence[str]]) -> List[re.Pattern[str]]:
    if not patterns:
        return []
    compiled: List[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            logging.warning("Invalid regex skipped (%s): %s", pattern, exc)
    return compiled


def decode_auto(path: Path) -> Tuple[str, str, Optional[bytes]]:
    raw = path.read_bytes()
    bom: Optional[bytes] = None
    if raw.startswith(b"\xff\xfe"):
        bom = b"\xff\xfe"
        return raw[len(bom):].decode("utf-16-le"), "utf-16-le", bom
    if raw.startswith(b"\xfe\xff"):
        bom = b"\xfe\xff"
        return raw[len(bom):].decode("utf-16-be"), "utf-16-be", bom
    if raw.startswith(b"\xef\xbb\xbf"):
        bom = b"\xef\xbb\xbf"
        return raw[len(bom):].decode("utf-8"), "utf-8", bom
    return raw.decode("utf-8"), "utf-8", bom


def detect_declared_encoding(content: str) -> Optional[str]:
    match = re.search(r"<\?xml[^>]*encoding=['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def has_xml_declaration(content: str) -> bool:
    stripped = content.lstrip("\ufeff \t\r\n")
    return stripped.startswith("<?xml")


def detect_newline(content: str) -> str:
    return "\r\n" if "\r\n" in content else "\n"


def is_path_like_text(text: str) -> bool:
    """Heuristic: detect strings that are *primarily* filesystem paths.

    Important: Many WoL strings contain literal escape sequences (\n, \t, ...),
    and/or escaped UI markup like &lt;icon="(58)(WoL\\ui\\...)"&gt; which includes
    backslashes. Those must NOT trigger the path heuristic, or we'd incorrectly
    skip real translatable text.
    """
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # Remove escaped markup blocks (common in WoL UI strings).
    # Example: &lt;icon="(58)(WoL\\ui\\...)"&gt; ... &lt;/font&gt;
    cleaned = re.sub(r"&lt;.*?&gt;", "", stripped)

    # Neutralize common escape sequences so they don't look like backslash paths.
    cleaned = cleaned.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    cleaned = re.sub(r"\\[ntr]", " ", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return False

    # Drive letter / UNC paths.
    if re.match(r"^[a-zA-Z]:[\\/]", cleaned):
        return True
    if cleaned.startswith("\\"):
        return True

    # If it looks like a sentence, it's not a path.
    # (Paths usually don't contain sentence punctuation.)
    if re.search(r"[.;!?]", cleaned):
        return False

    # If it contains printf-style placeholders, it is likely gameplay text, not a path.
    if re.search(r"%\d*\$?[sdif]", cleaned):
        return False

    # Must contain a separator to be considered a path.
    if ("\\" not in cleaned) and ("/" not in cleaned):
        return False

    # If it's extremely long, it's almost certainly UI/help text with embedded markup.
    if len(cleaned) > 160:
        return False

    # Disallow characters that are very uncommon in paths and common in markup/text.
    if re.search(r'[<>"|?*]', cleaned):
        return False

    sep_count = cleaned.count("\\") + cleaned.count("/")
    if sep_count >= 2:
        return True
    if cleaned.endswith("\\") or cleaned.endswith("/"):
        return True

    # For a single separator, require a filename-like suffix to treat it as a path.
    # This avoids misclassifying UI toggles such as "Show/Hide ..." as filesystem paths.
    if sep_count == 1 and re.search(r"[\\/][^\\/\s]+\.[A-Za-z0-9]{1,6}$", cleaned):
        return True

    return False


def yield_batches(strings: Iterable[str], max_budget_bytes: int, max_items: int = 50) -> Iterator[List[str]]:
    batch: List[str] = []
    current_len = 0
    for text in strings:
        text_len = len(text.encode("utf-8")) + 32  # account for quotes and tokens
        if batch and (current_len + text_len > max_budget_bytes or len(batch) >= max_items):
            yield batch
            batch = []
            current_len = 0
        batch.append(text)
        current_len += text_len
    if batch:
        yield batch


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def reconcile_batch_length(batch: Sequence[str], translations: Sequence[str]) -> List[str]:
    """Force the translations list to match the batch size.

    When the model returns a JSON array with missing or extra items, we repair it
    instead of failing the entire batch. Missing entries fall back to the source
    text to keep alignment stable; extra entries are truncated.
    """

    if len(translations) == len(batch):
        return list(translations)

    logging.warning(
        "Length mismatch: Sent %s, Received %s. Repairing response.",
        len(batch),
        len(translations),
    )

    if len(translations) < len(batch):
        missing = len(batch) - len(translations)
        logging.warning("Padding %s missing item(s) with original text.", missing)
        patched = list(translations) + list(batch[len(translations):])
        return patched

    # len(translations) > len(batch)
    extra = len(translations) - len(batch)
    logging.warning("Truncating %s extra item(s) from model response.", extra)
    return list(translations[: len(batch)])


def translate_batch_gemini(
    client: genai.Client,
    batch: Sequence[str],
    source_lang: str,
    target_lang: str,
    compact_prompt: bool,
    extra_rules: str = "",
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> List[str]:

    prompt = prompt_config.build(
        batch,
        source_lang,
        target_lang,
        compact_prompt,
        extra_rules=extra_rules,
    )

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        # Ask the API to return strict JSON whenever possible.
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[str],
        ),
    )

    # The google-genai SDK returns a GenerateContentResponse with .text, and may also include .candidates.
    candidates = getattr(response, "candidates", None)
    if candidates is not None and not candidates:
        raise ValueError("Response without candidates.")

    def _normalized_finish_reason(value: object) -> str:
        if value is None:
            return ""
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name.lower()
        raw_value = getattr(value, "value", None)
        if isinstance(raw_value, str):
            return raw_value.lower()
        return str(value).lower()

    finish_reason = None
    if candidates:
        first_candidate = candidates[0]
        finish_reason = getattr(first_candidate, "finish_reason", None)
        normalized_finish = _normalized_finish_reason(finish_reason)
        if normalized_finish and not ("stop" in normalized_finish or "unspecified" in normalized_finish):
            logging.warning(
                "Unexpected finish_reason (%s) but text was returned; continuing.",
                finish_reason,
            )

    response_text = getattr(response, "text", None)
    if not response_text:
        raise ValueError("Empty response or no usable text returned.")

    cleaned_text = clean_json_response(response_text)
    try:
        translations = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}. Received text: {cleaned_text[:120]}")

    return reconcile_batch_length(batch, translations)


def is_retryable_error(exc: Exception) -> bool:
    transient_signals = (
        "rate limit",
        "temporarily unavailable",
        "try again",
        "deadline exceeded",
        "overloaded",
        "server disconnected",
        "connection reset",
        "connection aborted",
        "timeout",
        "timed out",
        "read operation timed out",
    )
    value_error_retryables = (
        "response without candidates",
        "empty response",
        "invalid json",
        "length mismatch",
        "finish_reason=safety",
        "finish_reason=blocked",
    )
    # API/HTTP-level errors from google-genai that are always transient.
    api_error_types = ("googleapierror", "serviceunavailable", "resourceexhausted", "internalservererror")

    message = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if any(signal in message for signal in transient_signals):
        return True

    if any(hint in message for hint in ("finish_reason=safety", "finish_reason=blocked", "safety", "blocked")):
        return True

    if isinstance(exc, ValueError) and any(signal in message for signal in value_error_retryables):
        return True

    # Retry known transient API error classes.
    if any(api_type in exc_type for api_type in api_error_types):
        return True

    # Do NOT retry programming errors (AttributeError, TypeError, KeyError, etc.)
    # that would loop forever without any chance of recovery.
    return False


def translate_batch_with_retry(
    client,
    batch,
    source,
    target,
    max_retries,
    compact_prompt: bool,
    prompt_config: PromptConfig,
    strict_no_english_residue: bool,
    max_quality_retries: int = DEFAULT_MAX_QUALITY_RETRIES,
) -> List[str]:
    attempt = 0
    quality_attempt = 0
    last_partial: Optional[List[str]] = None
    quality_prompt_compact = compact_prompt
    extra_rules = ""
    while True:
        try:
            translations = translate_batch_gemini(
                client,
                batch,
                source,
                target,
                quality_prompt_compact,
                extra_rules=extra_rules,
                prompt_config=prompt_config,
            )
            # Quality gate: check for English residue and retry with stricter rules if needed.
            if strict_no_english_residue and target_is_spanish(target):
                residue = None
                for src_text, out_text in zip(batch, translations):
                    if has_english_residue(src_text, out_text, target):
                        residue = (src_text, out_text)
                        break
                if residue:
                    if quality_attempt < max_quality_retries:
                        quality_attempt += 1
                        quality_prompt_compact = False
                        extra_rules = STRICT_QUALITY_RULES
                        logging.warning(
                            "Quality retry %s/%s: English residue detected. src=%s out=%s",
                            quality_attempt,
                            max_quality_retries,
                            residue[0],
                            residue[1],
                        )
                        continue
                    logging.warning(
                        "Quality retries exhausted; English residue remains. src=%s out=%s",
                        residue[0],
                        residue[1],
                    )
            return translations
        except Exception as exc:
            attempt += 1
            partial = getattr(exc, "partial_translations", None)
            if partial:
                last_partial = partial
            retryable = is_retryable_error(exc)
            logging.warning(
                "Batch error (attempt %s/%s, retryable=%s): %s",
                attempt,
                max_retries,
                retryable,
                exc,
            )
            if (not retryable) or attempt > max_retries:
                logging.error("Giving up on batch after %s attempt(s): %s", attempt, exc)
                if last_partial and len(last_partial) == len(batch):
                    return list(last_partial)
                # Raise so the caller can avoid caching a fallback result.
                err = RuntimeError(f"Batch failed after {attempt} attempt(s): {exc}")
                setattr(err, "failed_batch", list(batch))
                raise err

            backoff = min(BACKOFF_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            backoff += random.uniform(0, BACKOFF_SECONDS)
            logging.info("Retrying batch in %.1fs...", backoff)
            time.sleep(backoff)


def _prune_empty_cache(cache: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of the cache without empty-string placeholders.

    Empty placeholders are internal runtime markers ("string enqueued for translation")
    that have no meaning once the run ends. Persisting them to disk would cause
    future runs to skip those strings instead of retrying them.
    """
    return {key: value for key, value in cache.items() if value and value.strip()}


def _write_cache_atomic(cache_path: Path, cache: Dict[str, str]) -> None:
    """Write the cache JSON atomically so a crash mid-write cannot corrupt it.

    Writes to a sibling temp file and renames into place. On POSIX, rename is atomic;
    on Windows, Path.replace() provides equivalent semantics.
    """
    data = json.dumps(_prune_empty_cache(cache), ensure_ascii=False, indent=2)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        tmp_path.write_text(data, encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception:
        # Best-effort cleanup; do not mask the original error.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def translate_strings(
    inners: Iterable[str],
    api_key: Optional[str],
    source_lang: str,
    target_lang: str,
    max_budget_bytes: int = MAX_BUDGET_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_workers: int = DEFAULT_MAX_WORKERS,
    compact_prompt: bool = DEFAULT_COMPACT_PROMPT,
    progress_callback: Optional[Callable[[Sequence[str]], None]] = None,
    cache_path: Optional[Path] = None,
    existing_translations: Optional[Sequence[str]] = None,
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
    protected_terms: Optional[Sequence[str]] = None,
    protected_regex: Optional[Sequence[re.Pattern[str]]] = None,
    acronym_exclude: Optional[Sequence[str]] = None,
    strict_no_english_residue: Optional[bool] = None,
    cache_only: bool = False,
    retry_empty_cache: bool = False,
    api_timeout_seconds: int = DEFAULT_API_TIMEOUT,
    batch_progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[List[str], TranslationStats]:
    
    inners_list = list(inners)
    stats = TranslationStats(total_strings=len(inners_list))

    protected_terms, protected_regex, acronym_exclude = _normalize_protection(
        protected_terms, protected_regex, acronym_exclude
    )
    strict_no_english_residue = (
        STRICT_NO_ENGLISH_RESIDUE and target_is_spanish(target_lang)
        if strict_no_english_residue is None
        else strict_no_english_residue
    )

    protected: List[str] = []
    token_maps: List[Dict[str, str]] = []
    phrase_maps: List[Dict[str, str]] = []
    original_texts: List[str] = []
    translations: List[str] = []
    indexes_by_protected: Dict[str, List[int]] = {}

    cache: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Unable to load previous cache (%s): %s", cache_path, exc)
            cache = {}

    for idx, inner in enumerate(inners_list):
        protected_text, token_map, phrase_map = protect_for_cache(
            inner, protected_terms, protected_regex, acronym_exclude
        )
        protected.append(protected_text)
        token_maps.append(token_map)
        phrase_maps.append(phrase_map)
        original_texts.append(inner)

        initial_translation = inner
        if existing_translations and idx < len(existing_translations):
            candidate = existing_translations[idx]
            if candidate and candidate.strip():
                initial_translation = candidate
                if candidate != inner:
                    cache.setdefault(protected_text, candidate)

        translations.append(initial_translation)
        indexes_by_protected.setdefault(protected_text, []).append(idx)
    unique_to_translate: List[str] = []
    already_enqueued: set[str] = set()

    for text in protected:
        if not text.strip():
            cache[text] = text
            # Propagate empty text as-is to every position.
            for idx in indexes_by_protected.get(text, []):
                translations[idx] = restore_all_tokens(
                    text, token_maps[idx], phrase_maps[idx], original_texts[idx]
                )
            continue

        cached_value = cache.get(text)

        if cached_value and cached_value.strip():
            # We already had a cached translation: reuse it everywhere and skip re-translation.
            stats.cache_used += len(indexes_by_protected.get(text, []))
            for idx in indexes_by_protected.get(text, []):
                restored = restore_all_tokens(
                    cached_value, token_maps[idx], phrase_maps[idx], original_texts[idx]
                )
                restored = apply_postprocess_overrides(original_texts[idx], restored, target_lang)
                restored = enforce_acronym_integrity(original_texts[idx], restored, exclude=acronym_exclude)
                restored = apply_source_casing(original_texts[idx], restored)
                translations[idx] = restored
            continue

        if cached_value is not None and not cached_value.strip():
            if not retry_empty_cache or cache_only:
                for idx in indexes_by_protected.get(text, []):
                    restored = restore_all_tokens(
                        text, token_maps[idx], phrase_maps[idx], original_texts[idx]
                    )
                    translations[idx] = restored
                stats.cache_empty_skipped += len(indexes_by_protected.get(text, []))
                continue

        if cache_only:
            for idx in indexes_by_protected.get(text, []):
                restored = restore_all_tokens(
                    text, token_maps[idx], phrase_maps[idx], original_texts[idx]
                )
                translations[idx] = restored
            continue

        # If there is no cache (or it is empty), register an entry and queue it for translation,
        # avoiding duplicates.
        if text not in cache:
            cache[text] = ""
        if text not in already_enqueued:
            already_enqueued.add(text)
            unique_to_translate.append(text)

    if cache_only or not unique_to_translate:
        if cache_path:
            try:
                _write_cache_atomic(cache_path, cache)
            except Exception as exc:
                logging.warning("Failed to write cache file: %s", exc)
        return translations, stats

    if not api_key:
        raise RuntimeError(
            "Missing --api-key: translation required for uncached strings."
        )

    client = setup_gemini(api_key, timeout_seconds=api_timeout_seconds)

    _cache_lock = threading.Lock()

    # Build all batches
    batches = list(yield_batches(unique_to_translate, max_budget_bytes))

    # Map to sort results: {batch_index: [original_texts]}
    batch_map = {i: batch for i, batch in enumerate(batches)}
    total_batches = len(batches)

    print(f"🚀 Starting MULTITHREAD engine: {max_workers} concurrent workers...")

    # Debounce state for cache persistence.
    _batches_completed = 0
    _last_flush_time = time.monotonic()

    # --- PARALLEL PROCESSING ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Launch all tasks
        future_to_batch_idx = {
            executor.submit(
                translate_batch_with_retry,
                client,
                batch,
                source_lang,
                target_lang,
                max_retries,
                compact_prompt,
                prompt_config,
                strict_no_english_residue,
            ): idx
            for idx, batch in batch_map.items()
        }

        # Process tasks as they complete
        _batches_done = 0
        for future in tqdm(
            as_completed(future_to_batch_idx),
            total=total_batches,
            desc="Translating in Parallel",
            unit="batch",
        ):
            # Check for cancellation: cancel pending futures and break out.
            if cancel_event is not None and cancel_event.is_set():
                for pending_future in future_to_batch_idx:
                    if not pending_future.done():
                        pending_future.cancel()
                logging.warning("Translation cancelled by user after %s/%s batches.",
                                _batches_done, total_batches)
                break

            batch_idx = future_to_batch_idx[future]
            original_batch = batch_map[batch_idx]

            try:
                translated_batch = future.result()
            except Exception as exc:
                logging.error(
                    "Unhandled thread exception (batch %s, %s items): %s",
                    batch_idx,
                    len(original_batch),
                    exc,
                )
                # Do NOT poison the cache with fallback originals; keep them retryable on the next run.
                translated_batch = None

            # Store in cache and update main list
            if translated_batch is None:
                # Mark these items as not-yet-translated (empty cache) so a rerun will retry them.
                for original in original_batch:
                    cache[original] = ""
                # Skip updating translations from this batch.
                _batches_done += 1
                if batch_progress_callback is not None:
                    try:
                        batch_progress_callback(_batches_done, total_batches)
                    except Exception:
                        pass
                continue

            with _cache_lock:
                for original, translated_item in zip(original_batch, translated_batch):
                    if strict_no_english_residue and has_english_residue(original, translated_item, target_lang):
                        logging.warning(
                            "Skipping cache/write due to English residue. src=%s out=%s",
                            original,
                            translated_item,
                        )
                        cache[original] = ""
                        continue
                    stats.api_translated += len(indexes_by_protected.get(original, []))
                    cache[original] = translated_item
                    for idx in indexes_by_protected.get(original, []):
                        restored = restore_all_tokens(
                            translated_item,
                            token_maps[idx],
                            phrase_maps[idx],
                            original_texts[idx],
                        )
                        restored = apply_postprocess_overrides(original_texts[idx], restored, target_lang)
                        restored = enforce_acronym_integrity(original_texts[idx], restored, exclude=acronym_exclude)
                        restored = apply_source_casing(original_texts[idx], restored)
                        translations[idx] = restored

            _batches_done += 1
            if batch_progress_callback is not None:
                try:
                    batch_progress_callback(_batches_done, total_batches)
                except Exception:
                    pass

            # Save partial progress only every N batches or every M seconds (debounce).
            _batches_completed += 1
            _now = time.monotonic()
            _should_flush = (
                _batches_completed >= CACHE_FLUSH_EVERY_N_BATCHES
                or (_now - _last_flush_time) >= CACHE_FLUSH_EVERY_SECONDS
            )
            if _should_flush and cache_path:
                with _cache_lock:
                    try:
                        _write_cache_atomic(cache_path, cache)
                        _last_flush_time = _now
                        _batches_completed = 0
                    except Exception as exc:
                        logging.warning("Could not persist cache for batch %s: %s", batch_idx, exc)

            if progress_callback:
                progress_callback(list(translations))

    # Final cache flush after all batches complete so we never lose the last in-memory updates.
    if cache_path:
        with _cache_lock:
            try:
                _write_cache_atomic(cache_path, cache)
            except Exception as exc:
                logging.warning("Failed final cache write: %s", exc)

    return translations, stats

# --- XML Utils ---


class CommentedTreeBuilder(ET.TreeBuilder):
    """TreeBuilder that preserves XML comments while parsing."""

    def comment(self, data):
        self.start(ET.Comment, {})
        self.data(data)
        self.end(ET.Comment)


def parse_strings_xml(path: Path) -> Tuple[ET.ElementTree, DocumentFormat]:
    content, detected_encoding, bom = decode_auto(path)
    declared = detect_declared_encoding(content)
    encoding = declared if declared else detected_encoding
    xml_decl = has_xml_declaration(content)
    newline = detect_newline(content)
    parser = ET.XMLParser(target=CommentedTreeBuilder())
    return (
        ET.ElementTree(ET.fromstring(content, parser=parser)),
        DocumentFormat(
            encoding=encoding,
            newline=newline,
            xml_declaration=xml_decl,
            bom=bom,
        ),
    )


def should_skip_element(elem: ET.Element, rules: SkipRules) -> Tuple[bool, Optional[str]]:
    text = elem.text or ""
    symbol = elem.attrib.get("symbol")
    symbol_lower = symbol.lower() if symbol else ""

    # Mandatory skip for folder-like symbols.
    if symbol and ("folder" in symbol_lower or symbol_lower.endswith("folder")):
        return True, "symbol-folder"

    normalized_exact = {s.lower() for s in rules.symbol_exact}
    normalized_contains = [s.lower() for s in rules.symbol_contains]

    if symbol and symbol_lower in normalized_exact:
        return True, "symbol-exact"

    if symbol and any(token in symbol_lower for token in normalized_contains):
        return True, "symbol-contains"

    if symbol and any(pattern.search(symbol) for pattern in rules.symbol_regex):
        return True, "symbol-regex"

    if any(pattern.search(text or "") for pattern in rules.text_regex):
        return True, "text-regex"

    if rules.enable_path_heuristic and is_path_like_text(text):
        return True, "path-like-text"

    return False, None


def iter_translatable_elements(root: ET.Element, skip_rules: SkipRules) -> Iterator[TranslationTarget]:
    def tag_matches(tag: str, name: str) -> bool:
        if not isinstance(tag, str):
            return False
        # Some special nodes (e.g., comments) can leak with an unexpected ``tag``;
        # use ``split`` defensively to avoid AttributeError when the tag is not a normal string.
        splitter = getattr(tag, "split", None)
        if splitter is None:
            return False
        return splitter("}")[-1].lower() == name

    def build_target(elem: ET.Element) -> TranslationTarget:
        skip, reason = should_skip_element(elem, skip_rules)
        return TranslationTarget(
            element=elem,
            text=elem.text or "",
            symbol=elem.attrib.get("symbol"),
            skip=skip,
            reason=reason,
            loc_id=elem.attrib.get("_locID"),
        )

    for elem in root.iter():
        if tag_matches(elem.tag, "string"):
            yield build_target(elem)
        elif tag_matches(elem.tag, "plurals"):
            for item in elem:
                if tag_matches(item.tag, "item"):
                    yield build_target(item)


def extract_texts(elements: Iterable[TranslationTarget]) -> List[str]:
    return [elem.text for elem in elements]


def indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + "  " * level
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i
    else:
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i


def update_elements_text(elements: Iterable[ET.Element], texts: Sequence[str]) -> None:
    for elem, text in zip(elements, texts):
        elem.text = text


def strip_known_bom(data: bytes) -> Tuple[bytes, Optional[bytes]]:
    for bom in (b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf"):
        if data.startswith(bom):
            return data[len(bom):], bom
    return data, None


def resolve_write_encoding(fmt: DocumentFormat) -> str:
    encoding_lower = fmt.encoding.lower()
    if fmt.bom == b"\xff\xfe":
        return "utf-16-le"
    if fmt.bom == b"\xfe\xff":
        return "utf-16-be"
    if fmt.bom == b"\xef\xbb\xbf":
        return "utf-8"
    if encoding_lower == "utf-8-sig":
        return "utf-8"
    if encoding_lower == "utf-16":
        return "utf-16-le"
    return fmt.encoding


def serialize_tree(tree: ET.ElementTree, elements, texts, fmt: DocumentFormat) -> bytes:
    update_elements_text(elements, texts)
    indent(tree.getroot())

    buffer = io.BytesIO()
    tree.write(
        buffer,
        encoding=fmt.encoding,
        xml_declaration=fmt.xml_declaration,
        short_empty_elements=False,
    )

    serialized_bytes, _ = strip_known_bom(buffer.getvalue())
    serialized_text = serialized_bytes.decode(resolve_write_encoding(fmt), errors="replace")
    if fmt.newline != "\n":
        serialized_text = serialized_text.replace("\n", fmt.newline)

    encoded = serialized_text.encode(resolve_write_encoding(fmt))
    if fmt.bom:
        encoded = fmt.bom + encoded
    return encoded


def atomic_write(data: bytes, output: Path) -> None:
    temp_path = output.with_name(output.name + ".tmp")
    with temp_path.open("wb") as fp:
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temp_path, output)


def print_diagnostics(path: Path, fmt: DocumentFormat) -> None:
    try:
        raw = path.read_bytes()
    except Exception as exc:
        logging.warning("Diagnostic read failed for %s: %s", path, exc)
        return
    first_bytes = " ".join(f"{b:02x}" for b in raw[:16])
    bom_label = "none"
    for label, bom in (("FF FE", b"\xff\xfe"), ("FE FF", b"\xfe\xff"), ("UTF-8 BOM", b"\xef\xbb\xbf")):
        if raw.startswith(bom):
            bom_label = label
            break
    encoding_used = resolve_write_encoding(fmt)
    print(
        f"🔍 Diagnostics -> encoding={encoding_used}, bom={bom_label}, "
        f"first16={first_bytes}, size={len(raw)} bytes"
    )


def write_output_snapshot(tree, elements, texts, output: Path, fmt: DocumentFormat, diagnostic: bool = False):
    serialized = serialize_tree(tree, elements, texts, fmt)
    atomic_write(serialized, output)
    if diagnostic:
        print_diagnostics(output, fmt)


def assemble_full_texts(
    targets: Sequence[TranslationTarget],
    translated: Sequence[str],
    enforce_skip_integrity: bool = True,
) -> List[str]:
    merged: List[str] = []
    translated_iter = iter(translated)
    for target in targets:
        if target.skip:
            merged.append(target.text)
            continue
        try:
            merged.append(next(translated_iter))
        except StopIteration:
            raise ValueError("Not enough translated items to map back to elements.")
    try:
        next(translated_iter)
        raise ValueError("Too many translated items supplied.")
    except StopIteration:
        pass

    if enforce_skip_integrity:
        for idx, target in enumerate(targets):
            if target.skip and merged[idx] != target.text:
                logging.warning(
                    "Restoring skipped element (symbol=%s, reason=%s) to original text.",
                    target.symbol,
                    target.reason,
                )
                merged[idx] = target.text
    return merged


def load_existing_translations(path: Path, reference_count: int, skip_rules: SkipRules) -> Optional[List[str]]:
    if not path.exists():
        return None

    try:
        existing_tree, _ = parse_strings_xml(path)
        existing_elements = list(iter_translatable_elements(existing_tree.getroot(), skip_rules))
        if len(existing_elements) != reference_count:
            logging.warning(
                "Existing output file (%s) length mismatch (expected %s, found %s). Ignoring.",
                path,
                reference_count,
                len(existing_elements),
            )
            return None
        return extract_texts(existing_elements)
    except Exception as exc:
        logging.warning("Could not load previous translations from %s: %s", path, exc)
        return None


# --- Merge by _locID (carry an old translation onto a new source version) ---


@dataclass(frozen=True)


class MergeEntry:
    """One new-version string and what we decided to do with it.

    Aligned 1:1 (and in order) with the NON-SKIP translatable targets of the new file.
    """
    loc_id: Optional[str]
    new_source: str
    status: str            # "unchanged" | "changed" | "new"
    draft: Optional[str]   # old translation to show as a starting point (may be None)
    seed: Optional[str]    # value safe to auto-seed into the cache; None if not safe
    reason: str            # "reuse-safe" | "changed-needs-review" | "new-needs-translation"
                           # | "placeholder-mismatch" | "old-equals-source" | "no-old-translation"
    english_old: Optional[str] = None  # the old-version English text (for a new-vs-old diff view)


@dataclass(frozen=True)


class MergeReport:
    entries: List[MergeEntry]
    counts: Dict[str, int]


def build_locid_index(targets: Sequence[TranslationTarget]) -> Dict[str, List[str]]:
    """Map _locID -> list of texts (document order). Targets without _locID are skipped."""
    index: Dict[str, List[str]] = {}
    for target in targets:
        if target.loc_id:
            index.setdefault(target.loc_id, []).append(target.text)
    return index


def build_source_content_map(
    old_source_targets: Sequence[TranslationTarget],
    old_trans_targets: Sequence[TranslationTarget],
) -> Dict[str, str]:
    """Build a robust {old_english_text: old_translation} map by joining on _locID.

    This is the content fallback used when a new string has no usable _locID (missing or
    duplicated). It only pairs entries whose _locID exists in BOTH files, aligning by
    position within a duplicate-id group.
    """
    src_by_id = build_locid_index(old_source_targets)
    trans_by_id = build_locid_index(old_trans_targets)
    content_map: Dict[str, str] = {}
    for loc_id, src_list in src_by_id.items():
        trans_list = trans_by_id.get(loc_id)
        if not trans_list:
            continue
        for i, src_text in enumerate(src_list):
            if i < len(trans_list):
                content_map.setdefault(src_text, trans_list[i])
    return content_map


def _pick_from_group(
    new_source: str,
    src_list: Sequence[str],
    trans_list: Sequence[str],
    cursor: Dict[str, int],
    loc_id: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a duplicated _locID: prefer an exact source-content match, else consume positionally."""
    for i, src_text in enumerate(src_list):
        if src_text == new_source and i < len(trans_list):
            return src_text, trans_list[i]
    pos = cursor.get(loc_id, 0)
    cursor[loc_id] = pos + 1
    src = src_list[pos] if pos < len(src_list) else None
    trans = trans_list[pos] if pos < len(trans_list) else None
    return src, trans


_MARKUP_RE = re.compile(r"<[^>]*>")
# Tokens with no translatable content: markup, escaped UI blocks, printf-style specifiers
# (including WoL's %1s / %2s form), and escaped whitespace.
_NONTRANSLATABLE_RE = re.compile(r"<[^>]*>|&lt;.*?&gt;|%\d*\$?[a-zA-Z]|\\[ntr]")


def _normalize_for_change(text: str) -> str:
    """Collapse *cosmetic-only* differences for change detection.

    Removes markup tags and ALL whitespace, then lowercases, so that an English string that
    only gained `<color=...>` wrapping, changed capitalization, or shifted spacing compares
    equal to its old version (its existing translation is still usable). Whitespace is dropped
    entirely so that replacing a tag with a space cannot create a spurious difference.
    """
    stripped = _MARKUP_RE.sub(" ", text or "")
    return re.sub(r"\s+", "", stripped).lower()


def _has_translatable_text(text: str) -> bool:
    """True if any alphabetic content remains after removing markup/placeholders/tokens.

    Used to keep purely non-translatable strings (format strings, filenames-as-markup,
    pure placeholder lines) out of the "needs translation" bucket.
    """
    cleaned = _NONTRANSLATABLE_RE.sub(" ", text or "")
    cleaned = PROTECT_TOKEN_RE.sub(" ", cleaned)
    cleaned = QUALITY_TOKEN_RE.sub(" ", cleaned)
    return any(ch.isalpha() for ch in cleaned)


def _classify_merge_entry(
    loc_id: Optional[str],
    new_source: str,
    english_old: Optional[str],
    candidate: Optional[str],
) -> MergeEntry:
    """Decide status/seed/draft for one new string given its old English + old translation.

    The English can relate to the new source in four ways:
      * unchanged  – byte-identical.
      * cosmetic   – differs only in markup (`<color=...>`), case, or whitespace; the old
                     translation is still usable, so we reuse it (reason "reuse-cosmetic").
                     The new markup is NOT re-applied — it is cosmetic.
      * changed    – the text really changed; re-translate.
      * new        – the _locID did not exist before.
    """
    if english_old is not None:
        if english_old == new_source:
            relation = "unchanged"
        elif _normalize_for_change(english_old) == _normalize_for_change(new_source):
            relation = "cosmetic"
        else:
            relation = "changed"
    else:
        # No old English text to compare. A content match means an identical old source
        # carried this translation, so it is effectively unchanged-by-content.
        relation = "unchanged" if (candidate is not None) else "new"

    draft = candidate if (candidate and candidate.strip()) else None
    seed: Optional[str] = None

    # The old translation is "just the English source" if it equals the new source, OR (for a
    # cosmetic change, where the markup differs) if it normalizes to the same text. This catches
    # strings the previous translator left in English so we never seed English as a translation.
    draft_is_source = bool(draft) and (
        draft == new_source
        or (relation == "cosmetic" and _normalize_for_change(draft) == _normalize_for_change(new_source))
    )

    if relation in ("unchanged", "cosmetic"):
        if not draft:
            status, reason = "unchanged", "no-old-translation"
        elif draft_is_source:
            # Old "translation" is identical to the source (left in English).
            if not _has_translatable_text(new_source):
                seed = new_source                 # nothing to translate -> keep as-is
                status, reason = "unchanged", "kept-as-source"
            else:
                status, reason = "unchanged", "old-equals-source"
        elif not placeholders_compatible(new_source, draft):
            status = "changed" if relation == "cosmetic" else "unchanged"
            reason = "placeholder-mismatch"
        elif relation == "cosmetic":
            seed = draft                          # reuse; only formatting/case changed
            status, reason = "changed", "reuse-cosmetic"
        else:
            seed = draft
            status, reason = "unchanged", "reuse-safe"
    elif relation == "changed":
        status, reason = "changed", "changed-needs-review"  # draft shown, never seeded
    else:
        status, reason = "new", "new-needs-translation"

    return MergeEntry(
        loc_id=loc_id,
        new_source=new_source,
        status=status,
        draft=draft,
        seed=seed,
        reason=reason,
        english_old=english_old,
    )


def merge_by_locid(
    new_targets: Sequence[TranslationTarget],
    old_source_targets: Sequence[TranslationTarget],
    old_trans_targets: Sequence[TranslationTarget],
) -> MergeReport:
    """Carry an old translation onto a new source version, matching by _locID.

    Inputs are the NON-SKIP translatable targets of, respectively: the new English file,
    the old English file, and the old translated file. The report is aligned 1:1 with
    ``new_targets``. Matching precedence per new string:
      1. Unique _locID present in the old translation -> direct lookup (+ old English for
         change detection).
      2. Duplicated _locID -> resolve within the id group (exact content, else positional).
      3. No usable _locID -> content fallback (old_english_text -> translation).
      4. Otherwise -> new.
    """
    old_src_by_id = build_locid_index(old_source_targets)
    old_trans_by_id = build_locid_index(old_trans_targets)
    content_map = build_source_content_map(old_source_targets, old_trans_targets)

    src_cursor: Dict[str, int] = {}

    entries: List[MergeEntry] = []
    for target in new_targets:
        new_source = target.text
        loc_id = target.loc_id
        english_old: Optional[str] = None
        candidate: Optional[str] = None

        if loc_id and loc_id in old_trans_by_id:
            trans_list = old_trans_by_id[loc_id]
            src_list = old_src_by_id.get(loc_id, [])
            if len(trans_list) == 1 and len(src_list) <= 1:
                candidate = trans_list[0]
                english_old = src_list[0] if src_list else None
            else:
                english_old, candidate = _pick_from_group(
                    new_source, src_list, trans_list, src_cursor, loc_id
                )

        if candidate is None:
            # Content fallback: an old English string identical to this new source.
            candidate = content_map.get(new_source)

        entries.append(_classify_merge_entry(loc_id, new_source, english_old, candidate))

    counts: Dict[str, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
        counts[f"reason:{entry.reason}"] = counts.get(f"reason:{entry.reason}", 0) + 1
    counts["seeded"] = sum(1 for entry in entries if entry.seed is not None)
    counts["total"] = len(entries)

    return MergeReport(entries=entries, counts=counts)


def seed_list_from_report(report: MergeReport) -> List[str]:
    """Turn a MergeReport into an ``existing_translations`` list for translate_strings.

    Only reuse-safe entries carry a value; everything else is "" (the engine's
    "needs translation" sentinel), so changed/new strings are translated, not reused.
    """
    return [entry.seed if entry.seed is not None else "" for entry in report.entries]


def write_merge_report(path: Path, report: MergeReport) -> None:
    """Write a JSON report: full counts plus every entry that still needs attention."""
    needs_attention = [
        {
            "loc_id": entry.loc_id,
            "status": entry.status,
            "reason": entry.reason,
            "new_source": entry.new_source,
            "draft": entry.draft,
        }
        for entry in report.entries
        if entry.seed is None
    ]
    data = {
        "counts": report.counts,
        "needs_attention_count": len(needs_attention),
        "needs_attention": needs_attention,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_skip_rules(args: argparse.Namespace) -> SkipRules:
    symbol_contains = list(DEFAULT_SKIP_SYMBOL_CONTAINS)
    if args.skip_symbol_contains:
        symbol_contains.extend(args.skip_symbol_contains)
    return SkipRules(
        symbol_exact=args.skip_symbol or [],
        symbol_contains=symbol_contains,
        symbol_regex=compile_regex_list(args.skip_symbol_regex),
        text_regex=compile_regex_list(args.skip_text_regex),
        enable_path_heuristic=not args.no_path_heuristic,
    )


def self_test_quality_gate() -> None:
    target_lang = "Spanish"
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Quality gate self-test failed: {message}")

    src1 = "The Torre del Oro"
    out1_bad = "The Torre del Oro"
    out1_good = "La Torre del Oro"
    _assert(has_english_residue(src1, out1_bad, target_lang), "expected residue for 'The Torre del Oro'")
    _assert(not has_english_residue(src1, out1_good, target_lang), "expected no residue for 'La Torre del Oro'")

    src2 = "Enter the IP address of the host to connect through direct IP."
    out2_bad = "Enter the IP address of the host to connect through direct IP."
    out2_good = "Introduce la dirección IP del host para conectar mediante IP directa."
    _assert(has_english_residue(src2, out2_bad, target_lang), "expected residue for IP address prompt")
    _assert("IP" in out2_good, "expected IP to remain unchanged")
    _assert(not has_english_residue(src2, out2_good, target_lang), "expected no residue in Spanish translation")

    print("✅ Quality gate self-test passed.")


def self_test_source_casing() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Source casing self-test failed: {message}")

    _assert(
        apply_source_casing("TEAM", "EQUIPO") == "EQUIPO",
        "TEAM should keep translated output in uppercase",
    )
    _assert(
        apply_source_casing("Team", "Equipo") == "Equipo",
        "Team should not force uppercase in translated output",
    )
    upper_ip = apply_source_casing("ENTER THE IP ADDRESS", "Ingrese la dirección IP")
    _assert(upper_ip == upper_ip.upper(), "ENTER THE IP ADDRESS should force uppercase output")
    _assert("IP" in upper_ip, "Acronym IP should remain intact")

    print("✅ Source casing self-test passed.")


def self_test_glossary() -> None:
    target_lang = "Spanish"

    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Glossary self-test failed: {message}")

    f = apply_postprocess_overrides

    # Named ages: 'Era <X>' (and lowercase 'edad') get forced to the canonical 'Edad <X>'.
    cases = [
        ("II: National Age", "II: Era Nacional", "Edad Nacional"),
        ("Build in the Industrial Age", "Construye en la era industrial", "Edad Industrial"),
        ("You reach the Imperial Age!", "¡Alcanzas la Era Imperial!", "Edad Imperial"),
        ("Advance to the Capital Age", "Avanza a la Era Capital", "Edad Capital"),
        ("I: Enlightenment Age", "I: Era de la Ilustración", "Edad de la Ilustración"),
        ("A new Golden Age", "Una nueva Era de Oro", "Edad de Oro"),
        ("The Stone Age", "La Era de Piedra", "Edad de Piedra"),
    ]
    for src, bad, expected in cases:
        out = f(src, bad, target_lang)
        _assert(expected in out, f"expected '{expected}' in {out!r} (source {src!r})")
        _assert(" Era " not in f" {out} ", f"'Era' should be gone from {out!r}")

    # Generic epoch sense after an advance verb: only the trailing noun 'Era' becomes 'Edad'.
    adv = f(
        "Allows you to send the Cavalier, who advances you to the National Age.",
        "...quien te avanza a la Era Nacional.",
        target_lang,
    )
    _assert("Edad Nacional" in adv, f"expected 'Edad Nacional' in {adv!r}")
    adv_up = f("Age up quickly", "avanza de Era rápido", target_lang)
    _assert("avanza de Edad" in adv_up, f"expected 'avanza de Edad' in {adv_up!r}")

    # CRITICAL negative case: a bare 'era' (the verb 'was') must NEVER become 'Edad'.
    src_verb = "In a bygone age the world was different"
    out_verb = "En una época pasada el mundo era diferente"
    _assert(
        f(src_verb, out_verb, target_lang) == out_verb,
        f"verb 'era' must stay untouched, got {f(src_verb, out_verb, target_lang)!r}",
    )

    # Regression: Home City / team still work; non-Spanish targets are untouched.
    _assert(f("Go to your Home City", "Ve a tu ciudad natal", target_lang) == "Ve a tu Metrópoli",
            "Home City should map to Metrópoli")
    _assert(f("Manage your Home Cities", "Gestiona tus ciudades natales", target_lang) == "Gestiona tus Metrópolis",
            "Home Cities should map to Metrópolis")
    _assert(f("TEAM bonus", "Bono de TEAM", target_lang) == "Bono de EQUIPO",
            "TEAM should become EQUIPO")
    _assert(f("Team bonus", "Bono de Team", target_lang) == "Bono de Equipo",
            "Team should become Equipo")
    _assert(f("II: National Age", "II: Era Nacional", "Portuguese") == "II: Era Nacional",
            "non-Spanish target must be left untouched")

    print("✅ Glossary self-test passed.")


def self_test_merge() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Merge self-test failed: {message}")

    def _t(text: str, loc_id: Optional[str] = None) -> TranslationTarget:
        return TranslationTarget(
            element=ET.Element("String"),
            text=text,
            symbol=None,
            skip=False,
            reason=None,
            loc_id=loc_id,
        )

    # 1) Unique _locID, English unchanged -> reuse-safe seed.
    # 2) Duplicate _locID -> aligned within the group by content.
    # 3) No _locID -> content fallback.
    # 4) Changed English -> draft shown, never seeded.
    # 5) Placeholder lost in old translation -> not reused.
    new = [
        _t("Gang Saw", "1"),                       # unchanged
        _t("Attack now", "2"),                     # changed (old was "Attack")
        _t("%s is not ready.", "3"),               # unchanged but old trans lost %s
        _t("Apple", "5"),                          # duplicate id group
        _t("Banana", "5"),                         # duplicate id group
        _t("Cat", None),                           # no _locID -> content fallback
        _t("Brand New String", "999"),             # new id
    ]
    old_source = [
        _t("Gang Saw", "1"),
        _t("Attack", "2"),
        _t("%s is not ready.", "3"),
        _t("Apple", "5"),
        _t("Banana", "5"),
        _t("Cat", "9"),
    ]
    old_trans = [
        _t("Sierra de banda", "1"),
        _t("Atacar", "2"),
        _t("no está listo.", "3"),                 # lost the %s
        _t("Manzana", "5"),
        _t("Plátano", "5"),
        _t("Gato", "9"),
    ]

    report = merge_by_locid(new, old_source, old_trans)
    by_source = {e.new_source: e for e in report.entries}

    e = by_source["Gang Saw"]
    _assert(e.status == "unchanged" and e.seed == "Sierra de banda" and e.reason == "reuse-safe",
            f"Gang Saw should be reuse-safe, got {e}")

    e = by_source["Attack now"]
    _assert(e.status == "changed" and e.seed is None and e.draft == "Atacar"
            and e.reason == "changed-needs-review",
            f"changed string must keep draft but never seed, got {e}")

    e = by_source["%s is not ready."]
    _assert(e.status == "unchanged" and e.seed is None and e.reason == "placeholder-mismatch",
            f"placeholder loss must block reuse, got {e}")

    _assert(by_source["Apple"].seed == "Manzana", f"Apple should map to Manzana, got {by_source['Apple']}")
    _assert(by_source["Banana"].seed == "Plátano", f"Banana should map to Plátano, got {by_source['Banana']}")

    e = by_source["Cat"]
    _assert(e.seed == "Gato" and e.reason == "reuse-safe",
            f"no-_locID string should reuse via content fallback, got {e}")

    e = by_source["Brand New String"]
    _assert(e.status == "new" and e.seed is None and e.reason == "new-needs-translation",
            f"unknown id should be new, got {e}")

    seeds = seed_list_from_report(report)
    _assert(len(seeds) == len(new), "seed list must align 1:1 with new targets")
    _assert(seeds[1] == "" and seeds[6] == "", "changed/new entries must seed as empty string")
    _assert(report.counts["seeded"] == 4, f"expected 4 safe seeds, got {report.counts.get('seeded')}")

    # --- Cosmetic changes / kept-as-source / case-insensitive placeholders ---
    new2 = [
        _t("Good against <color=0.1,0.2,0.3>shock units</color>.", "100"),  # markup-only change
        _t("Manoeuvre Cavalry", "101"),                                     # case-only change
        _t("XP %s", "102"),                                                 # unchanged, trans uses %S
        _t("%1s%2s. %3s", "103"),                                           # pure format -> keep
        _t("eulay.rtf", "104"),                                             # English filename left as-is
        _t("Light cavalry that raids.", "105"),                             # real text change
        _t("Defends against <color=1,2,3>Cannons</color>.", "106"),         # cosmetic, but old trans still English
    ]
    old_source2 = [
        _t("Good against shock units.", "100"),
        _t("manoeuvre cavalry", "101"),
        _t("XP %s", "102"),
        _t("%1s%2s. %3s", "103"),
        _t("eulay.rtf", "104"),
        _t("Heavy infantry that defends.", "105"),
        _t("Defends against Cannons.", "106"),
    ]
    old_trans2 = [
        _t("Bueno contra unidades de choque.", "100"),
        _t("Caballería de maniobra", "101"),
        _t("XP %S", "102"),
        _t("%1s%2s. %3s", "103"),
        _t("eulay.rtf", "104"),
        _t("Infantería pesada que defiende.", "105"),
        _t("Defends against Cannons.", "106"),                              # never translated
    ]
    r2 = merge_by_locid(new2, old_source2, old_trans2)
    b2 = {e.new_source: e for e in r2.entries}

    e = b2["Good against <color=0.1,0.2,0.3>shock units</color>."]
    _assert(e.reason == "reuse-cosmetic" and e.seed == "Bueno contra unidades de choque.",
            f"markup-only change must reuse the old translation, got {e}")

    e = b2["Manoeuvre Cavalry"]
    _assert(e.reason == "reuse-cosmetic" and e.seed == "Caballería de maniobra",
            f"case-only change must reuse the old translation, got {e}")

    e = b2["XP %s"]
    _assert(e.reason == "reuse-safe" and e.seed == "XP %S",
            f"%S should be treated as compatible with %s, got {e}")

    e = b2["%1s%2s. %3s"]
    _assert(e.reason == "kept-as-source" and e.seed == "%1s%2s. %3s",
            f"pure format string should be kept as-is, got {e}")

    e = b2["eulay.rtf"]
    _assert(e.reason == "old-equals-source" and e.seed is None,
            f"English filename with letters should stay flagged, got {e}")

    e = b2["Light cavalry that raids."]
    _assert(e.reason == "changed-needs-review" and e.seed is None,
            f"a real text change must still be flagged, got {e}")
    _assert(e.english_old == "Heavy infantry that defends.",
            f"changed entry must expose the old English for the diff view, got {e.english_old!r}")

    e = b2["Defends against <color=1,2,3>Cannons</color>."]
    _assert(e.seed is None and e.reason == "old-equals-source",
            f"a cosmetic change whose old translation is still English must NOT be seeded, got {e}")

    print("✅ Merge self-test passed.")


def self_test_cache_key_parity() -> None:
    """The public cache-key helper must produce the exact key translate_strings stores."""
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Cache-key parity self-test failed: {message}")

    samples = [
        "Need %s right now",
        "Pop: %d  <icon=\"(32)(ui/ingame/resource_population)\">",
        "This game requires Microsoft Windows XP or later.",
        "Plain text with no tokens",
    ]
    for text in samples:
        terms, regex, exclude = _normalize_protection(None, None, None)
        engine_key, _tokens, _phrases = protect_for_cache(text, terms, regex, exclude)
        public_key = protected_cache_key(text)
        _assert(public_key == engine_key,
                f"protected_cache_key diverged from engine key for {text!r}")

    key = protected_cache_key("Need %s right now")
    _assert("%s" not in key and "__TOK" in key, "placeholders must be tokenized in the cache key")

    print("✅ Cache-key parity self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel XML localization tool powered by Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Positional I/O ---
    parser.add_argument("input", type=Path, help="Input XML file to translate.")
    parser.add_argument("output", type=Path, help="Output XML file path.")

    # --- API group ---
    api_group = parser.add_argument_group("API")
    api_group.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        help="Gemini API key. Falls back to GEMINI_API_KEY / GOOGLE_API_KEY env vars.",
    )
    api_group.add_argument(
        "--api-timeout",
        type=int,
        default=DEFAULT_API_TIMEOUT,
        help="Per-request API timeout in seconds.",
    )
    api_group.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of concurrent worker threads calling the API.",
    )
    api_group.add_argument(
        "--max-budget-bytes",
        type=int,
        default=MAX_BUDGET_BYTES,
        help="Soft byte-size budget per batch sent to the API.",
    )

    # --- Language group ---
    lang_group = parser.add_argument_group("Language")
    lang_group.add_argument("--source", default=DEFAULT_SOURCE_LANG, help="Source language name.")
    lang_group.add_argument("--target", default=DEFAULT_TARGET_LANG, help="Target language name.")

    # --- Prompt / quality group ---
    quality_group = parser.add_argument_group("Prompt & quality")
    quality_group.add_argument(
        "--compact-prompt",
        action="store_true",
        dest="compact_prompt",
        default=DEFAULT_COMPACT_PROMPT,
        help="Use the condensed prompt (default, token-efficient).",
    )
    quality_group.add_argument(
        "--detailed-prompt",
        action="store_false",
        dest="compact_prompt",
        help="Use the detailed prompt for maximum explicitness at higher token cost.",
    )
    quality_group.add_argument(
        "--strict-no-english-residue",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="strict_no_english_residue",
        help=(
            "Enable/disable strict English residue detection for Spanish targets. "
            "Defaults to on when target language is Spanish."
        ),
    )

    # --- Skip rules group ---
    skip_group = parser.add_argument_group("Skip rules (do not translate)")
    skip_group.add_argument(
        "--skip-symbol",
        action="append",
        default=[],
        help="Exact symbol names to skip (repeatable).",
    )
    skip_group.add_argument(
        "--skip-symbol-contains",
        action="append",
        default=[],
        help="Substring match (case-insensitive) for symbol names to skip (repeatable).",
    )
    skip_group.add_argument(
        "--skip-symbol-regex",
        action="append",
        default=[],
        help="Regular expression for symbol names to skip (repeatable).",
    )
    skip_group.add_argument(
        "--skip-text-regex",
        action="append",
        default=[],
        help="Regular expression for element text to skip (repeatable).",
    )
    skip_group.add_argument(
        "--no-path-heuristic",
        action="store_true",
        help="Disable automatic path-like text detection.",
    )

    # --- Protection group ---
    protect_group = parser.add_argument_group("Term protection")
    protect_group.add_argument(
        "--protect",
        action="append",
        default=[],
        help='Exact phrases to protect from translation (repeatable). Example: --protect "My Games"',
    )
    protect_group.add_argument(
        "--protect-regex",
        action="append",
        default=[],
        help="Regular expressions for phrases to protect from translation (repeatable).",
    )
    protect_group.add_argument(
        "--acronym-exclude",
        action="append",
        default=[],
        help="ALL-CAPS tokens that should be allowed to translate (repeatable). Example: --acronym-exclude ONE",
    )

    # --- Cache group ---
    cache_group = parser.add_argument_group("Cache")
    cache_group.add_argument(
        "--cache-file",
        type=Path,
        help="Use a specific cache JSON file instead of <output>.cache.json.",
    )
    cache_group.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached translations; do not call the Gemini API.",
    )
    cache_group.add_argument(
        "--retry-empty-cache",
        action="store_true",
        help='Retry translations that were cached as empty ("").',
    )

    # --- Merge group (carry an old translation onto a new source version by _locID) ---
    merge_group = parser.add_argument_group("Merge (version update by _locID)")
    merge_group.add_argument(
        "--match-by-locid",
        action="store_true",
        help="Reuse an old translation by matching _locID instead of position. "
             "Requires --prev-source and --prev-translation.",
    )
    merge_group.add_argument(
        "--prev-source",
        type=Path,
        help="Old source XML (the English version the old translation was made from).",
    )
    merge_group.add_argument(
        "--prev-translation",
        type=Path,
        help="Old translated XML to carry over (e.g. the previous Spanish file).",
    )
    merge_group.add_argument(
        "--report",
        type=Path,
        help="Write a JSON merge report (counts + strings that still need attention).",
    )

    # --- Diagnostics group ---
    diag_group = parser.add_argument_group("Diagnostics")
    diag_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )
    diag_group.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print encoding/BOM diagnostics after each write.",
    )
    diag_group.add_argument(
        "--self-test-quality-gate",
        action="store_true",
        help="Run quick quality gate tests and exit.",
    )
    diag_group.add_argument(
        "--self-test-merge",
        action="store_true",
        help="Run the _locID merge / cache-key self-tests and exit.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    if args.self_test_quality_gate:
        self_test_quality_gate()
        self_test_source_casing()
        self_test_glossary()
        return
    if args.self_test_merge:
        self_test_merge()
        self_test_cache_key_parity()
        return
    skip_rules = build_skip_rules(args)
    protected_terms = list(DEFAULT_PROTECTED_TERMS)
    if args.protect:
        protected_terms.extend(args.protect)
    protected_regex = compile_regex_list(args.protect_regex)
    acronym_exclude = list(DEFAULT_ACRONYM_EXCLUDE)
    if args.acronym_exclude:
        acronym_exclude.extend([t.strip() for t in args.acronym_exclude if t and t.strip()])

    if not args.input.exists():
        raise SystemExit(f"File does not exist: {args.input}")

    if not args.api_key and not args.cache_only:
        raise SystemExit(
            "Missing API key. Pass --api-key or set the GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
        )

    tree, doc_format = parse_strings_xml(args.input)
    targets = list(iter_translatable_elements(tree.getroot(), skip_rules))
    elements = [target.element for target in targets]
    translatable_targets = [target for target in targets if not target.skip]
    translatable_texts = [target.text for target in translatable_targets]

    print(f"🔥 HIGH-POWER MODE: {DEFAULT_MODEL} + {args.max_workers} threads.")
    if args.compact_prompt:
        print("💾 Compact prompt enabled (token-efficient with all rules).")
    else:
        print("🧭 Detailed prompt enabled (more context, higher token cost).")

    skipped_count = len(targets) - len(translatable_targets)
    if skipped_count:
        print(f"🛑 Skip filter engaged: {skipped_count} element(s) protected from translation.")

    # --- Merge by _locID: carry an old translation onto this new source version ---
    merge_report: Optional[MergeReport] = None
    if args.match_by_locid:
        if not args.prev_source or not args.prev_translation:
            raise SystemExit("--match-by-locid requires both --prev-source and --prev-translation.")
        if not args.prev_source.exists():
            raise SystemExit(f"File does not exist: {args.prev_source}")
        if not args.prev_translation.exists():
            raise SystemExit(f"File does not exist: {args.prev_translation}")
        prev_src_tree, _ = parse_strings_xml(args.prev_source)
        prev_trans_tree, _ = parse_strings_xml(args.prev_translation)
        old_source_targets = [
            t for t in iter_translatable_elements(prev_src_tree.getroot(), skip_rules) if not t.skip
        ]
        old_trans_targets = [
            t for t in iter_translatable_elements(prev_trans_tree.getroot(), skip_rules) if not t.skip
        ]
        merge_report = merge_by_locid(translatable_targets, old_source_targets, old_trans_targets)
        c = merge_report.counts
        print(
            f"🔗 Merge by _locID: {c.get('unchanged', 0)} unchanged, "
            f"{c.get('changed', 0)} changed, {c.get('new', 0)} new "
            f"→ {c.get('seeded', 0)} reused safely (the rest will be translated)."
        )
        if args.report:
            write_merge_report(args.report, merge_report)
            print(f"📝 Merge report written: {args.report}")

    existing_translations_subset: Optional[List[str]] = None
    if merge_report is not None:
        # The merge takes precedence over resume-from-output: only reuse-safe entries
        # are seeded ("" elsewhere -> translated). The snapshot falls back to the new
        # source text where nothing was reused yet.
        existing_translations_subset = seed_list_from_report(merge_report)
        starting_subset = [
            seed if seed else target.text
            for seed, target in zip(existing_translations_subset, translatable_targets)
        ]
        starting_texts = assemble_full_texts(
            targets, starting_subset, enforce_skip_integrity=True
        )
    else:
        existing_translations_full = load_existing_translations(args.output, len(targets), skip_rules)
        if existing_translations_full:
            print("↩️  Resuming translation from existing output file.")
            existing_translations_subset = [
                text for target, text in zip(targets, existing_translations_full) if not target.skip
            ]
            for target, text in zip(targets, existing_translations_full):
                if target.skip and text != target.text:
                    logging.warning(
                        "Existing output differs for skipped element (symbol=%s, reason=%s); restoring input text.",
                        target.symbol,
                        target.reason,
                    )
            starting_texts = assemble_full_texts(
                targets, existing_translations_subset, enforce_skip_integrity=True
            )
        else:
            starting_texts = [target.text for target in targets]

    write_output_snapshot(tree, elements, starting_texts, args.output, doc_format, diagnostic=args.diagnostic)

    if not translatable_texts:
        print("🔒 No elements eligible for translation. Output snapshot written.")
        print("\n📊 Summary")
        print("Total strings: 0")
        print("Used from cache: 0")
        print("Translated with API: 0")
        print("Skipped due to empty cache (skipped (cached empty)): 0")
        print(f"\n✅ Completed: {args.output}")
        return

    cache_file = args.cache_file or args.output.with_suffix(args.output.suffix + ".cache.json")

    # Cache diagnostic: show cache state and how many strings need translation.
    _translatable_count = len(translatable_texts)
    if cache_file.exists():
        try:
            _preview = json.loads(cache_file.read_text(encoding="utf-8"))
            _total = len(_preview)
            _empty = sum(1 for v in _preview.values() if not (v or "").strip())
            _full = _total - _empty
            print(
                f"💾 Cache found at {cache_file} "
                f"({_total} entries: {_full} translated, {_empty} empty/failed)."
            )
            if _translatable_count > _full:
                _gap = _translatable_count - _full
                print(
                    f"   ⚠️  Approximately {_gap} string(s) are NOT in the cache. "
                    f"These will be (re)translated. Likely causes: interrupted previous run, "
                    f"batch failures, or duplicate source text (counted separately)."
                )
        except Exception as _exc:
            print(f"⚠️  Cache file exists but could not be parsed: {_exc}")
    else:
        print(f"💾 No cache file at {cache_file} (first run or cache deleted).")
    if args.strict_no_english_residue is None:
        strict_no_english_residue = STRICT_NO_ENGLISH_RESIDUE and target_is_spanish(args.target)
    else:
        strict_no_english_residue = args.strict_no_english_residue

    def progress_callback(current_subset: Sequence[str]) -> None:
        merged = assemble_full_texts(targets, current_subset, enforce_skip_integrity=True)
        write_output_snapshot(
            tree, elements, merged, args.output, doc_format, diagnostic=args.diagnostic
        )

    try:
        translated_subset, stats = translate_strings(
            translatable_texts,
            api_key=args.api_key,
            source_lang=args.source,
            target_lang=args.target,
            cache_path=cache_file,
            existing_translations=existing_translations_subset,
            max_workers=args.max_workers,
            max_budget_bytes=args.max_budget_bytes,
            compact_prompt=args.compact_prompt,
            prompt_config=DEFAULT_PROMPT_CONFIG,
            progress_callback=progress_callback,
            protected_terms=protected_terms,
            protected_regex=protected_regex,
            acronym_exclude=acronym_exclude,
            strict_no_english_residue=strict_no_english_residue,
            cache_only=args.cache_only,
            retry_empty_cache=args.retry_empty_cache,
            api_timeout_seconds=args.api_timeout,
        )
        final_texts = assemble_full_texts(
            targets, translated_subset, enforce_skip_integrity=True
        )
        write_output_snapshot(tree, elements, final_texts, args.output, doc_format, diagnostic=args.diagnostic)
        # Count any strings that remained untranslated after the run.
        # (assemble_full_texts preserves originals for untranslated entries.)
        _untranslated = sum(
            1 for target, translated in zip(translatable_targets, translated_subset)
            if translated == target.text and target.text.strip()
        )
        _processed = stats.cache_used + stats.api_translated
        _pending = max(0, stats.total_strings - _processed - stats.cache_empty_skipped)

        summary = [
            "\n📊 Summary",
            f"  Total strings    : {stats.total_strings}",
            f"  From cache       : {stats.cache_used}",
            f"  Translated (API) : {stats.api_translated}",
            f"  Skipped (empty)  : {stats.cache_empty_skipped}",
        ]
        if _pending > 0:
            summary.append(f"  ⚠️  Pending        : {_pending} (not yet translated)")
        print("\n".join(summary))

        # Loud warning if the job finished with untranslated material.
        if _pending > 0 or stats.cache_empty_skipped > 0:
            print(
                f"\n⚠️  Finished with unresolved strings: "
                f"{_pending} pending, {stats.cache_empty_skipped} skipped due to previous failures."
            )
            print("   Re-run the script to retry pending strings.")
            if stats.cache_empty_skipped > 0:
                print("   Use --retry-empty-cache to retry strings that previously failed.")
            print(f"\n✅ Output written: {args.output}")
        else:
            print(f"\n✅ Completed: {args.output}")

    except Exception as e:
        logging.debug("Unhandled exception", exc_info=True)
        raise SystemExit(f"\n❌ Error: {e}") from e

if __name__ == "__main__":
    main()
