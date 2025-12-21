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

PLACEHOLDER_RE = re.compile(r"(%\d+\$[sdif]|%[sdif]|\\n|\\t|\\r)")
DEFAULT_SKIP_SYMBOL_CONTAINS = ["folder", "path", "dir", "directory"]
DEFAULT_PROTECTED_TERMS = ["Age of Empires III: Wars of Liberty"]
DEFAULT_PROTECTED_REGEX = [
    r"\bXP\b",
    r"\bHP\b",
    r"\bMP\b",
    r"\bDPS\b",
    r"\bPvP\b",
    r"\bPvE\b",
    r"\bGG\b",
    r"\bMVP\b",
]


@dataclass(frozen=True)
class PromptConfig:
    """Holds prompt templates for translation requests."""

    compact_template: str
    detailed_template: str

    def build(self, batch: Sequence[str], source_lang: str, target_lang: str, compact: bool) -> str:
        template = self.compact_template if compact else self.detailed_template
        return template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            input_list=json.dumps(batch, ensure_ascii=False),
        )


DEFAULT_PROMPT_CONFIG = PromptConfig(
    compact_template=(
        "You are a professional video game localization specialist. "
        "Translate the provided list from {source_lang} to {target_lang} "
        "for a historical video game set between 1789 and 1916 "
        "(Age of Empires III: Wars of Liberty). "
        "Use historically appropriate terminology from the late 18th to early 20th century, "
        "avoid modern slang, and keep the language clear and playable. "
        "Keep globally recognized gaming abbreviations (XP, HP, MP, DPS, PvP, PvE, GG, MVP) unchanged. "
        "DO NOT modernize or embellish the text. "
        "Keep all placeholders (__TOK#, %s, %1$s, %d, \n, \t) unchanged and in the same position. "
        "Treat any __PROTECT_x__ tokens as immutable placeholders. "
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
    - Leave globally recognized gaming abbreviations (XP, HP, MP, DPS, PvP, PvE, GG, MVP) unchanged.

    TECHNICAL RULES (STRICT)
    1. Do NOT translate, modify, reorder, or remove placeholders such as:
       __TOK#, %s, %1$s, %d, \n, \t, and __PROTECT_x__ tokens.
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


@dataclass(frozen=True)
class SkipRules:
    symbol_exact: Sequence[str]
    symbol_contains: Sequence[str]
    symbol_regex: Sequence[re.Pattern[str]]
    text_regex: Sequence[re.Pattern[str]]
    enable_path_heuristic: bool = True

def setup_gemini(api_key: str) -> genai.Client:
    """Create a Google GenAI client (google-genai SDK)."""
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
) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}
    protected = text
    idx = 0

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
            token = f"__PROTECT_{idx}__"
            token_map[token] = match.group(0)
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

    return restored


def restore_all_tokens(
    text: str,
    placeholder_map: Dict[str, str],
    protected_map: Dict[str, str],
    original_text: str,
) -> str:
    restored = unprotect_tokens(text, placeholder_map)
    restored = restore_protected_terms(restored, protected_map, original_text)
    return restored


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
    cleaned = cleaned.replace("\n", " ").replace("\t", " ").replace("\r", " ").strip()
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

    # Minimal path pattern like: "My Games\Wars of Liberty" (may contain spaces).
    if re.match(r"^[\w\s\-\.\(\)]+[\\/][\w\s\-\.\(\)]+(?:[\\/][\w\s\-\.\(\)]+)*[\\/]?$", cleaned):
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
    prompt_config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> List[str]:

    prompt = prompt_config.build(batch, source_lang, target_lang, compact_prompt)

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
    )
    value_error_retryables = (
        "response without candidates",
        "empty response",
        "invalid json",
        "length mismatch",
        "finish_reason=safety",
        "finish_reason=blocked",
    )

    message = str(exc).lower()

    if any(signal in message for signal in transient_signals):
        return True

    finish_reason_hints = ("finish_reason=safety", "finish_reason=blocked", "safety", "blocked")

    if any(hint in message for hint in finish_reason_hints):
        return True

    if isinstance(exc, ValueError) and any(signal in message for signal in value_error_retryables):
        return True

    return not isinstance(exc, ValueError)

def translate_batch_with_retry(
    client,
    batch,
    source,
    target,
    max_retries,
    compact_prompt: bool,
    prompt_config: PromptConfig,
) -> List[str]:
    attempt = 0
    last_partial: Optional[List[str]] = None
    while True:
        try:
            return translate_batch_gemini(
                client,
                batch,
                source,
                target,
                compact_prompt,
                prompt_config=prompt_config,
            )
        except Exception as exc:
            attempt += 1
            partial = getattr(exc, "partial_translations", None)
            if partial:
                last_partial = partial
            retryable = is_retryable_error(exc)
            logging.warning(
                "Batch error (attempt %s/%s, retry=%s): %s",
                attempt,
                max_retries,
                retryable,
                exc,
            )
            if (not retryable) or attempt > max_retries:
                logging.error("Critical failure in worker. Skipping batch.")
                if last_partial:
                    return last_partial
                return list(batch)

            backoff = min(BACKOFF_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            backoff += random.uniform(0, BACKOFF_SECONDS)
            time.sleep(backoff)

def translate_strings(
    inners: Iterable[str],
    api_key: str,
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
) -> List[str]:
    
    client = setup_gemini(api_key)

    protected_terms = protected_terms or []
    protected_regex = protected_regex or []

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

    for idx, inner in enumerate(inners):
        phrase_protected, phrase_map = protect_phrases(inner, protected_terms, protected_regex)
        protected_text, token_map = protect_tokens(phrase_protected)
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
            for idx in indexes_by_protected.get(text, []):
                translations[idx] = restore_all_tokens(
                    cached_value, token_maps[idx], phrase_maps[idx], original_texts[idx]
                )
            continue

        # If there is no cache (or it is empty), register an entry and queue it for translation,
        # avoiding duplicates.
        if text not in cache:
            cache[text] = ""
        if text not in already_enqueued:
            already_enqueued.add(text)
            unique_to_translate.append(text)

    # Build all batches
    batches = list(yield_batches(unique_to_translate, max_budget_bytes))

    # Map to sort results: {batch_index: [original_texts]}
    batch_map = {i: batch for i, batch in enumerate(batches)}
    total_batches = len(batches)

    print(f"🚀 Starting MULTITHREAD engine: {max_workers} concurrent workers...")

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
            ): idx
            for idx, batch in batch_map.items()
        }

        # Process tasks as they complete
        for future in tqdm(
            as_completed(future_to_batch_idx),
            total=total_batches,
            desc="Translating in Parallel",
            unit="batch",
        ):
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
                translated_batch = original_batch  # Fallback

            # Store in cache and update main list
            for original, translated_item in zip(original_batch, translated_batch):
                cache[original] = translated_item
                for idx in indexes_by_protected.get(original, []):
                    translations[idx] = restore_all_tokens(
                        translated_item,
                        token_maps[idx],
                        phrase_maps[idx],
                        original_texts[idx],
                    )

            # Save partial progress (thread-safe because we are on the main thread)
            if cache_path:
                try:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    logging.warning("Could not persist cache for batch %s: %s", batch_idx, exc)

            if progress_callback:
                progress_callback(list(translations))

    return translations

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-budget-bytes", type=int, default=MAX_BUDGET_BYTES)
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        dest="compact_prompt",
        default=DEFAULT_COMPACT_PROMPT,
        help="Use a condensed prompt (default) to save tokens.",
    )
    parser.add_argument(
        "--detailed-prompt",
        action="store_false",
        dest="compact_prompt",
        help="Use the detailed prompt for maximum explicitness at higher token cost.",
    )
    parser.add_argument(
        "--skip-symbol",
        action="append",
        default=[],
        help="Exact symbol names to skip translation (repeatable).",
    )
    parser.add_argument(
        "--skip-symbol-contains",
        action="append",
        default=[],
        help="Substring match (case-insensitive) for symbol names to skip (repeatable).",
    )
    parser.add_argument(
        "--skip-symbol-regex",
        action="append",
        default=[],
        help="Regular expression for symbol names to skip (repeatable).",
    )
    parser.add_argument(
        "--skip-text-regex",
        action="append",
        default=[],
        help="Regular expression for element text to skip (repeatable).",
    )
    parser.add_argument(
        "--no-path-heuristic",
        action="store_true",
        help="Disable automatic path-like text detection.",
    )
    parser.add_argument(
        "--protect",
        action="append",
        default=[],
        help='Exact phrases to protect from translation (repeatable). Example: --protect "Age of Empires III: Wars of Liberty"',
    )
    parser.add_argument(
        "--protect-regex",
        action="append",
        default=[],
        help="Regular expressions for phrases to protect from translation (repeatable).",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print encoding/BOM diagnostics after each write.",
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    skip_rules = build_skip_rules(args)
    protected_terms = list(DEFAULT_PROTECTED_TERMS)
    if args.protect:
        protected_terms.extend(args.protect)
    protected_regex = compile_regex_list(DEFAULT_PROTECTED_REGEX + (args.protect_regex or []))

    if not args.input.exists():
        raise SystemExit(f"File does not exist: {args.input}")

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

    existing_translations_full = load_existing_translations(args.output, len(targets), skip_rules)
    existing_translations_subset: Optional[List[str]] = None
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
        print(f"\n✅ Completed: {args.output}")
        return

    cache_file = args.output.with_suffix(args.output.suffix + ".cache.json")

    def progress_callback(current_subset: Sequence[str]) -> None:
        merged = assemble_full_texts(targets, current_subset, enforce_skip_integrity=True)
        write_output_snapshot(
            tree, elements, merged, args.output, doc_format, diagnostic=args.diagnostic
        )

    try:
        translated_subset = translate_strings(
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
        )
        final_texts = assemble_full_texts(
            targets, translated_subset, enforce_skip_integrity=True
        )
        write_output_snapshot(tree, elements, final_texts, args.output, doc_format, diagnostic=args.diagnostic)
        print(f"\n✅ Completed: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
