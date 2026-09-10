"""Parallel translation script powered by Gemini 2.5 Flash."""

from __future__ import annotations

import argparse
import collections
import io
import logging
import os
import re
import sys
import hashlib
import time
import json
import random
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from tqdm import tqdm

def _force_utf8_console() -> None:
    """Make stdout/stderr able to print the report's emoji and accents on Windows.

    A default Windows console is cp1252, and the audit died with
    ``UnicodeEncodeError: 'charmap' codec can't encode character '📋'`` before printing
    a single line. ``errors="replace"`` keeps a legacy console usable instead of crashing, and
    the whole thing is guarded because under ``pythonw.exe`` there is no stdout at all.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_console()

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
# 2.5 Flash defaults to temperature 1.0. The prompt asks for identical translations of identical
# strings, but batches run in parallel and independently, so only low-variance sampling can
# actually deliver that. This is why "deck" came out as both "mazo" and "baraja" on one screen.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_SEED = 12345
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
# Order matters: protect_phrases consumes this list in order, so the longest title must come
# first or the shorter one would mask half of it. The bare "Age of Empires III" MUST be here:
# without it, its "of" trips ENGLISH_RESIDUE_STOPWORDS and the engine discards a perfectly good
# Spanish translation and ships the English source instead (144 strings did exactly that).
# "Age of Empire III" (sic) is the game's own typo and appears in real strings.
DEFAULT_PROTECTED_TERMS = [
    "Age of Empires III: Wars of Liberty",
    "Age of Empires III",
    "Age of Empire III",
    "Wars of Liberty",
    "My Games",
]
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

# Only words that are unambiguously English and never appear in a correct Spanish string.
# Deliberately NOT here (each caused false positives that shipped English to players):
#   "original"/"version" - both are real Spanish words;
#   "new"/"world"/"trade"/"center" - too common inside untranslatable proper nouns
#     ("New World Trade Center", "Fort Ross"). The exact phrase is still caught by
#     ENGLISH_RESIDUE_PHRASES below, which is precise where a bare word is not.
ENGLISH_RESIDUE_STOPWORDS = {
    "the",
    "of",
    "to",
    "through",
    "enter",
    "address",
    "host",
    "connect",
}
ENGLISH_RESIDUE_PHRASES = {
    "of the",
    "new world trade center",
}
STRICT_MARKUP_RULES = (
    "STRICT MARKUP RULE\n"
    "Every <color=...>...</color> pair MUST still wrap the translated word it wrapped in the "
    "source. Move the tags together with the word when the word order changes. Never leave a "
    "pair empty and never drop a pair: the coloured word tells the player what the unit is "
    "strong against, so losing the colour loses meaning, not just styling.\n"
    "For 'Nepalese <color=0.07, 0.68, 0.17>skirmisher</color> that is accurate' return "
    "'<color=0.07, 0.68, 0.17>infanteria a distancia</color> nepali certera', "
    "NOT 'infanteria a distancia nepali <color=0.07, 0.68, 0.17> </color> certera'."
)

STRICT_COUNTER_KEYWORD_RULES = (
    "STRICT UNIT-CLASS RULE\n"
    "A coloured unit-class keyword came back naming the WRONG class, or empty. The word inside "
    "each <color=...> pair must be exactly the term listed for that colour: it is the class name "
    "the game prints in its own effect tooltips, so swapping it for another class tells the "
    "player the wrong counter, and an empty pair tells them nothing. Re-translate keeping every "
    "keyword exact and inside its own tags.\n"
)

STRICT_QUALITY_RULES = (
    "STRICT QUALITY RULE\n"
    "Do not leave ANY English articles/prepositions (the/of/to/through/enter/address/host/connect/original/version) "
    "in the output. Translate them to Spanish.\n"
    "Keep names/acronyms and protected tokens unchanged."
)


def target_is_spanish(target_lang: str) -> bool:
    """True for any way a user might name a Spanish target.

    This one predicate gates the ENTIRE Spanish path (glossary, residue gate, LatAm gate), so a
    name it fails to recognize silently turns all of it off. Locale codes and 'castellano' are
    included for exactly that reason.
    """
    tl = (target_lang or "").lower()
    # Every marker is unambiguous on its own. A bare "latino"/"es-la" is deliberately NOT here:
    # it would also fire on a target like "Português latino" and apply Spanish rules to it.
    return any(marker in tl for marker in
               ("spanish", "español", "espanol", "castellano", "es-419", "es_419"))

# Everything inside a markup tag: <color=...>, <font=Arial 96>, <icon="(58)(War of the Triple
# Alliance\Icons\...)">. A translator never touches these, but they are full of English words and
# file paths, and the residue gate was reading them as untranslated prose -- "War of the Triple
# Alliance" alone trips the "of the" rule. Since that gate fails INTO the source language, 19
# correctly translated strings were shipping in English because of an icon path.
_MARKUP_TAG_RE = re.compile(r"<[^>]*>")


def _strip_quality_tokens(text: str) -> str:
    cleaned = PROTECT_TOKEN_RE.sub(" ", text)
    cleaned = QUALITY_TOKEN_RE.sub(" ", cleaned)
    cleaned = PLACEHOLDER_RE.sub(" ", cleaned)
    # Only the tags themselves; the text they wrap sits between them and is still checked.
    cleaned = _MARKUP_TAG_RE.sub(" ", cleaned)
    return cleaned


COLOR_OPEN_RE = re.compile(r"<color=[^>]*>", re.IGNORECASE)
COLOR_CLOSE_RE = re.compile(r"</color>", re.IGNORECASE)
# An opening tag followed by nothing but whitespace before its closing tag.
EMPTY_COLOR_RE = re.compile(r"<color=[^>]*>\s*</color>", re.IGNORECASE)
# Every <color=...>...</color> span, with its colour code and its inner text.
_COLOR_SPAN_RE = re.compile(r"<color=([^>]*)>(.*?)</color>", re.IGNORECASE | re.DOTALL)


def _normalize_color_code(code: str) -> str:
    """'0.07, 0.68, 0.17' and '0.07,0.68,0.17' are the same colour."""
    return re.sub(r"\s+", "", code or "").lower()


def _color_open_pattern(code: str) -> "re.Pattern[str]":
    """Compile an opening-tag matcher for one colour code, tolerant of spacing."""
    parts = [re.escape(part.strip()) for part in code.split(",")]
    return re.compile(r"<color=\s*" + r"\s*,\s*".join(parts) + r"\s*>", re.IGNORECASE)


# --- Wars of Liberty's counter-keyword system -------------------------------------------------
# WoL does NOT use vanilla AoE3 counter wording: it defines its own set of unit classes and codes
# each one with a colour. The coloured word is therefore VOCABULARY, not decoration -- it names
# the class the text is about, and the engine prints that same class in its effect tooltips
# ("Ranged Infantry: Changes Melee ATK Damage by 15.00%").
#
# Nothing used to check WHAT was inside the tag -- markup_integrity_ok only counts tags -- so the
# class vocabulary rotted silently: 15 strings called `shock units` "unidades de asalto" (the name
# of a DIFFERENT class), `standing units` shipped under four names, and the green class shipped as
# "hostigador", a word the game never displays anywhere else.
#
# The green colour is special: it wraps only the word "skirmisher" in all 475 of its occurrences,
# so its presence is an exact signal for "this is the CLASS, not the unit called Skirmisher".
# Its canon is "infanteria a distancia" -- what the engine itself prints for that class
# (_locID 430207 "Ranged Infantry") -- which is also the only rendering consistent with the other
# six classes, all of which are functional descriptions rather than unit names.
COUNTER_COLOR_SKIRMISHER = "0.07, 0.68, 0.17"

COUNTER_KEYWORDS: Dict[str, Dict[str, str]] = {
    COUNTER_COLOR_SKIRMISHER: {
        "skirmisher": "infantería a distancia",
        "skirmishers": "infantería a distancia",
        # In 6 strings the tag wraps the bare ADJECTIVE, as half of a compound class:
        # "<green>skirmish</green>-<orange>maneuver cavalry</orange>". Forcing the noun canon
        # there produces nonsense ("Caballería de infantería a distancia-maniobra"), so the
        # adjective gets its own canon. "hostigamiento" is an action, not a unit, so unlike
        # "hostigador" it collides with nothing the game displays.
        "skirmish": "hostigamiento",
    },
    "0.19, 0.52, 0.76": {
        "line unit": "unidad de línea",
        "line units": "unidades de línea",
    },
    "0.92, 0.84, 0.09": {
        "shock unit": "unidad de choque",
        "shock units": "unidades de choque",
        "shock cavalry": "caballería de choque",
    },
    "0.74, 0.11, 0.58": {
        "standing unit": "unidad estática",
        "standing units": "unidades estáticas",
    },
    "0.78, 0.35, 0.13": {
        "maneuver unit": "unidad de maniobra",
        "maneuver units": "unidades de maniobra",
        "maneuver cavalry": "caballería de maniobra",
        "maneuver infantry": "infantería de maniobra",
    },
    "0.74, 0.25, 0.11": {
        "assault unit": "unidad de asalto",
        "assault units": "unidades de asalto",
        "assault infantry": "infantería de asalto",
    },
    "0.82, 0.80, 0.60": {
        "scout unit": "unidad de exploración",
        "scout units": "unidades de exploración",
        "scout cavalry": "caballería de exploración",
        "scout infantry": "infantería de exploración",
    },
}

# Deterministic repairs, applied ONLY inside a span of the matching colour and ONLY where the
# swap keeps gender and number ("unidades estacionarias" -> "unidades estaticas"). Being
# colour-scoped is load-bearing: "unidades de asalto" is the CORRECT canon under the assault
# colour and a wrong-class defect under the shock colour, so this can never be a global regex.
#
# The green class is deliberately absent: "hostigador" (m.) -> "infanteria a distancia" (f.)
# changes gender and head noun, so a regex would leave "los infanteria a distancia". Same rule as
# crate/outpost/deck below -- prompt-only, and existing strings are re-translated instead.
COUNTER_KEYWORD_REPAIRS: Dict[str, Tuple[Tuple["re.Pattern[str]", str], ...]] = {
    "0.92, 0.84, 0.09": (
        (re.compile(r"unidades\s+de\s+asalto", re.IGNORECASE), "unidades de choque"),
        (re.compile(r"unidad\s+de\s+asalto", re.IGNORECASE), "unidad de choque"),
    ),
    "0.74, 0.11, 0.58": (
        (re.compile(r"unidades\s+(?:estacionarias|fijas|estacionadas)", re.IGNORECASE),
         "unidades estáticas"),
        (re.compile(r"unidad\s+(?:estacionaria|fija|estacionada)", re.IGNORECASE),
         "unidad estática"),
    ),
    "0.82, 0.80, 0.60": (
        (re.compile(r"caballer[íi]a\s+exploradora", re.IGNORECASE), "caballería de exploración"),
        (re.compile(r"unidades\s+exploradoras", re.IGNORECASE), "unidades de exploración"),
        (re.compile(r"unidad\s+exploradora", re.IGNORECASE), "unidad de exploración"),
    ),
}

_COUNTER_KEYWORDS_BY_CODE = {
    _normalize_color_code(code): lexicon for code, lexicon in COUNTER_KEYWORDS.items()
}
_COUNTER_REPAIRS_BY_CODE = {
    _normalize_color_code(code): fixes for code, fixes in COUNTER_KEYWORD_REPAIRS.items()
}
# The opening tag of the green class, used to tell the CLASS apart from the unit named
# "Skirmisher" (see the two SPANISH_GLOSSARY entries below).
GREEN_SKIRMISHER_RE = _color_open_pattern(COUNTER_COLOR_SKIRMISHER)


def _counter_spans(text: str) -> List[Tuple[str, str]]:
    """[(normalized colour code, inner text)] for every counter-keyword span in `text`."""
    spans = []
    for match in _COLOR_SPAN_RE.finditer(text or ""):
        code = _normalize_color_code(match.group(1))
        if code in _COUNTER_KEYWORDS_BY_CODE:
            spans.append((code, match.group(2)))
    return spans


def _fold(text: str) -> str:
    """Lowercase and strip accents, so 'Infanteria' matches 'infantería'."""
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def counter_keyword_ok(src: str, out: str, target_lang: str) -> bool:
    """True when every counter-keyword span still names the RIGHT unit class.

    The twin of ``markup_integrity_ok``: that one checks the tags survived, this one checks the
    word inside them is still the class the source named. A span that ends up empty, holding the
    name of another class, or holding a synonym the game never displays, costs the player the
    "strong against" cue even though the markup is intact.

    Accepts any canon of the same colour, so an English singular rendered as a Spanish plural is
    not a defect -- only naming the wrong class, or no class at all, is.

    Spanish-gated, unlike markup_integrity_ok: the canon table is Spanish. Other targets pass.
    """
    if not target_is_spanish(target_lang):
        return True
    out_spans = _counter_spans(out)
    for code, inner in _counter_spans(src):
        lexicon = _COUNTER_KEYWORDS_BY_CODE.get(code, {})
        if _fold(inner).strip() not in {_fold(k) for k in lexicon}:
            continue  # not a keyword we hold a canon for; nothing to assert
        accepted = [_fold(v) for v in lexicon.values()]
        same_colour = [_fold(text) for c, text in out_spans if c == code]
        if not any(any(canon in text for canon in accepted) for text in same_colour):
            return False
    return True


def counter_keyword_rules(batch: Sequence[str], target_lang: str) -> str:
    """Prompt rules for the counter-keyword colours that actually occur in `batch`.

    Same batch-filtering as ``user_glossary_rules``: strings without coloured keywords pay zero
    extra tokens.
    """
    if not target_is_spanish(target_lang):
        return ""
    joined = "\n".join(batch)
    lines = []
    for code, lexicon in COUNTER_KEYWORDS.items():
        if not _color_open_pattern(code).search(joined):
            continue
        pairs = "; ".join(f"'{en}' -> '{es}'" for en, es in lexicon.items())
        lines.append(f"- Inside <color={code}>...</color>: {pairs}")
    if not lines:
        return ""
    return (
        "UNIT-CLASS KEYWORDS (Wars of Liberty counter system -- these are fixed terms)\n"
        "The coloured word names a unit CLASS the game also prints in its effect tooltips, so it "
        "must use the exact term below and stay inside its own <color> tags. Never swap in the "
        "name of another class and never leave the tag empty.\n" + "\n".join(lines)
    )


def apply_counter_keyword_fixes(original_text: str, translated_text: str, target_lang: str) -> str:
    """Deterministic, colour-scoped repair of same-gender counter-keyword variants.

    Only rewrites text INSIDE a coloured span, and only for the swaps listed in
    ``COUNTER_KEYWORD_REPAIRS``, which all keep gender and number. Leading and trailing
    whitespace inside the tag is preserved: many spans are written
    ``<color=...>skirmisher </color>`` and that space is part of the sentence.
    """
    if not target_is_spanish(target_lang) or not translated_text:
        return translated_text
    if not _counter_spans(original_text):
        return translated_text

    def repair(match: "re.Match[str]") -> str:
        code = _normalize_color_code(match.group(1))
        fixes = _COUNTER_REPAIRS_BY_CODE.get(code)
        if not fixes:
            return match.group(0)
        inner = match.group(2)
        core = inner.strip()
        if not core:
            return match.group(0)
        for pattern, replacement in fixes:
            if pattern.fullmatch(core):
                lead = inner[: len(inner) - len(inner.lstrip())]
                trail = inner[len(inner.rstrip()):]
                return f"<color={match.group(1)}>{lead}{replacement}{trail}</color>"
        return match.group(0)

    return _COLOR_SPAN_RE.sub(repair, translated_text)




def markup_integrity_ok(src: str, out: str) -> bool:
    """True when the candidate preserves the source's ``<color>`` markup.

    In this game the coloured word is the counter keyword -- it tells the player what a unit is
    strong against -- so losing the colour loses information, not just styling. Spanish reorders
    adjective and noun ("Nepalese skirmisher" -> "infanteria a distancia nepali"), and the model
    routinely moves the word out of its tag, leaving `<color=...> </color>` empty or dropping the
    tag pair altogether.

    Language-agnostic on purpose: markup is identical in every target language, unlike
    has_english_residue which only makes sense for a Spanish target.

    Only DETECTS. A regex cannot repair this -- once the sentence is reordered we cannot know
    where the word went -- so the caller retries with a stricter prompt instead.
    """
    src = src or ""
    out = out or ""
    if len(COLOR_OPEN_RE.findall(src)) != len(COLOR_OPEN_RE.findall(out)):
        return False
    if len(COLOR_CLOSE_RE.findall(src)) != len(COLOR_CLOSE_RE.findall(out)):
        return False
    # An empty pair is only a defect if the source did not already have one.
    if EMPTY_COLOR_RE.search(out) and not EMPTY_COLOR_RE.search(src):
        return False
    return True


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
    # When this matches the ENGLISH original the whole entry is skipped. Needed where a term is
    # only wrong in isolation: "Crates of 500 food and Chests of 500 coin" correctly yields both
    # "Cajas" and "Cofres", so the crate rule must not rewrite Cofre there.
    source_block: Optional["re.Pattern[str]"] = None


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
            # Generic epoch sense ("By Age" was coming out as "Por Era", visible in game).
            # Every pattern below REQUIRES a determiner, preposition or a following ordinal
            # adjective, and the Spanish verb 'era' ("was") never takes one -- so the critical
            # negative case in self_test_glossary still holds.
            (re.compile(r"\bPor\s+Era\b"), "Por Edad"),
            (re.compile(r"\bpor\s+Era\b"), "por Edad"),
            (re.compile(r"\b(l|est|es|aquell|un|otr|primer|segund|nuev|mism)(a|as)\s+Era(s?)\b",
                        re.IGNORECASE),
             lambda m: m.group(1) + m.group(2) + (" Edades" if m.group(3) else " Edad")),
            (re.compile(r"\bEra\s+(anterior|siguiente|actual|posterior|previa)\b", re.IGNORECASE),
             lambda m: "Edad " + m.group(1).lower()),
            # "Avanzar a la Siguiente Era": an adjective sits between the article and the noun.
            (re.compile(r"\b(Siguiente|Pr[óo]xima|Anterior|Primera|[ÚU]ltima)\s+Era\b", re.IGNORECASE),
             lambda m: m.group(1) + " Edad"),
            (re.compile(r"\b(de|en|a|desde|hasta|entre)\s+Eras\b", re.IGNORECASE),
             lambda m: m.group(1) + " Edades"),
        ),
    ),
    GlossaryEntry(
        name="hitpoints",
        source_trigger=re.compile(r"\bhit\s?points?\b", re.IGNORECASE),
        prompt_hint="- Translate 'hitpoints'/'hit points' as 'Puntos de Vida' (never 'Puntos de Golpe').\n",
        # Safe to rewrite deterministically: the head noun 'Puntos' is untouched, so no article
        # or adjective agreement can break. One rule per casing keeps the original capitalization.
        output_fixes=(
            (re.compile(r"\bPuntos\s+de\s+(?:Golpe|Resistencia|Salud)\b"), "Puntos de Vida"),
            (re.compile(r"\bpuntos\s+de\s+(?:golpe|resistencia|salud)\b"), "puntos de vida"),
        ),
    ),
    GlossaryEntry(
        name="settler",
        source_trigger=re.compile(r"\bsettlers?\b", re.IGNORECASE),
        # 'Aldeano' belongs to Villager; using it for Settler makes two units share one name.
        source_block=re.compile(r"\bvillagers?\b", re.IGNORECASE),
        prompt_hint=("- Translate 'Settler' as 'Colono' and 'Villager' as 'Aldeano'. "
                     "They are different units; never use 'Aldeano' for 'Settler'.\n"),
        output_fixes=(
            (re.compile(r"\bAldeanos\b"), "Colonos"), (re.compile(r"\bAldeano\b"), "Colono"),
            (re.compile(r"\baldeanos\b"), "colonos"), (re.compile(r"\baldeano\b"), "colono"),
        ),
    ),
    GlossaryEntry(
        name="shipment",
        source_trigger=re.compile(r"\bshipments?\b", re.IGNORECASE),
        prompt_hint="- Translate 'Shipment' as 'Envío'.\n",
        output_fixes=(
            (re.compile(r"\bCargamentos\b"), "Envíos"), (re.compile(r"\bCargamento\b"), "Envío"),
            (re.compile(r"\bcargamentos\b"), "envíos"), (re.compile(r"\bcargamento\b"), "envío"),
        ),
    ),
    GlossaryEntry(
        name="potato",
        source_trigger=re.compile(r"\bpotato(?:es)?\b", re.IGNORECASE),
        prompt_hint="- Use Latin American vocabulary: 'potato' is 'papa', never 'patata'.\n",
        output_fixes=(
            (re.compile(r"\bPatatas\b"), "Papas"), (re.compile(r"\bPatata\b"), "Papa"),
            (re.compile(r"\bpatatas\b"), "papas"), (re.compile(r"\bpatata\b"), "papa"),
        ),
    ),
    # --- Prompt-only entries -------------------------------------------------------------
    # These three change grammatical gender (Cofre/Puesto are masculine, Caja/Avanzada feminine;
    # Baraja is feminine, Mazo masculine), so a regex swap would leave a broken article and
    # adjective behind ("Crear una Mazo nueva", "los Avanzada"). Gemini conjugates correctly, so
    # the rule is preventive only; strings already wrong are found by --audit-spanish and re-run.
    GlossaryEntry(
        name="crate",
        source_trigger=re.compile(r"\bcrates?\b", re.IGNORECASE),
        # "Crates of 500 food and Chests of 500 coin" correctly yields both Cajas and Cofres.
        source_block=re.compile(r"\bchests?\b", re.IGNORECASE),
        prompt_hint="- Translate 'Crate' as 'Caja' (and 'Chest' as 'Cofre'; they are different).\n",
        output_fixes=(),
    ),
    GlossaryEntry(
        name="outpost",
        source_trigger=re.compile(r"\boutposts?\b", re.IGNORECASE),
        prompt_hint="- Translate 'Outpost' as 'Avanzada' (not 'Puesto de Avanzada').\n",
        output_fixes=(),
    ),
    GlossaryEntry(
        name="deck",
        source_trigger=re.compile(r"\bdecks?\b", re.IGNORECASE),
        # A ship's 'Steel Decks' really is 'Cubiertas de Acero'.
        source_block=re.compile(r"\bsteel\s+decks?\b", re.IGNORECASE),
        prompt_hint="- Translate the card-game 'Deck' as 'Mazo' (never 'Baraja'), consistently.\n",
        output_fixes=(),
    ),
    # --- Wars of Liberty unit/building names ------------------------------------------------
    # Each of these shipped under several different Spanish names, so a player could not tell
    # that the card, the unit and the upgrade were the same thing. The canon is the variant the
    # community settled on; only same-gender, same-number swaps get output_fixes.
    # "Skirmisher" is TWO different things in this file and they need two different Spanish
    # words. Inside the green counter colour it is WoL's unit CLASS -- the one the engine labels
    # "Ranged Infantry" in its effect tooltips -- and a card reading "hostigadores" while the
    # tooltip reads "Infanteria a Distancia" is what the community reported: the Peruvian Legion
    # is line infantry, yet it gets the bonus. Outside the tag it is the unit actually named
    # Skirmisher. The green tag wraps nothing but this word in all 475 of its occurrences, so it
    # separates the two cases exactly.
    # "Hand infantry/cavalry" is the melee class paired with "ranged infantry/cavalry", and the
    # game labels it "Caballeria de cuerpo a cuerpo" itself (_locID 430105). "de mano" is a
    # calque that had already split the corpus in half (113 vs 111) before this rule existed.
    # Unlike the skirmisher class the swap keeps gender AND head noun -- only the modifier
    # changes -- so the deterministic fix is safe here, anchored to the whole phrase so that
    # "canon de mano" (hand cannon) and "armas de mano" are never touched.
    GlossaryEntry(
        name="hand-unit",
        source_trigger=re.compile(r"\bhand\s+(?:infantry|cavalry|units?)\b", re.IGNORECASE),
        prompt_hint=(
            "- 'Hand infantry' / 'hand cavalry' is the melee class: translate them 'infantería "
            "cuerpo a cuerpo' / 'caballería cuerpo a cuerpo', never 'de mano'.\n"
        ),
        output_fixes=(
            # IGNORECASE, and the backreference keeps whatever case the noun had: the phrase
            # appears both as a title ("Caballería de Mano") and mid-sentence ("la infantería
            # de mano"), and a case-sensitive pattern only repaired the first form.
            (re.compile(r"\b(Infanter[íi]as?|Caballer[íi]as?)\s+de\s+[Mm]ano\b", re.IGNORECASE),
             r"\1 cuerpo a cuerpo"),
        ),
    ),
    GlossaryEntry(
        name="skirmisher-class",
        source_trigger=GREEN_SKIRMISHER_RE,
        prompt_hint=(
            "- The green <color=0.07, 0.68, 0.17>skirmisher(s)</color> is a unit CLASS, the one "
            "the game labels 'Ranged Infantry'. Translate it 'infantería a distancia' -- "
            "FEMININE SINGULAR -- and make the articles and adjectives agree: 'la infantería a "
            "distancia', 'contra la infantería a distancia', never 'los hostigadores' or 'los "
            "infantería'. Keep the <color> tags wrapping those same words.\n"
        ),
        # Gender and head noun both change (hostigador m. -> infantería f.), so a regex would
        # leave "los infantería a distancia". Prompt-only, exactly like crate/outpost/deck above;
        # strings already wrong are found by --audit-spanish and re-translated.
        output_fixes=(),
    ),
    GlossaryEntry(
        name="skirmisher-unit",
        source_trigger=re.compile(r"\bskirmishers?\b", re.IGNORECASE),
        # When the green tag is present this is the class, handled by the entry above.
        source_block=GREEN_SKIRMISHER_RE,
        prompt_hint=(
            "- Translate the unit name 'Skirmisher' as 'Hostigador'. Never 'Guerrillero' (that "
            "is Wars of Liberty's own 'Guerrilla' unit) and never 'Cazador' (that is the "
            "'Cacadore' unit).\n"
        ),
        output_fixes=(
            # "Escararuzador" (Escara+R+uzador) is a misspelling of "Escaramuzador"
            # (Escara+M+uzador) that shipped in 9 strings; [mr] catches both.
            (re.compile(r"\bEscara[mr]uzadores\b"), "Hostigadores"),
            (re.compile(r"\bEscara[mr]uzador\b"), "Hostigador"),
            (re.compile(r"\bescara[mr]uzadores\b"), "hostigadores"),
            (re.compile(r"\bescara[mr]uzador\b"), "hostigador"),
        ),
    ),
    GlossaryEntry(
        name="hajduk",
        source_trigger=re.compile(r"\bhajduks?\b", re.IGNORECASE),
        prompt_hint="- Keep the unit name 'Hajduk' unchanged (never 'Hayduk').\n",
        output_fixes=((re.compile(r"\bHayduk"), "Hajduk"), (re.compile(r"\bhayduk"), "hajduk")),
    ),
    GlossaryEntry(
        name="boneguard",
        source_trigger=re.compile(r"\bboneguards?\b", re.IGNORECASE),
        prompt_hint="- Translate 'Boneguard' (the Circle's elite corps) as 'Guardia Ósea'.\n",
        # Every variant is feminine, like the canon, so number/gender cannot break.
        output_fixes=(
            (re.compile(r"\bGuardia\s+de\s+Hueso\b", re.IGNORECASE), "Guardia Ósea"),
            (re.compile(r"\bGuardahuesos?\b", re.IGNORECASE), "Guardia Ósea"),
            (re.compile(r"\bGuarda[óo]sea\b", re.IGNORECASE), "Guardia Ósea"),
        ),
    ),
    GlossaryEntry(
        name="pasha",
        source_trigger=re.compile(r"\bpashas?\b", re.IGNORECASE),
        prompt_hint="- Keep the title 'Pasha' unchanged; it is part of a personal name.\n",
        output_fixes=(
            (re.compile(r"\bPash[áa]\b"), "Pasha"),
            (re.compile(r"\bBaj[áa]\b"), "Pasha"),
        ),
    ),
    GlossaryEntry(
        name="madrasah",
        source_trigger=re.compile(r"\bmadrasahs?\b", re.IGNORECASE),
        prompt_hint="- Translate the building 'Madrasah' as 'Madrasa'.\n",
        output_fixes=(
            (re.compile(r"\bMadrazas\b"), "Madrasas"), (re.compile(r"\bMadraza\b"), "Madrasa"),
            (re.compile(r"\bmadrazas\b"), "madrasas"), (re.compile(r"\bmadraza\b"), "madrasa"),
        ),
    ),
    GlossaryEntry(
        name="warlord",
        source_trigger=re.compile(r"\bwarlords?\b", re.IGNORECASE),
        prompt_hint="- Translate the hero unit 'Warlord' as 'Caudillo'.\n",
        output_fixes=(
            (re.compile(r"\bSeñores\s+de\s+la\s+[Gg]uerra\b"), "Caudillos"),
            (re.compile(r"\bSeñor\s+de\s+la\s+[Gg]uerra\b"), "Caudillo"),
            (re.compile(r"\bseñores\s+de\s+la\s+guerra\b"), "caudillos"),
            (re.compile(r"\bseñor\s+de\s+la\s+guerra\b"), "caudillo"),
        ),
    ),
    GlossaryEntry(
        name="righteous-fighter",
        source_trigger=re.compile(r"\brighteous\s+fighters?\b", re.IGNORECASE),
        # Pairs with 'Righteous Army' -> 'Ejercito Justo', which already ships that way.
        prompt_hint="- Translate 'Righteous Fighter' as 'Guerrero Justo'.\n",
        output_fixes=(
            (re.compile(r"\bCombatientes\s+Justos\b", re.IGNORECASE), "Guerreros Justos"),
            (re.compile(r"\bCombatiente\s+Justo\b", re.IGNORECASE), "Guerrero Justo"),
            (re.compile(r"\bLuchadores\s+Justicieros\b", re.IGNORECASE), "Guerreros Justos"),
            (re.compile(r"\bLuchador\s+Justiciero\b", re.IGNORECASE), "Guerrero Justo"),
            (re.compile(r"\bGuerreros\s+Justicieros\b", re.IGNORECASE), "Guerreros Justos"),
            (re.compile(r"\bGuerrero\s+Justiciero\b", re.IGNORECASE), "Guerrero Justo"),
        ),
    ),
    GlossaryEntry(
        name="lodge",
        source_trigger=re.compile(r"\blodges?\b", re.IGNORECASE),
        # A Masonic lodge really would be a 'Logia'; the game's Lodge is the Hunter's Lodge.
        source_block=re.compile(r"\bmasonic\s+lodges?\b", re.IGNORECASE),
        prompt_hint="- Translate the building 'Lodge' as 'Cabaña' (never 'Logia', which is Masonic).\n",
        output_fixes=(
            (re.compile(r"\bLogias\b"), "Cabañas"), (re.compile(r"\bLogia\b"), "Cabaña"),
            (re.compile(r"\blogias\b"), "cabañas"), (re.compile(r"\blogia\b"), "cabaña"),
        ),
    ),
    GlossaryEntry(
        name="square-formation",
        source_trigger=re.compile(r"\bsquares?\b", re.IGNORECASE),
        # A town square is a real 'plaza'; only the infantry formation is a 'cuadro'.
        source_block=re.compile(r"\b(?:town|city|market|village)\s+squares?\b", re.IGNORECASE),
        prompt_hint=("- 'Square' is the infantry square FORMATION: translate it as 'Cuadro', "
                     "never 'Plaza' or 'Cuadrado'. 'Spanish Square' is the 'Tercio Español'.\n"),
        # Only the masculine->masculine swap is safe; 'Plaza Española'(f) -> 'Tercio'(m) would
        # need the article and adjective rewritten, so the prompt handles that one.
        output_fixes=((re.compile(r"\bCuadrados\b"), "Cuadros"), (re.compile(r"\bCuadrado\b"), "Cuadro")),
    ),
    GlossaryEntry(
        name="conscript",
        source_trigger=re.compile(r"\bconscripts?\b", re.IGNORECASE),
        # 'Conscript' is ALSO a verb: "Conscript Sepoys" -> "Reclutar Sepoys" is correct.
        # Without this guard the rule would corrupt those action strings.
        source_block=re.compile(r"\bconscript\s+[A-Z]", re.UNICODE),
        prompt_hint=("- Translate the unit 'Conscript' as 'Conscripto' (not 'Recluta', which is "
                     "the separate unit 'Recruit'). As a verb, 'to conscript' is 'reclutar'.\n"),
        # 'Recluta' is deliberately NOT rewritten: it is another unit's name.
        output_fixes=(
            (re.compile(r"\bConscritos\b"), "Conscriptos"), (re.compile(r"\bConscrito\b"), "Conscripto"),
            (re.compile(r"\bconscritos\b"), "conscriptos"), (re.compile(r"\bconscrito\b"), "conscripto"),
        ),
    ),
    # --- Prompt-only: gender changes, so a regex swap would break the article/adjective ------
    GlossaryEntry(
        name="allotment",
        source_trigger=re.compile(r"\ballotments?\b", re.IGNORECASE),
        # 'Land Reallotment' is a different card and is correctly 'Reasignación de Tierras'.
        source_block=re.compile(r"\breallotments?\b", re.IGNORECASE),
        prompt_hint=("- An 'Allotment' is a BLOCK OF TROOPS mustered at once (Swedish allotment "
                     "system), not a plot of land: translate it as 'Contingente', never "
                     "'Parcela', 'Reparto' or 'Asignación'.\n"),
        output_fixes=(),
    ),
    GlossaryEntry(
        name="revolt",
        source_trigger=re.compile(r"\brevolt(?:ing|s|ed)?\b", re.IGNORECASE),
        prompt_hint=("- 'To revolt' (the Revolution mechanic) is 'sublevarse', never 'revolverse' "
                     "(which means to stir). The noun 'Revolt' in a named uprising stays "
                     "'Revuelta' ('Arab Revolt' -> 'Revuelta Árabe').\n"),
        # Only the verb forms that are outright wrong. Deliberately NOT touched:
        #   'Revuelta'  -- correct for the named uprisings (13 strings);
        #   'Revólver'  -- the Colt Revolver weapon (7 strings), saved by the accent;
        #   'repugnante'-- 'revolting' also means disgusting, and one line is a pun on both.
        output_fixes=(
            (re.compile(r"\bRevolverse\b"), "Sublevarse"),
            (re.compile(r"\brevolverse\b"), "sublevarse"),
            (re.compile(r"\bRevuélvanse\b"), "Sublévense"),
            (re.compile(r"\brevuélvanse\b"), "sublévense"),
            (re.compile(r"\bRevolucionarse\b"), "Sublevarse"),
            (re.compile(r"\brevolucionarse\b"), "sublevarse"),
        ),
    ),
    GlossaryEntry(
        name="zapotec",
        source_trigger=re.compile(r"\bzapotecs?\b", re.IGNORECASE),
        prompt_hint="- Translate 'Zapotec' as 'Zapoteca', consistently.\n",
        output_fixes=(),
    ),
]


# Region rules for a Spanish target. The target language name alone ("Latin American Spanish")
# was not enough: the model still produced peninsular forms and dropped opening punctuation.
LATAM_SPANISH_RULES = (
    "LATIN AMERICAN SPANISH (apply ONLY when the target language is Spanish)\n"
    "- The target is Latin American Spanish (es-419), NOT peninsular Spanish. NEVER use "
    "'vosotros', 'os' or 'vuestro' forms, nor the -ad/-ed/-id imperative; use 'ustedes' "
    "(e.g. 'Atacad' -> 'Ataquen', '¿Qué os parece?' -> '¿Qué les parece?').\n"
    "- Address the player with 'tú' (informal second person singular), consistently: "
    "'Presiona', 'Selecciona', 'Haz clic' -- not 'Presione', 'Seleccione', 'Haga clic'.\n"
    "- Use Latin American vocabulary: 'papa' (not 'patata'), 'computadora' (not 'ordenador'), "
    "'jugo' (not 'zumo').\n"
    "- Use complete Spanish punctuation: open every exclamation with '¡' and every question "
    "with '¿'.\n"
)

# Preventive twin of STRICT_MARKUP_RULES: cheap enough to send on every request, and it is
# language-agnostic, so unlike the Spanish blocks it is never gated on the target.
MARKUP_PROMPT_RULE = (
    "MARKUP\n"
    "- Keep every <color=...>...</color> pair wrapping the SAME word it wraps in the source. "
    "When the target language reorders the words, move the tags with the word: never leave a "
    "pair empty and never drop one. The coloured word names the unit type the text is about.\n"
)


def terminology_overrides_for_target(
    target_lang: str,
    batch: Optional[Sequence[str]] = None,
) -> str:
    """Extra instructions appended to the prompt, only when needed.

    Built from ``SPANISH_GLOSSARY`` so the prompt and the post-process fixes share one source
    of truth. Keep this language-conditional so the script remains global (multi-language).

    When ``batch`` is given, only the entries whose ``source_trigger`` actually appears in it are
    emitted (the same batch-filtering ``user_glossary_rules`` does). Sending every term on every
    request wastes tokens and nudges the model toward that vocabulary in unrelated strings.

    ``source_block`` is honoured here too, and PER STRING rather than per batch: two entries can
    cover the same word in different contexts (the green counter tag makes 'skirmisher' the unit
    CLASS, its absence makes it the unit NAME), and testing the joined batch would let a single
    blocked string silence the rule for every other string travelling with it.
    """
    if not target_is_spanish(target_lang):
        return ""

    if batch is None:
        entries = SPANISH_GLOSSARY
    else:
        def fires(entry: GlossaryEntry) -> bool:
            return any(
                entry.source_trigger.search(text)
                and not (entry.source_block is not None and entry.source_block.search(text))
                for text in batch
            )

        entries = [e for e in SPANISH_GLOSSARY if fires(e)]

    hints = "".join(entry.prompt_hint for entry in entries)
    if hints:
        hints = "TERMINOLOGY OVERRIDES (apply ONLY when target language is Spanish)\n" + hints
    return (LATAM_SPANISH_RULES + "\n" + hints) if hints else LATAM_SPANISH_RULES


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
        if entry.source_block is not None and entry.source_block.search(original_text):
            continue
        for pattern, replacement in entry.output_fixes:
            out = pattern.sub(replacement, out)

    return out


# UI imperatives only. Every pair below is a regular -e/-a verb with no clitic attached, so the
# swap can never break agreement. Deliberately NOT extended to general usted->tu conversion:
# in "El juego puede fallar", "puede" is third person, not usted, and rewriting it is wrong.
_LATAM_UI_IMPERATIVES: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"\bHaga\s+clic\b"), "Haz clic"),
    (re.compile(r"\bhaga\s+clic\b"), "haz clic"),
    (re.compile(r"\bPresione\b"), "Presiona"), (re.compile(r"\bpresione\b"), "presiona"),
    (re.compile(r"\bPulse\b"), "Presiona"),    (re.compile(r"\bpulse\b"), "presiona"),
    (re.compile(r"\bSeleccione\b"), "Selecciona"), (re.compile(r"\bseleccione\b"), "selecciona"),
    (re.compile(r"\bElija\b"), "Elige"),       (re.compile(r"\belija\b"), "elige"),
    (re.compile(r"\bEscriba\b"), "Escribe"),   (re.compile(r"\bescriba\b"), "escribe"),
    (re.compile(r"\bIngrese\b"), "Ingresa"),   (re.compile(r"\bingrese\b"), "ingresa"),
    (re.compile(r"\bIntroduzca\b"), "Introduce"), (re.compile(r"\bintroduzca\b"), "introduce"),
)

# Trailing decoration that may sit after the final '!' or '?': markup, escaped whitespace, spaces.
_TRAILING_DECORATION_RE = re.compile(r"(?:</?[^>]*>|\\[ntr]|\s)+$")
# Where the last sentence starts: after terminal punctuation, a line break or an opening tag.
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?:;]|\\n|>)\s*")


def _add_opening_punctuation(text: str) -> str:
    """Add the Spanish opening '¡'/'¿' when a sentence ends with '!'/'?' and lacks it.

    Only the LAST sentence is considered, and only when its opening mark is absent, so a string
    that is already correct is returned untouched.
    """
    if not text:
        return text
    body = _TRAILING_DECORATION_RE.sub("", text)
    if not body:
        return text
    closing = body[-1]
    if closing not in "!?":
        return text
    opening = "¡" if closing == "!" else "¿"
    if opening in body:
        return text

    # Find where the final sentence begins; bail out if that leaves nothing but punctuation.
    start = 0
    for match in _SENTENCE_START_RE.finditer(body[:-1]):
        start = match.end()
    # Require real words: a sentence that is only placeholders/tokens ("%s!") is left alone,
    # since we cannot know what the engine will substitute into it.
    sentence = _strip_quality_tokens(body[start:])
    if not any(ch.isalpha() for ch in sentence):
        return text
    return text[:start] + opening + text[start:]


def normalize_latam_spanish(original_text: str, translated_text: str, target_lang: str) -> str:
    """Latin-American Spanish fixes that are safe to apply deterministically.

    Scope is deliberately narrow. Peninsular 'vosotros' forms are NOT rewritten here: the real
    strings carry enclitics ("Atacadnos", "ponedlos") and irregulars ("sabed", "Despertad") that
    a lookup table cannot conjugate, and the obvious markers are landmines -- 'sed' is the noun
    *thirst*, 'os' appears as a Portuguese article in a deliberately Portuguese line, and 'id'
    occurs in "ID de Passport". Those are handled by the prompt and reported by --audit-spanish.
    """
    if not target_is_spanish(target_lang) or not translated_text:
        return translated_text

    out = translated_text
    for pattern, replacement in _LATAM_UI_IMPERATIVES:
        out = pattern.sub(replacement, out)
    return _add_opening_punctuation(out)


@dataclass(frozen=True)
class UserGlossaryEntry:
    """One glossary.txt line: a term pair plus the context it is allowed to fire in.

    A flat 'source = target' map cannot express a word that needs two different translations
    depending on context -- which is exactly what broke the counter keywords, where 'Skirmisher'
    is a unit CLASS inside the green colour tag and a unit NAME outside it. The conditions make
    the distinction expressible in the file instead of only in code.
    """

    source_term: str
    target_term: str
    require: Tuple[str, ...] = ()   # every one of these must appear in the English original
    exclude: Tuple[str, ...] = ()   # none of these may appear in the English original

    def applies_to(self, text: str) -> bool:
        if any(needle not in text for needle in self.require):
            return False
        return not any(needle in text for needle in self.exclude)

    def describe_condition(self) -> str:
        parts = [f"the text contains {needle!r}" for needle in self.require]
        parts += [f"the text does NOT contain {needle!r}" for needle in self.exclude]
        return " and ".join(parts)


class UserGlossary(Dict[str, str]):
    """A plain {source term: target term} mapping, with the per-line conditions on ``.entries``.

    Subclassing dict keeps every existing caller working unchanged (the GUI prints ``len()``,
    the audit iterates ``.items()``, tests compare against ``{}``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entries: List[UserGlossaryEntry] = []


def _glossary_entries(glossary: Optional[Dict[str, str]]) -> List[UserGlossaryEntry]:
    """Entries of a UserGlossary, or unconditional ones for a plain dict."""
    if not glossary:
        return []
    entries = getattr(glossary, "entries", None)
    if entries is not None:
        return entries
    return [UserGlossaryEntry(source, target) for source, target in glossary.items()]


_GLOSSARY_CONDITION_RE = re.compile(r"^(si|if|solo si|only if|no|not|unless)\s*:\s*(.+)$", re.I)


def load_user_glossary(path: Optional[Path]) -> Dict[str, str]:
    """Parse a user glossary file: one 'source term = target term' per line.

    Lines starting with '#' and blank lines are ignored; malformed lines are
    skipped with a warning. Returns {} when the file is missing/unreadable.
    Unlike SPANISH_GLOSSARY this is pair-agnostic: entries simply never fire
    when their source term does not appear in the strings being translated.

    A line may carry '|'-separated conditions before the '=', checked against the ENGLISH
    original, so one word can map to different terms in different contexts::

        Skirmisher | si:<color=0.07, 0.68, 0.17> = infanteria a distancia
        Skirmisher | no:<color=0.07, 0.68, 0.17> = Hostigador

    'si:'/'if:'/'solo si:'/'only if:' require the text; 'no:'/'not:'/'unless:' forbid it.
    """
    glossary = UserGlossary()
    if not path:
        return glossary
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return glossary
    except Exception as exc:
        logging.warning("Could not read glossary file %s: %s", path, exc)
        return glossary
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # A condition can itself contain '=' (a colour code is "<color=0.07, 0.68, 0.17>"), so a
        # conditional line splits at its LAST '='. Plain lines keep splitting at the first one,
        # so nothing that parsed before parses differently now.
        if "|" in line:
            left, sep, target_term = line.rpartition("=")
        else:
            left, sep, target_term = line.partition("=")
        target_term = target_term.strip()
        chunks = [chunk.strip() for chunk in left.split("|")]
        source_term = chunks[0]
        if not sep or not source_term or not target_term:
            logging.warning("Glossary line %s ignored (expected 'source = target'): %s", line_no, line)
            continue
        require: List[str] = []
        exclude: List[str] = []
        malformed = False
        for chunk in chunks[1:]:
            match = _GLOSSARY_CONDITION_RE.match(chunk)
            if not match:
                logging.warning(
                    "Glossary line %s ignored (condition must be 'si:TEXT' or 'no:TEXT'): %s",
                    line_no, line,
                )
                malformed = True
                break
            keyword, needle = match.group(1).lower(), match.group(2).strip()
            if not needle:
                malformed = True
                break
            (exclude if keyword in {"no", "not", "unless"} else require).append(needle)
        if malformed:
            continue
        glossary[source_term] = target_term
        glossary.entries.append(
            UserGlossaryEntry(source_term, target_term, tuple(require), tuple(exclude))
        )
    return glossary


def user_glossary_rules(batch: Sequence[str], glossary: Optional[Dict[str, str]]) -> str:
    """Prompt rules for the user-glossary terms that actually occur in `batch`.

    Filtering per batch keeps the prompt small: strings without glossary terms
    pay zero extra tokens.
    """
    if not glossary:
        return ""
    lines = []
    for entry in _glossary_entries(glossary):
        if not any(entry.source_term in text and entry.applies_to(text) for text in batch):
            continue
        rule = (
            f"- Translate '{entry.source_term}' as '{entry.target_term}' (official game term); "
            "use it consistently."
        )
        condition = entry.describe_condition()
        if condition:
            # A batch can carry both contexts of the same word, so the condition has to travel
            # with the rule -- otherwise two contradictory rules would arrive with no way to
            # tell which string each applies to.
            rule = (
                f"- When {condition}, translate '{entry.source_term}' as "
                f"'{entry.target_term}' (official game term)."
            )
        lines.append(rule)
    if not lines:
        return ""
    return "MANDATORY TERMINOLOGY (user glossary)\n" + "\n".join(lines)


def apply_user_glossary_fixes(
    original_text: str,
    translated_text: str,
    glossary: Optional[Dict[str, str]],
) -> str:
    """Deterministic layer of the user glossary: fix source terms left untranslated.

    Only handles the model leaving the SOURCE term verbatim in the output (whole-word
    for Latin terms, plain replace for CJK). Wrong-but-translated synonyms can't be
    fixed deterministically — that is what the prompt rules are for.
    """
    if not glossary:
        return translated_text
    out = translated_text
    for entry in _glossary_entries(glossary):
        source_term, target_term = entry.source_term, entry.target_term
        if source_term == target_term or source_term not in original_text or source_term not in out:
            continue
        if not entry.applies_to(original_text):
            continue
        if re.search(r"[A-Za-z]", source_term):
            out = re.sub(
                rf"(?<!\w){re.escape(source_term)}(?!\w)",
                lambda _m: target_term,
                out,
            )
        else:
            out = out.replace(source_term, target_term)
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
        # Only worth sending when the batch actually carries markup.
        if any("<color=" in text for text in batch):
            prompt = prompt + "\n\n" + MARKUP_PROMPT_RULE
        overrides = terminology_overrides_for_target(target_lang, batch)
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
    # Several guards deliberately fall back to the untranslated source rather than emit a
    # corrupted string. That is the right call, but it used to happen silently -- these counters
    # make it visible how many strings actually shipped in the source language, and why.
    quality_rejected: int = 0   # failed the English-residue gate after every retry
    markup_rejected: int = 0    # <color> markup still broken after every retry
    counter_keyword_rejected: int = 0  # a coloured keyword still named the wrong unit class
    terms_stale: int = 0        # re-translated because its terminology rules changed
    consistency_fixed: int = 0  # re-translated by the post-run consistency sweep
    batch_failed: int = 0       # the whole batch errored out; kept retryable in the cache


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
# WoL also uses its own numbered form WITHOUT the dollar (%1s, %2d) and widths (%2.2f).
# Those were invisible here, so the merge guard never fired on them.
_FORMAT_SPECIFIER_RE = re.compile(r"%\d+\$[sdif]|%\d*\.?\d*[sdif]", re.IGNORECASE)


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
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = DEFAULT_SEED,
) -> List[str]:

    prompt = prompt_config.build(
        batch,
        source_lang,
        target_lang,
        compact_prompt,
        extra_rules=extra_rules,
    )

    # Ask the API to return strict JSON whenever possible. thinking_budget=0
    # disables 2.5 Flash's default "dynamic thinking": those hidden reasoning
    # tokens are billed as output and add nothing to mechanical translation.
    config_kwargs = dict(
        response_mime_type="application/json",
        response_schema=list[str],
        temperature=temperature,
    )
    # `seed` is not in every google-genai release, so it is added separately and dropped if the
    # installed SDK rejects it -- losing the seed only costs reproducibility, not correctness.
    seeded_kwargs = dict(config_kwargs)
    if seed is not None:
        seeded_kwargs["seed"] = seed

    config = None
    for kwargs in (seeded_kwargs, config_kwargs):
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                **kwargs,
            )
            break
        except (AttributeError, TypeError):  # older SDK: no ThinkingConfig and/or no seed
            try:
                config = types.GenerateContentConfig(**kwargs)
                break
            except TypeError:
                continue
    if config is None:
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=list[str]
        )

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
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
    base_extra_rules: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = DEFAULT_SEED,
) -> List[str]:
    attempt = 0
    quality_attempt = 0
    markup_attempt = 0
    keyword_attempt = 0
    last_partial: Optional[List[str]] = None
    quality_prompt_compact = compact_prompt
    extra_rules = base_extra_rules
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
                temperature=temperature,
                seed=seed,
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
                        extra_rules = (base_extra_rules + "\n\n" + STRICT_QUALITY_RULES).strip()
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

            # Markup gate: same shape as the residue gate above, but language-agnostic --
            # a lost <color> tag costs the player the "strong against" cue in every language.
            broken = None
            for src_text, out_text in zip(batch, translations):
                if not markup_integrity_ok(src_text, out_text):
                    broken = (src_text, out_text)
                    break
            if broken:
                if markup_attempt < max_quality_retries:
                    markup_attempt += 1
                    quality_prompt_compact = False
                    extra_rules = (extra_rules + "\n\n" + STRICT_MARKUP_RULES).strip()
                    logging.warning(
                        "Markup retry %s/%s: <color> markup broken. src=%s out=%s",
                        markup_attempt,
                        max_quality_retries,
                        broken[0],
                        broken[1],
                    )
                    continue
                logging.warning(
                    "Markup retries exhausted; <color> markup still broken. src=%s out=%s",
                    broken[0],
                    broken[1],
                )

            # Unit-class gate: the tags may be intact and the word inside them still wrong.
            # markup_integrity_ok cannot see this -- it only counts tags -- and naming another
            # class is worse than losing the tag, because the text then states a false counter.
            wrong_class = None
            for src_text, out_text in zip(batch, translations):
                if not counter_keyword_ok(src_text, out_text, target):
                    wrong_class = (src_text, out_text)
                    break
            if wrong_class:
                if keyword_attempt < max_quality_retries:
                    keyword_attempt += 1
                    quality_prompt_compact = False
                    # The lexicon itself is already in base_extra_rules; only the emphasis is new.
                    extra_rules = (extra_rules + "\n\n" + STRICT_COUNTER_KEYWORD_RULES).strip()
                    logging.warning(
                        "Unit-class retry %s/%s: coloured keyword names the wrong class. "
                        "src=%s out=%s",
                        keyword_attempt,
                        max_quality_retries,
                        wrong_class[0],
                        wrong_class[1],
                    )
                    continue
                logging.warning(
                    "Unit-class retries exhausted; coloured keyword still wrong. src=%s out=%s",
                    wrong_class[0],
                    wrong_class[1],
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


def terminology_fingerprint(
    original_text: str,
    target_lang: str,
    user_glossary: Optional[Dict[str, str]] = None,
) -> str:
    """A short hash of every terminology rule that applies to this one string.

    Editing glossary.txt or a canon does not invalidate the cache -- by design, since the key is
    the source text -- so a wording decision used to reach only the strings that happened to be
    re-translated afterwards, and the only way to force the rest was to purge everything the
    audit flagged. Recording which rules produced a string makes the narrow question answerable:
    "whose rules changed since?".

    Only the rules that actually fire for this string go into the hash, so adding an unrelated
    term does not invalidate the whole cache.
    """
    parts: List[str] = [target_lang or ""]
    if target_is_spanish(target_lang):
        for entry in SPANISH_GLOSSARY:
            if not entry.source_trigger.search(original_text):
                continue
            if entry.source_block is not None and entry.source_block.search(original_text):
                continue
            fixes = "|".join(f"{pat.pattern}=>{rep}" for pat, rep in entry.output_fixes)
            parts.append(f"{entry.name}:{entry.prompt_hint}:{fixes}")
        for code, inner in _counter_spans(original_text):
            lexicon = _COUNTER_KEYWORDS_BY_CODE.get(code, {})
            parts.append(f"kw:{code}:{inner.strip().lower()}:" + ",".join(sorted(lexicon.values())))
    for entry in _glossary_entries(user_glossary):
        if entry.source_term in original_text and entry.applies_to(original_text):
            parts.append(f"ug:{entry.source_term}={entry.target_term}")
    digest = hashlib.blake2s("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def _terms_sidecar_path(cache_path: Path) -> Path:
    """Where the fingerprints live: next to the cache, never inside it.

    The cache format ({protected source: translation}, "" meaning "retry me") is the project's
    load-bearing invariant, and every key in it is a real source string. Adding a metadata key
    would put a non-string in that namespace, so the fingerprints get their own file. A missing
    sidecar simply means "unknown", which behaves exactly like the old code.
    """
    return cache_path.with_name(cache_path.stem + ".terms.json")


def _load_terms_sidecar(cache_path: Optional[Path]) -> Dict[str, str]:
    if not cache_path:
        return {}
    path = _terms_sidecar_path(cache_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logging.warning("Unable to load terminology fingerprints (%s): %s", path, exc)
        return {}


def _write_terms_sidecar(cache_path: Optional[Path], fingerprints: Dict[str, str]) -> None:
    if not cache_path or not fingerprints:
        return
    path = _terms_sidecar_path(cache_path)
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(fingerprints, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        logging.warning("Could not write terminology fingerprints (%s): %s", path, exc)


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


def finalize_translation(
    original_text: str,
    translated_text: str,
    target_lang: str,
    user_glossary: Optional[Dict[str, str]] = None,
    acronym_exclude: Optional[Sequence[str]] = None,
) -> str:
    """The deterministic last mile, shared by every path that produces a finished string.

    Cache hits, fresh translations and the consistency sweep must all end up identical, so the
    order of these gates lives in exactly one place.
    """
    out = apply_postprocess_overrides(original_text, translated_text, target_lang)
    out = apply_counter_keyword_fixes(original_text, out, target_lang)
    out = apply_user_glossary_fixes(original_text, out, user_glossary)
    out = normalize_latam_spanish(original_text, out, target_lang)
    out = enforce_acronym_integrity(original_text, out, exclude=acronym_exclude)
    return apply_source_casing(original_text, out)


def run_consistency_sweep(
    client,
    originals: Sequence[str],
    translations: Sequence[str],
    source_lang: str,
    target_lang: str,
    *,
    max_retries: int,
    prompt_config: PromptConfig,
    strict_no_english_residue: bool,
    max_budget_bytes: int,
    user_glossary: Optional[Dict[str, str]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = DEFAULT_SEED,
    limit: int = 400,
) -> Dict[str, str]:
    """One bounded pass that re-translates the minority renderings of a repeated term.

    Batches run in eight independent threads with no shared state, so "translate identical
    strings identically" is unenforceable across them -- that is why temperature and seed are
    pinned. What neither can fix is one TERM coming out two ways in different batches, because
    no batch ever sees the other's choice.

    This closes that loop after the fact: the game's own short label strings say what each thing
    is called, so any longer string that disagrees with its label is re-translated once with the
    label's wording passed in as an explicit rule.

    Deliberately ONE pass and capped at ``limit`` strings: it is a cleanup, not a search, and it
    must never turn into an unbounded spend. Returns {original text: new translation}.
    """
    pairs = list(zip(originals, translations))
    divergences = [row for row in _audit_discover_candidates(pairs, limit=60) if row[4]]
    if not divergences:
        return {}

    # One string can diverge on several terms; collect every rule that applies to it.
    rules_by_source: Dict[str, List[str]] = {}
    for label, canonical, _ok, _count, sources in divergences:
        for src in sources:
            rules = rules_by_source.setdefault(src, [])
            rule = f"- Translate '{label}' as '{canonical}' (this is the name the game shows)."
            if rule not in rules:
                rules.append(rule)
        if len(rules_by_source) >= limit:
            break

    targets = list(rules_by_source)[:limit]
    print(f"🔁 Consistency sweep: {len(targets)} string(s) disagree with their own label; "
          f"re-translating them once with the label's wording.")

    fixed: Dict[str, str] = {}
    # Group by identical rule set so each batch carries one coherent instruction block.
    by_rules: Dict[Tuple[str, ...], List[str]] = {}
    for src in targets:
        by_rules.setdefault(tuple(rules_by_source[src]), []).append(src)

    for rules, group in by_rules.items():
        extra = "CONSISTENCY (use the game's own names)\n" + "\n".join(rules)
        base = "\n\n".join(part for part in (
            counter_keyword_rules(group, target_lang),
            user_glossary_rules(group, user_glossary),
            extra,
        ) if part)
        for batch in yield_batches(group, max_budget_bytes):
            try:
                out = translate_batch_with_retry(
                    client, batch, source_lang, target_lang, max_retries,
                    False, prompt_config, strict_no_english_residue,
                    base_extra_rules=base, temperature=temperature, seed=seed,
                )
            except Exception as exc:
                logging.warning("Consistency sweep batch failed, keeping the old wording: %s", exc)
                continue
            for src_text, out_text in zip(batch, out):
                # Same guards as the main path: never take a candidate that is worse.
                if not out_text or not out_text.strip():
                    continue
                if strict_no_english_residue and has_english_residue(src_text, out_text, target_lang):
                    continue
                if not markup_integrity_ok(src_text, out_text):
                    continue
                if not counter_keyword_ok(src_text, out_text, target_lang):
                    continue
                fixed[src_text] = out_text
    return fixed


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
    retranslate_stale_terms: bool = False,
    consistency_sweep: bool = False,
    api_timeout_seconds: int = DEFAULT_API_TIMEOUT,
    batch_progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    user_glossary: Optional[Dict[str, str]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = DEFAULT_SEED,
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
    # Cached strings whose terminology rules changed since they were translated. Unknown
    # fingerprints are never stale, so a cache with no sidecar behaves exactly as before.
    term_fingerprints: Dict[str, str] = _load_terms_sidecar(cache_path)
    stale_keys: set[str] = set()
    if retranslate_stale_terms and term_fingerprints:
        for key, idxs in indexes_by_protected.items():
            stored = term_fingerprints.get(key)
            if stored is None or not idxs:
                continue
            if stored != terminology_fingerprint(original_texts[idxs[0]], target_lang, user_glossary):
                stale_keys.add(key)
        if stale_keys:
            print(f"♻️  {len(stale_keys)} cached string(s) have stale terminology; re-translating.")

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

        # This loop walks the SLOTS, so the same string arrives once per occurrence. Once a
        # string is queued its cache entry holds our own "" placeholder, and every later
        # occurrence used to fall into the "cached empty" branch below and be counted as a
        # previous failure -- one real run reported 4646 unresolved strings when 37 were.
        # Its slots are filled from the batch result either way, so skip it here.
        if text in already_enqueued:
            continue

        if cached_value and cached_value.strip() and text in stale_keys:
            # Its terminology rules changed: drop the cached wording and re-translate it.
            stats.terms_stale += len(indexes_by_protected.get(text, []))
            cache[text] = ""
            if text not in already_enqueued:
                already_enqueued.add(text)
                unique_to_translate.append(text)
            continue

        if cached_value and cached_value.strip():
            # We already had a cached translation: reuse it everywhere and skip re-translation.
            stats.cache_used += len(indexes_by_protected.get(text, []))
            for idx in indexes_by_protected.get(text, []):
                restored = restore_all_tokens(
                    cached_value, token_maps[idx], phrase_maps[idx], original_texts[idx]
                )
                translations[idx] = finalize_translation(
                    original_texts[idx], restored, target_lang, user_glossary, acronym_exclude
                )
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

    def _apply_consistency_sweep(active_client) -> None:
        """Re-translate the strings that disagree with the game's own labels, once."""
        keys = [key for key in indexes_by_protected if (cache.get(key) or "").strip()]
        if not keys:
            return
        fixed = run_consistency_sweep(
            active_client,
            keys,
            [cache[key] for key in keys],
            source_lang,
            target_lang,
            max_retries=max_retries,
            prompt_config=prompt_config,
            strict_no_english_residue=strict_no_english_residue,
            max_budget_bytes=max_budget_bytes,
            user_glossary=user_glossary,
            temperature=temperature,
            seed=seed,
        )
        for key, value in fixed.items():
            idxs = indexes_by_protected.get(key, [])
            if not idxs:
                continue
            cache[key] = value
            stats.consistency_fixed += len(idxs)
            term_fingerprints[key] = terminology_fingerprint(
                original_texts[idxs[0]], target_lang, user_glossary
            )
            for idx in idxs:
                restored = restore_all_tokens(
                    value, token_maps[idx], phrase_maps[idx], original_texts[idx]
                )
                translations[idx] = finalize_translation(
                    original_texts[idx], restored, target_lang, user_glossary, acronym_exclude
                )

    if cache_only or not unique_to_translate:
        # A fully cached run still deserves the sweep when it was asked for.
        if consistency_sweep and api_key and not cache_only:
            try:
                _apply_consistency_sweep(
                    setup_gemini(api_key, timeout_seconds=api_timeout_seconds)
                )
            except Exception as exc:
                logging.warning("Consistency sweep skipped: %s", exc)
        if cache_path:
            try:
                _write_cache_atomic(cache_path, cache)
            except Exception as exc:
                logging.warning("Failed to write cache file: %s", exc)
            _write_terms_sidecar(cache_path, term_fingerprints)
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
                base_extra_rules="\n\n".join(
                    part for part in (
                        counter_keyword_rules(batch, target_lang),
                        user_glossary_rules(batch, user_glossary),
                    ) if part
                ),
                temperature=temperature,
                seed=seed,
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
                    stats.batch_failed += len(indexes_by_protected.get(original, []))
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
                        stats.quality_rejected += len(indexes_by_protected.get(original, []))
                        continue
                    if not markup_integrity_ok(original, translated_item):
                        # Kept (a broken tag beats an untranslated string) but counted, so the
                        # user can find them with --audit-spanish instead of never knowing.
                        stats.markup_rejected += len(indexes_by_protected.get(original, []))
                    if not counter_keyword_ok(original, translated_item, target_lang):
                        # Same policy: kept, counted, and listed by --audit-spanish. The
                        # deterministic repair below still fixes the same-gender variants.
                        stats.counter_keyword_rejected += len(
                            indexes_by_protected.get(original, [])
                        )
                    stats.api_translated += len(indexes_by_protected.get(original, []))
                    cache[original] = translated_item
                    # Record which rules produced this wording. Only freshly translated strings
                    # are stamped: back-filling the whole cache with today's rules would mark
                    # the very strings a rule change is meant to catch as already up to date.
                    idxs = indexes_by_protected.get(original, [])
                    if idxs:
                        term_fingerprints[original] = terminology_fingerprint(
                            original_texts[idxs[0]], target_lang, user_glossary
                        )
                    for idx in indexes_by_protected.get(original, []):
                        restored = restore_all_tokens(
                            translated_item,
                            token_maps[idx],
                            phrase_maps[idx],
                            original_texts[idx],
                        )
                        translations[idx] = finalize_translation(
                            original_texts[idx], restored, target_lang, user_glossary,
                            acronym_exclude,
                        )

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

    if consistency_sweep:
        try:
            _apply_consistency_sweep(client)
        except Exception as exc:
            logging.warning("Consistency sweep skipped: %s", exc)

    # Final cache flush after all batches complete so we never lose the last in-memory updates.
    if cache_path:
        with _cache_lock:
            try:
                _write_cache_atomic(cache_path, cache)
            except Exception as exc:
                logging.warning("Failed final cache write: %s", exc)
            _write_terms_sidecar(cache_path, term_fingerprints)

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


@dataclass(frozen=True)


class BuildCacheStats:
    """Outcome of building a cache from an already-translated file."""

    paired: int = 0                # new-source strings matched to a translation
    seeded_reused: int = 0         # a real translation (differs from the English)
    seeded_identical: int = 0      # translation == English and judged safe (proper nouns, markup)
    skipped_english: int = 0       # translation == English but it IS untranslated English
    skipped_placeholder: int = 0   # %-format placeholders do not match; unsafe to reuse
    skipped_unmatched: int = 0     # no translation found for this _locID / content

    @property
    def written(self) -> int:
        return self.seeded_reused + self.seeded_identical


def _identical_translation_is_safe(text: str, target_lang: str) -> bool:
    """Decide whether a translation identical to its source may be cached.

    Two very different things produce an identical string:
      * a proper noun / pure markup that legitimately survives translation
        ("Yamabushi", "Alexander von Humboldt", "<color=...>__TOK0__</color>") -- caching it
        is valuable: it stops the next run from paying for it and from letting the model
        "translate" a name it should have left alone;
      * real source-language text the previous translator never got to -- caching that would
        poison the cache, which is exactly what the cache contract forbids.

    For a Spanish target we can tell them apart with has_english_residue (the gate already
    used elsewhere for this). For any other target we have no such detector, so we fall back
    to the conservative rule merge_by_locid already applies: only keep it when there is
    nothing translatable in it at all.
    """
    if not _has_translatable_text(text):
        return True
    if target_is_spanish(target_lang):
        return not has_english_residue(text, text, target_lang)
    return False


def build_cache_from_translation(
    source_targets: Sequence[TranslationTarget],
    translated_targets: Sequence[TranslationTarget],
    *,
    protected_terms: Optional[Sequence[str]] = None,
    protected_regex: Optional[Sequence["re.Pattern[str]"]] = None,
    acronym_exclude: Optional[Sequence[str]] = None,
    target_lang: str = DEFAULT_TARGET_LANG,
    existing_cache: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], BuildCacheStats]:
    """Build a {cache_key: translation} cache from a source XML + its translated XML. No API.

    Pairs the two files by ``_locID`` through merge_by_locid (passing the source file as both
    the "new" and the "old" source, so every entry compares as unchanged and the seed is the
    reusable translation, inheriting its duplicate-id and content fallbacks).

    It deliberately does NOT reuse merge_by_locid's seeding policy wholesale. That policy
    serves version updates, where a translation identical to the source means "still needs
    translating". Here the same input usually means a proper noun we want to keep, so
    _identical_translation_is_safe re-decides those -- see its docstring.

    Keys come from protected_cache_key, so they are byte-identical to the ones
    translate_strings reads (guarded by self_test_cache_key_parity).
    """
    report = merge_by_locid(source_targets, source_targets, translated_targets)

    cache: Dict[str, str] = dict(existing_cache) if existing_cache else {}
    paired = seeded_reused = seeded_identical = 0
    skipped_english = skipped_placeholder = skipped_unmatched = 0

    for entry in report.entries:
        if entry.draft is None:
            skipped_unmatched += 1
            continue
        paired += 1

        if entry.reason == "placeholder-mismatch":
            skipped_placeholder += 1
            continue

        if entry.seed is not None:
            value, identical = entry.seed, (entry.reason == "kept-as-source")
        elif entry.reason == "old-equals-source":
            # merge_by_locid refuses these; re-decide with the target-aware rule.
            if not _identical_translation_is_safe(entry.new_source, target_lang):
                skipped_english += 1
                continue
            value, identical = entry.new_source, True
        else:
            skipped_unmatched += 1
            continue

        cache[protected_cache_key(
            entry.new_source,
            protected_terms=protected_terms,
            protected_regex=protected_regex,
            acronym_exclude=acronym_exclude,
        )] = value
        if identical:
            seeded_identical += 1
        else:
            seeded_reused += 1

    return cache, BuildCacheStats(
        paired=paired,
        seeded_reused=seeded_reused,
        seeded_identical=seeded_identical,
        skipped_english=skipped_english,
        skipped_placeholder=skipped_placeholder,
        skipped_unmatched=skipped_unmatched,
    )


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
    # "host" is deliberately in ENGLISH_RESIDUE_STOPWORDS (translate it as "anfitrión"),
    # so a Spanish translation that keeps "host" must be flagged too.
    out2_residue = "Introduce la dirección IP del host para conectar mediante IP directa."
    out2_good = "Introduce la dirección IP del anfitrión para conectar mediante IP directa."
    _assert(has_english_residue(src2, out2_bad, target_lang), "expected residue for IP address prompt")
    _assert(has_english_residue(src2, out2_residue, target_lang), "expected residue for kept 'host'")
    _assert("IP" in out2_good, "expected IP to remain unchanged")
    _assert(not has_english_residue(src2, out2_good, target_lang), "expected no residue in Spanish translation")

    # REGRESSION: the game title is not protected by name alone, so its "of" used to trip the
    # gate and 144 correctly-translated strings shipped in English. The gate sees PROTECTED text,
    # so reproduce that here rather than passing the raw string.
    def _protected(text: str) -> str:
        terms, regex, exclude = _normalize_protection(None, None, None)
        key, _tok, _phr = protect_for_cache(text, terms, regex, exclude)
        return key

    for src_raw, out_raw, why in [
        ("There is an updated version of Age of Empires III that is required to play.",
         "Hay una versión actualizada de Age of Empires III que se requiere para jugar.",
         "the bare game title must not count as English residue"),
        ("Welcome to Age of Empires III: Wars of Liberty.",
         "Bienvenido a Age of Empires III: Wars of Liberty.",
         "the full mod title must not count as English residue"),
        ("This is the original version.", "Esta es la versión original.",
         "'original' and 'versión' are ordinary Spanish words"),
    ]:
        _assert(
            not has_english_residue(_protected(src_raw), _protected(out_raw), target_lang),
            f"false positive: {why} ({out_raw!r})",
        )

    # Markup internals are not prose: an icon path carrying "War of the Triple Alliance" must
    # not make a perfectly translated string ship in English.
    icon = ('Cada Cuartel genera 2 Klephts.<font=Arial 96>'
            '<icon="(58)(War of the Triple Alliance\\Icons\\huntinglodge)"></font>')
    _assert(not has_english_residue("Each Barracks spawns 2 Klephts.", icon, target_lang),
            "English inside a markup tag must not be read as residue")
    _assert(has_english_residue("x", "<color=1,1,1>Increases the attack of the unit</color>",
                                target_lang),
            "residue in the text a tag WRAPS must still be caught")

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

    # Generic epoch sense: "By Age" was shipping as "Por Era" (seen in-game).
    for src, bad, expected in [
        ("By Age", "Por Era", "Por Edad"),
        ("You are still in the previous Age?", "¿Todavía estás en la Era anterior?", "Edad anterior"),
        ("advance through the Ages", "avanzar por las Eras", "las Edades"),
        ("Advance to the Next Age", "Avanzar a la Siguiente Era", "Siguiente Edad"),
        ("Each card gives one unit per Age.", "Cada carta da una unidad por Era.", "por Edad"),
    ]:
        out = f(src, bad, target_lang)
        _assert(expected in out, f"expected {expected!r} in {out!r} (source {src!r})")

    # Terminology entries added from the measured corpus.
    _assert("Puntos de Vida" in f("Settler hitpoints increased.", "Puntos de Golpe aumentados.", target_lang),
            "hitpoints should become Puntos de Vida")
    _assert(f("hitpoints increased", "puntos de resistencia aumentados", target_lang)
            == "puntos de vida aumentados", "lowercase hitpoints must stay lowercase")
    _assert(f("Train a Settler", "Entrena un Aldeano", target_lang) == "Entrena un Colono",
            "Settler should become Colono")
    _assert(f("Shipment has arrived.", "Cargamento ha llegado.", target_lang) == "Envío ha llegado.",
            "Shipment should become Envío")
    _assert(f("A patch of potatoes", "Un sembrado de patatas", target_lang) == "Un sembrado de papas",
            "potatoes should become papas")

    # NEGATIVE cases, every one taken from a real string in the shipped translation.
    # A ship's deck really is a "cubierta"; the deck rule must never touch it.
    _assert(f("Steel Decks", "Cubiertas de Acero", target_lang) == "Cubiertas de Acero",
            "Steel Decks must stay Cubiertas de Acero")
    # Crate and Chest coexist in one string and are correctly two different words.
    both = "Cajas de 500 alimento y Cofres de 500 moneda"
    _assert(f("Crates of 500 food and Chests of 500 coin", both, target_lang) == both,
            "Cofres must survive when the source mentions Chests too")
    # Settler and Villager are different units; when both appear, Aldeano is legitimate.
    mixed = "Aldeanos y Colonos trabajan juntos"
    _assert(f("Villagers and Settlers work together", mixed, target_lang) == mixed,
            "Aldeano must survive when the source mentions Villager too")
    # "Christmas Card" is a greeting card, not a game card.
    _assert(f("Christmas Card", "Tarjeta de Navidad", target_lang) == "Tarjeta de Navidad",
            "Christmas Card must stay Tarjeta de Navidad")

    # --- Wars of Liberty unit/building names ------------------------------------------------
    for src, bad, expected in [
        # "Escararuzador" is Escara+R+uzador, a real misspelling shipped in 9 strings.
        ("Light skirmisher", "Escararuzador ligero", "Hostigador ligero"),
        ("Skirmisher attack", "Escaramuzador mejorado", "Hostigador mejorado"),
        ("skirmishers shoot faster", "Los escararuzadores disparan", "Los hostigadores disparan"),
        ("Hand cavalry, good against skirmishers",
         "Caballería de mano, buena contra hostigadores",
         "Caballería cuerpo a cuerpo, buena contra hostigadores"),
        ("Hand infantry unit", "Infantería de Mano", "Infantería cuerpo a cuerpo"),
        ("Hajduk attack increased", "Ataque de Hayduk aumentado", "Ataque de Hajduk aumentado"),
        ("The Boneguard attack", "La Guardia de Hueso ataca", "La Guardia Ósea ataca"),
        ("Boneguard Fort", "Fuerte Guardahuesos", "Fuerte Guardia Ósea"),
        ("Ali Pasha revives", "Ali Pashá revive", "Ali Pasha revive"),
        ("Muhammad Ali Pasha", "Muhammad Ali Bajá", "Muhammad Ali Pasha"),
        ("Madrasah technologies", "Tecnologías de la Madraza", "Tecnologías de la Madrasa"),
        ("Afghan warlord", "Señor de la guerra afgano", "Caudillo afgano"),
        ("3 Righteous Fighters", "3 Combatientes Justos", "3 Guerreros Justos"),
        ("Calls Righteous Fighters", "Invoca Luchadores Justicieros", "Invoca Guerreros Justos"),
        ("Hunting Lodge", "Logia de Caza", "Cabaña de Caza"),
        ("Square", "Cuadrado", "Cuadro"),
        ("Argentine Conscript", "Conscrito Argentino", "Conscripto Argentino"),
        # 'To revolt' is the Revolution mechanic, not 'revolverse' (to stir).
        ("Revolting is cheaper", "Revolverse es más barato", "Sublevarse es más barato"),
        ("Revolt!", "¡Revuélvanse!", "¡Sublévense!"),
        ("Aging up and revolting is cheaper.", "Avanzar de edad y revolucionarse es más barato.",
         "Avanzar de edad y sublevarse es más barato."),
    ]:
        out = f(src, bad, target_lang)
        _assert(out == expected, f"expected {expected!r}, got {out!r} (source {src!r})")

    # NEGATIVES for the new entries. Each one is a real string from the shipped translation
    # that a careless rule would have corrupted.
    for src, text, why in [
        # 'Conscript' is also a verb; this action string is already correct.
        ("Conscript Sepoys", "Reclutar Sepoys", "the verb sense must not be touched"),
        # 'Recluta' is the separate unit 'Recruit' -- rewriting it would be a worse error.
        ("Recruit and Hajduk range increased", "Se aumenta el alcance de Reclutas y Hajduk",
         "Recluta is another unit and must survive"),
        ("Town Square", "Plaza del Pueblo", "a town square really is a plaza"),
        ("Land Reallotment", "Reasignación de Tierras", "Reallotment is a different card"),
        ("Masonic Lodge", "Logia Masónica", "a Masonic lodge really is a Logia"),
        # 'Revolt' has three senses here and only the verb one was wrong.
        ("Colt Revolver", "Revólver Colt", "the Colt Revolver is a weapon, not a revolt"),
        ("Revolver Hammer", "Martillo de Revólver", "same weapon sense"),
        ("Arab Revolt", "Revuelta Árabe", "a named uprising really is a Revuelta"),
        ("Bolívar's Revolt", "La Revuelta de Bolívar", "same"),
        ("You have revolted from your mother country!", "¡Te has sublevado de tu metrópoli!",
         "already the canonical verb"),
        ("The King found me quite revolting.", "El Rey me encontró bastante repugnante.",
         "'revolting' also means disgusting"),
    ]:
        out = f(src, text, target_lang)
        _assert(out == text, f"{text!r} must stay untouched ({why}), got {out!r}")

    # Gender-changing terms are prompt-only: they must never be rewritten here.
    for src, text in [
        ("Allotments are 10% cheaper.", "Las Parcelas son un 10% más baratas."),
        ("Spanish Square", "Plaza Española"),
        ("Zapotec Settlement", "Asentamiento zapoteco"),
    ]:
        _assert(f(src, text, target_lang) == text,
                f"{text!r} changes gender; it must be left to the prompt layer")

    print("✅ Glossary self-test passed.")


def self_test_markup_integrity() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Markup integrity self-test failed: {message}")

    green = "<color=0.07, 0.68, 0.17>"
    blue = "<color=0.19, 0.52, 0.76>"

    # The two failure shapes actually observed in the shipped translation.
    _assert(
        not markup_integrity_ok(
            f"Nepalese {green}skirmisher </color>that is accurate.",
            f"infantería a distancia nepalí {green} </color>con precisión."),
        "an emptied <color> pair must be rejected")
    _assert(
        not markup_integrity_ok(
            f"Native Asian ranged {blue}line unit</color>",
            "Unidad de línea a distancia asiática nativa"),
        "a dropped <color> pair must be rejected")

    # Correct markup, and text without any, must pass.
    _assert(markup_integrity_ok(f"{green}Skirmisher </color>with low hitpoints.",
                                f"{green}infantería a distancia </color>con pocos puntos de vida."),
            "correctly moved markup must pass")
    _assert(markup_integrity_ok("Plain text, no markup.", "Texto plano, sin markup."),
            "text without markup must pass")
    _assert(markup_integrity_ok(f"Good against {green}a</color> and {blue}b</color>",
                                f"Bueno contra {green}a</color> y {blue}b</color>"),
            "two intact pairs must pass")
    # A source that is itself empty-tagged must not be flagged; we only catch NEW damage.
    _assert(markup_integrity_ok(f"{green}</color>", f"{green}</color>"),
            "an already-empty pair in the source is not our defect")

    print("✅ Markup integrity self-test passed.")


def self_test_counter_keywords() -> None:
    """The coloured word must keep naming the right unit class."""
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Counter keyword self-test failed: {message}")

    target = "Latin American Spanish"
    green = "<color=0.07, 0.68, 0.17>"      # skirmisher class
    shock = "<color=0.92, 0.84, 0.09>"      # shock units
    assault = "<color=0.74, 0.25, 0.11>"    # assault units
    standing = "<color=0.74, 0.11, 0.58>"   # standing units
    scout = "<color=0.82, 0.80, 0.60>"      # scout units

    ok = counter_keyword_ok
    # The defect the community reported: correct markup, wrong class inside it.
    _assert(not ok(f"Improves melee attack of {green}skirmishers</color>.",
                   f"Mejora el ataque de los {green}hostigadores</color>.", target),
            "the old term must be rejected under the green class colour")
    _assert(ok(f"Improves melee attack of {green}skirmishers</color>.",
               f"Mejora el ataque de la {green}infantería a distancia</color>.", target),
            "the canon must be accepted")
    # An English singular rendered as a Spanish plural is not a defect.
    _assert(ok(f"A {green}skirmisher</color> unit.",
               f"Unidades de {green}infantería a distancia</color>.", target),
            "another canon of the same colour must be accepted")
    # Naming a DIFFERENT class is the worst case: the text then states a false counter.
    _assert(not ok(f"Good against {shock}shock units</color>.",
                   f"Eficaz contra {shock}unidades de asalto</color>.", target),
            "one class must not be labelled with the name of another")
    _assert(ok(f"Good against {assault}assault units</color>.",
               f"Eficaz contra {assault}unidades de asalto</color>.", target),
            "the same words are correct under the assault colour")
    _assert(not ok(f"Good against {green}skirmishers</color>.",
                   f"Eficaz contra {green}</color>.", target),
            "an emptied class tag must be rejected")
    # "Cazador" is the Cacadore unit; using it here makes two units look like one.
    _assert(not ok(f"Light {green}skirmishers</color>",
                   f"Ligeros {green}cazadores</color>", target),
            "another unit's name must not be used for the class")
    # Not Spanish -> no canon table -> never fails.
    _assert(ok(f"Good against {green}skirmishers</color>.", "irrelevant", "German"),
            "non-Spanish targets must pass unchanged")

    fix = apply_counter_keyword_fixes
    _assert(fix(f"Good against {standing}standing units</color>.",
                f"Buena contra {standing}unidades estacionarias</color>.", target)
            == f"Buena contra {standing}unidades estáticas</color>.",
            "same-gender variants must be repaired deterministically")
    _assert(fix(f"{scout}Scout cavalry </color>with a lance.",
                f"{scout}caballería exploradora </color>con lanza.", target)
            == f"{scout}caballería de exploración </color>con lanza.",
            "the whitespace inside the tag is part of the sentence and must survive")
    _assert(fix(f"Good against {assault}assault units</color>.",
                f"Eficaz contra {assault}unidades de asalto</color>.", target)
            == f"Eficaz contra {assault}unidades de asalto</color>.",
            "a correct canon must never be rewritten because another colour dislikes it")
    # The green class changes gender, so a regex must NOT touch it -- see the comment on
    # COUNTER_KEYWORD_REPAIRS. Repairing it would produce "los infantería a distancia".
    stale = f"Mejora el ataque de los {green}hostigadores</color>."
    _assert(fix(f"Improves melee attack of {green}skirmishers</color>.", stale, target) == stale,
            "the gender-changing class must be left for re-translation, never regexed")

    rules = counter_keyword_rules([f"Good against {shock}shock units</color>."], target)
    _assert("unidades de choque" in rules and "infantería a distancia" not in rules,
            "only the colours present in the batch may reach the prompt")

    print("✅ Counter keyword self-test passed.")


def self_test_skirmisher_class() -> None:
    """'Skirmisher' is the unit CLASS inside the green tag and the unit NAME outside it."""
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Skirmisher class self-test failed: {message}")

    target = "Latin American Spanish"
    green = "<color=0.07, 0.68, 0.17>"
    tagged = f"{green}Skirmisher </color>hand attack increased."
    plain = "8 Skirmishers"

    tagged_rules = terminology_overrides_for_target(target, [tagged])
    _assert("infantería a distancia" in tagged_rules,
            "the class rule must reach a batch that carries the green tag")
    _assert("'Hostigador'" not in tagged_rules,
            "the unit-name rule must be blocked when the green tag is present")

    plain_rules = terminology_overrides_for_target(target, [plain])
    _assert("'Hostigador'" in plain_rules,
            "the unit-name rule must fire without the tag")
    _assert("infantería a distancia" not in plain_rules,
            "the class rule must not fire without the tag")

    # A batch carrying both contexts must carry both rules: source_block is per string.
    both = terminology_overrides_for_target(target, [tagged, plain])
    _assert("'Hostigador'" in both and "infantería a distancia" in both,
            "a mixed batch must keep both rules")

    # No output_fixes for the class: the gender change makes a regex unsafe.
    stale = f"Ataque de los {green}hostigadores </color>aumentado."
    _assert(apply_postprocess_overrides(tagged, stale, target) == stale,
            "the class must never be regex-swapped")
    # The unit name keeps its same-gender repair.
    _assert(apply_postprocess_overrides("Light skirmisher", "Escararuzador ligero", target)
            == "Hostigador ligero",
            "the misspelled unit name must still be repaired")

    # The audit must split the two cases as well.
    ok_class, bad_class = _audit_term_consistency(
        [(tagged, stale)], "skirmisher", "infantería a distancia",
        ["Hostigador"], None, GREEN_SKIRMISHER_RE)
    _assert(bad_class and not ok_class, "the audit must flag the old term inside the green tag")
    ok_unit, bad_unit = _audit_term_consistency(
        [(plain, "8 Hostigadores")], "skirmisher", "Hostigador",
        ["Guerrillero"], GREEN_SKIRMISHER_RE, None)
    _assert(ok_unit and not bad_unit, "the audit must leave the unit name alone")

    print("✅ Skirmisher class self-test passed.")


def self_test_misalignment() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Misalignment self-test failed: {message}")

    # The real _locID 42595 pair from the shipped file: a description slot holding a name.
    real = [("Upgraded version of the red and gold French New World Trade Center.",
             "Suministros de J.C. Kiley")]
    _assert(len(_audit_misaligned(real)) == 1, "the real 42595 misalignment must be detected")

    # Its swapped partner at 42596: a short English name whose slot holds the long description.
    # Both halves must be caught or a purge would fix one slot and leave the other wrong.
    mirror = [("J.C. Kiley's Outfitters",
               "Versión mejorada del Centro de Comercio francés del Nuevo Mundo azul y blanco.")]
    _assert(len(_audit_misaligned(mirror)) == 1, "the mirror half of a swap must be detected too")

    # The same shift between two SHORT labels, where the length rule is blind. These are the
    # worst kind: the card promises one number of units and the game shows another.
    _assert(len(_audit_misaligned([("8 Riflemen", "6 Fusileros")])) == 1,
            "a shifted unit count must be detected")
    _assert(not _audit_misaligned([("8 Riflemen", "8 Fusileros")]),
            "a matching unit count must not be flagged")
    # Legitimate cases the count rule must stay away from.
    _assert(not _audit_misaligned([("1 Settler Wagon", "una Carreta de Colono")]),
            "a number written as a word is not a mismatch")
    _assert(not _audit_misaligned([
        ("March 1421 - We sail with 300 ships and 20,000 crew aboard the fleet.",
         "Marzo de 1421 - Zarpamos con 300 barcos y 20.000 tripulantes a bordo.")]),
        "a thousands separator is formatting, not a changed quantity")

    # NEGATIVES: everything a false positive would cost an unnecessary API call.
    safe = [
        # A normal, complete translation.
        ("Villagers gather wood faster from Mills and Plantations everywhere.",
         "Los Aldeanos recolectan madera más rápido de Molinos y Plantaciones en todas partes."),
        # Spanish is legitimately more compact but still a sentence.
        ("Upgraded version of the original Russian New World Trade Center.",
         "Versión mejorada del Centro de Comercio ruso original."),
        # Short label translated by a short label: the rule must not look at these at all.
        ("Gang Saw", "Sierra de banda"),
        # Long English WITHOUT a final period is out of scope (headings, list items).
        ("An offensive army of swordsmen and skirmishers ready for the front",
         "Ejército ofensivo"),
    ]
    for src, tgt in safe:
        _assert(not _audit_misaligned([(src, tgt)]),
                f"false positive on {tgt!r}: it would be re-translated for nothing")

    # Markup and placeholders must not count toward the length comparison.
    _assert(not _audit_misaligned([
        ("<color=1.0, 1.0, 0.0>Ships %1s and %2s to your colony right away.</color>",
         "<color=1.0, 1.0, 0.0>Envía %1s y %2s a tu colonia de inmediato.</color>")]),
        "markup and placeholders must be stripped before measuring length")

    print("✅ Misalignment self-test passed.")


def self_test_repair_from() -> None:
    """Pin the accept/reject rules that decide when a trusted cache value may be reused.

    Mirrors the logic in run_audit_spanish_cli's --repair-from block. The negatives matter most:
    a donor cache can be wrong too, and swapping one error for another would be worse than
    leaving the string alone, because the audit would then stop flagging it.
    """
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Repair-from self-test failed: {message}")

    def accepts(src: str, current: str, good: Optional[str]) -> bool:
        if not good or not good.strip() or good.strip() == (current or "").strip():
            return False
        if _audit_misaligned([(src, good)]):
            return False
        return placeholders_compatible(src, good)

    english = "Upgraded version of the red and gold French New World Trade Center."
    broken = "Suministros de J.C. Kiley"
    good = "Versión mejorada del Centro de Comercio francés rojo y dorado del Nuevo Mundo."

    _assert(accepts(english, broken, good), "a sound trusted value must be accepted")
    _assert(not accepts(english, broken, None), "a missing trusted value must be refused")
    _assert(not accepts(english, broken, "   "), "a blank trusted value must be refused")
    _assert(not accepts(english, broken, broken), "an identical value is not a repair")
    # The donor is misaligned in the same way -> refusing keeps the string flagged for review.
    _assert(not accepts(english, broken, "Otro Nombre Corto"),
            "a trusted value that is itself misaligned must be refused")

    # Placeholders are load-bearing for the game engine.
    ph_src = "%1s has destroyed %2s!"
    _assert(accepts(ph_src, "Texto equivocado", "¡%1s ha destruido %2s!"),
            "a trusted value keeping both placeholders must be accepted")
    _assert(not accepts(ph_src, "Texto equivocado", "¡%1s ha destruido algo!"),
            "a trusted value that drops a placeholder must be refused")

    # And the key written must be the one translate_strings reads.
    terms, regex, exclude = _normalize_protection(None, None, None)
    key, _tok, _phr = protect_for_cache(english, terms, regex, exclude)
    _assert(key == protected_cache_key(english), "the repaired key must match the engine's key")

    # --- Which defects may be repaired at all -----------------------------------------------
    # Structural: the slot is objectively broken, so an older sound value is strictly better.
    long_en = "Playback stopped because it is out of sync with the original game."
    _assert(_structurally_broken(long_en, long_en),
            "a long string left in the source language is structurally broken")
    _assert(_structurally_broken(english, broken), "a misaligned slot is structurally broken")
    _assert(_structurally_broken("Ranged <color=1>line unit</color>", "Unidad de línea"),
            "a lost <color> pair is structurally broken")

    # Wording: NOT structural, so --repair-from must leave it to the post-process / re-run,
    # or an older cache would undo the newer terminology.
    _assert(not _structurally_broken("Allotments are 10% cheaper.",
                                     "Las Parcelas son un 10% más baratas."),
            "wrong terminology is a wording problem, not a structural one")
    # A proper noun that legitimately survives translation is not "untranslated".
    _assert(not _structurally_broken("Yamabushi", "Yamabushi"),
            "a proper noun identical in both languages is not structurally broken")

    # --- Donor guards, each from a real entry in the July cache ------------------------------
    def donor_ok(src: str, current_value: str, good: Optional[str]) -> bool:
        if not good or not good.strip() or good.strip() == (current_value or "").strip():
            return False
        if not placeholders_compatible(src, good):
            return False
        if not _has_translatable_text(good):
            return False
        if _structurally_broken(src, good):
            return False
        return not has_english_residue(src, good, "Spanish")

    _assert(donor_ok(long_en, long_en,
                     "La reproducción se detuvo porque está desincronizada con la partida original."),
            "a sound Spanish value for an untranslated string must be accepted")
    # The July cache really does store a bare token as one "translation".
    _assert(not donor_ok("Age of Empires III: Wars of Liberty",
                         "Age of Empires III: Wars of Liberty", "__PROTECT_0__"),
            "a donor value that is only a protect token must be refused")
    _assert(not donor_ok(long_en, long_en, "Playback stopped because of the original game."),
            "a donor value still in English must be refused")
    _assert(not donor_ok("Ranged <color=1>line unit</color>", "Unidad de línea",
                         "Unidad de línea a distancia"),
            "a donor value that also lost the markup must be refused")

    print("✅ Repair-from self-test passed.")


def self_test_latam_spanish() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"LatAm Spanish self-test failed: {message}")

    def f(text: str, target: str = "Spanish") -> str:
        return normalize_latam_spanish("", text, target)

    # UI imperatives are unified on the tú form.
    _assert(f("Presione ESC para cancelar.") == "Presiona ESC para cancelar.", "Presione -> Presiona")
    _assert(f("Haga clic en una carta.") == "Haz clic en una carta.", "Haga clic -> Haz clic")
    _assert(f("Seleccione una opción.") == "Selecciona una opción.", "Seleccione -> Selecciona")
    _assert(f("Introduzca su nombre.") == "Introduce su nombre.", "Introduzca -> Introduce")

    # Opening punctuation is added only where it is missing.
    _assert(f("John Black ha colocado los explosivos!")
            == "¡John Black ha colocado los explosivos!", "missing ¡ should be added")
    _assert(f("Borrar permanentemente %s?") == "¿Borrar permanentemente %s?", "missing ¿ should be added")
    _assert(f("Primera oración. Segunda es pregunta?")
            == "Primera oración. ¿Segunda es pregunta?", "only the last sentence gets the mark")
    _assert(f("¡Ya lo tiene!") == "¡Ya lo tiene!", "an existing ¡ must not be duplicated")
    _assert(f("¿Y este también?") == "¿Y este también?", "an existing ¿ must not be duplicated")
    _assert(f("Sin puntuación final") == "Sin puntuación final", "no terminal mark, no change")
    _assert(f("%s!") == "%s!", "a placeholder-only sentence must be left alone")
    # Trailing markup must not hide the terminal '!'.
    _assert(f("Infantería a distancia! <color=1.0, 1.0, 0.0>")
            == "¡Infantería a distancia! <color=1.0, 1.0, 0.0>", "trailing markup must be ignored")

    # CRITICAL negatives: real strings that naive peninsular-form rules would have destroyed.
    for text, why in [
        ("La sed de venganza Cheyenne", "'sed' is the noun *thirst*, not a vosotros imperative"),
        ("Los dioses tienen sed de su sangre.", "same: 'sed' must never be conjugated away"),
        ("ID de Passport existente:", "'id' here is ID, not the imperative of 'ir'"),
        ("Gaucho: Até que enfim os castelhanos cansaram.", "'os' is Portuguese, deliberately"),
        ("Estos blancos móviles son un buen objetivo.", "'móviles' is the adjective *moving*"),
        ("Todos los edificios se reemplazan por caravanas móviles.", "same adjective sense"),
        ("El juego puede fallar.", "'puede' is third person, not usted"),
    ]:
        _assert(f(text) == text, f"{text!r} must stay untouched: {why}")

    # Other targets are never touched.
    _assert(normalize_latam_spanish("", "Presione ESC!", "Portuguese") == "Presione ESC!",
            "non-Spanish target must be left alone")
    # Locale-style target names must still enable the gate.
    _assert(normalize_latam_spanish("", "Presione ESC.", "es-419") == "Presiona ESC.",
            "es-419 must be recognized as Spanish")

    print("✅ LatAm Spanish self-test passed.")


def self_test_user_glossary() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"User glossary self-test failed: {message}")

    import tempfile
    content = (
        "# official terms\n"
        "\n"
        "主城 = Home City\n"
        "Sepoy=Sepoy\n"
        "malformed line without an equals sign\n"
        "  设置  =  Settings  \n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "glossary.txt"
        path.write_text(content, encoding="utf-8")
        glossary = load_user_glossary(path)
    _assert(glossary == {"主城": "Home City", "Sepoy": "Sepoy", "设置": "Settings"},
            f"unexpected parse result: {glossary}")
    _assert(load_user_glossary(Path("no_such_glossary_file.txt")) == {},
            "missing file should parse to an empty glossary")

    rules = user_glossary_rules(["你的主城遭到攻击。"], glossary)
    _assert("主城" in rules and "Home City" in rules, "expected a rule for the matching term")
    _assert("设置" not in rules, "terms absent from the batch must not be hinted")
    _assert(user_glossary_rules(["nothing relevant"], glossary) == "",
            "no rules expected for a non-matching batch")
    _assert(user_glossary_rules(["anything"], None) == "", "None glossary must yield no rules")

    fixed = apply_user_glossary_fixes("你的主城遭到攻击。", "Your 主城 is under attack.", glossary)
    _assert(fixed == "Your Home City is under attack.", f"CJK leftover fix failed: {fixed}")
    latin = {"Metropoli": "Home City"}
    fixed = apply_user_glossary_fixes("La Metropoli está en peligro", "The Metropoli is in danger", latin)
    _assert(fixed == "The Home City is in danger", f"Latin leftover fix failed: {fixed}")
    fixed = apply_user_glossary_fixes("La Metropolitana", "The Metropolitana", latin)
    _assert(fixed == "The Metropolitana", "whole-word: longer words must not be rewritten")
    fixed = apply_user_glossary_fixes("Train a Sepoy", "Train a Sepoy", glossary)
    _assert(fixed == "Train a Sepoy", "source == target entries must be a no-op")

    # --- Per-context conditions ------------------------------------------------------------
    # One English word, two Spanish terms, told apart by what surrounds it in the source.
    conditional = (
        "Skirmisher | si:<color=0.07, 0.68, 0.17> = infantería a distancia\n"
        "Skirmisher | no:<color=0.07, 0.68, 0.17> = Hostigador\n"
        "Broken | si: = ignored\n"
        "Broken | maybe:x = ignored\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conditional.txt"
        path.write_text(conditional, encoding="utf-8")
        cond = load_user_glossary(path)
    _assert([e.source_term for e in cond.entries] == ["Skirmisher", "Skirmisher"],
            f"malformed conditions must be dropped, got {[e.source_term for e in cond.entries]}")

    tagged = "<color=0.07, 0.68, 0.17>Skirmisher </color>attack increased."
    plain = "8 Skirmishers"
    tagged_rules = user_glossary_rules([tagged], cond)
    _assert("infantería a distancia" in tagged_rules and "Hostigador" not in tagged_rules,
            f"only the matching condition may fire: {tagged_rules}")
    plain_rules = user_glossary_rules([plain], cond)
    _assert("Hostigador" in plain_rules and "infantería a distancia" not in plain_rules,
            f"the excluded condition must fire without the tag: {plain_rules}")
    _assert("When the text contains" in tagged_rules,
            "a conditional rule must state its condition, since a batch can carry both contexts")

    # The deterministic layer honours the condition too.
    _assert(apply_user_glossary_fixes(tagged, "Ataque de Skirmisher aumentado.", cond)
            == "Ataque de infantería a distancia aumentado.",
            "the conditional fix must apply in its own context")
    _assert(apply_user_glossary_fixes(plain, "8 Skirmisher", cond) == "8 Hostigador",
            "the excluded-condition fix must apply outside that context")

    # An unconditional plain dict still works: every caller that predates this keeps working.
    _assert(user_glossary_rules(["Sepoy here"], {"Sepoy": "Cipayo"}).count("Cipayo") == 1,
            "a plain dict must still produce rules")

    print("✅ User glossary self-test passed.")


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


def self_test_build_cache() -> None:
    def _assert(condition: bool, message: str) -> None:
        if not condition:
            raise SystemExit(f"Build-cache self-test failed: {message}")

    def _t(text: str, loc_id: Optional[str] = None) -> TranslationTarget:
        return TranslationTarget(
            element=ET.Element("String"),
            text=text,
            symbol=None,
            skip=False,
            reason=None,
            loc_id=loc_id,
        )

    english = [
        _t("Villagers gather wood faster.", "1"),   # normal translation
        _t("Yamabushi", "2"),                       # proper noun, identical in Spanish
        _t("Convento de San Felipe Neri", "3"),     # already-Spanish name, identical
        _t("The Asian Dynasties", "4"),             # real English left untranslated
        _t("%s is not ready.", "5"),                # translation drops the placeholder
        _t("<color=1.0, 1.0, 0.0>", "6"),           # pure markup, nothing translatable
        _t("Only in the new file", "7"),            # no counterpart in the translation
    ]
    spanish = [
        _t("Los Aldeanos recolectan madera mas rapido.", "1"),
        _t("Yamabushi", "2"),
        _t("Convento de San Felipe Neri", "3"),
        _t("The Asian Dynasties", "4"),
        _t("no esta listo.", "5"),
        _t("<color=1.0, 1.0, 0.0>", "6"),
    ]

    cache, stats = build_cache_from_translation(english, spanish, target_lang="Spanish")

    def _key(text: str) -> str:
        return protected_cache_key(text)

    # A real translation is stored under exactly the key translate_strings would read.
    _assert(cache.get(_key("Villagers gather wood faster.")) == "Los Aldeanos recolectan madera mas rapido.",
            "a normal translation must be cached under its protected key")

    # THE FIX: a translation identical to the source is kept when it is a name, not English.
    _assert(cache.get(_key("Yamabushi")) == "Yamabushi",
            "an identical proper noun must be cached (it is not untranslated English)")
    _assert(cache.get(_key("Convento de San Felipe Neri")) == "Convento de San Felipe Neri",
            "an identical Spanish place name must be cached")

    # CRITICAL negative case: never cache real untranslated English as if it were a translation.
    _assert(_key("The Asian Dynasties") not in cache,
            "untranslated English must NEVER be cached")
    _assert(stats.skipped_english == 1, f"expected 1 English skip, got {stats.skipped_english}")

    # A translation that lost a %-placeholder is unsafe to reuse.
    _assert(_key("%s is not ready.") not in cache, "placeholder mismatch must not be cached")
    _assert(stats.skipped_placeholder == 1,
            f"expected 1 placeholder skip, got {stats.skipped_placeholder}")

    # Pure markup has nothing to translate: kept as-is (merge_by_locid already allows this).
    _assert(_key("<color=1.0, 1.0, 0.0>") in cache, "pure markup should be kept as-is")

    # A string with no counterpart is not invented.
    _assert(_key("Only in the new file") not in cache, "unmatched string must not be cached")
    _assert(stats.skipped_unmatched == 1,
            f"expected 1 unmatched, got {stats.skipped_unmatched}")

    _assert(stats.seeded_reused == 1, f"expected 1 reused seed, got {stats.seeded_reused}")
    _assert(stats.seeded_identical == 3, f"expected 3 identical seeds, got {stats.seeded_identical}")
    _assert(stats.written == len(cache), "written count must match the cache size")

    # Non-Spanish target: has_english_residue cannot help, so identical text with real words
    # must fall back to the conservative rule and be refused.
    de_cache, de_stats = build_cache_from_translation(
        [_t("Yamabushi", "2"), _t("<color=1.0, 1.0, 0.0>", "6")],
        [_t("Yamabushi", "2"), _t("<color=1.0, 1.0, 0.0>", "6")],
        target_lang="German",
    )
    _assert(_key("Yamabushi") not in de_cache,
            "non-Spanish target must not seed identical text that has real words")
    _assert(_key("<color=1.0, 1.0, 0.0>") in de_cache,
            "non-Spanish target should still keep pure markup")
    _assert(de_stats.skipped_english == 1,
            f"expected 1 skip for German, got {de_stats.skipped_english}")

    # An existing cache is merged into, not replaced.
    seeded, _ = build_cache_from_translation(
        english, spanish, target_lang="Spanish", existing_cache={"pre-existing": "value"})
    _assert(seeded.get("pre-existing") == "value", "existing cache entries must be preserved")

    print("✅ Build-cache self-test passed.")


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
    # Optional at the argparse level so the --self-test-* flags work without files;
    # main() re-enforces them (with the same argparse error) for normal runs.
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="Input XML file to translate.")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="Output XML file path.")

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
    api_group.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Model sampling temperature. Low values keep terminology consistent across the "
             "independent parallel batches (default: %(default)s).",
    )
    api_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampling seed, for reproducible runs. Ignored by older google-genai SDKs "
             "(default: %(default)s).",
    )

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
    protect_group.add_argument(
        "--glossary-file",
        type=Path,
        default=None,
        help=("User glossary file: one 'source term = target term' per line, '#' comments. "
              "Defaults to glossary.txt next to this script when it exists."),
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
        "--consistency-sweep",
        action="store_true",
        help="After translating, re-translate once the strings whose wording disagrees with the "
             "game's own label for the same thing. Batches run in parallel and never see each "
             "other's word choices, so this is the only place that inconsistency can be closed. "
             "Costs extra API calls; capped at 400 strings and a single pass.",
    )
    cache_group.add_argument(
        "--retranslate-stale-terms",
        action="store_true",
        help="Re-translate the cached strings whose terminology rules have changed since they "
             "were translated (tracked in the sibling <cache>.terms.json). Without it, editing "
             "the glossary leaves every already-cached string with its old wording.",
    )
    cache_group.add_argument(
        "--retry-empty-cache",
        action="store_true",
        help='Retry translations that were cached as empty ("").',
    )
    cache_group.add_argument(
        "--build-cache-from",
        type=Path,
        metavar="TRANSLATED_XML",
        help="Build a cache from an ALREADY-TRANSLATED XML instead of translating. "
             "Pair it with the matching source XML as the 'input' positional and send the "
             "result to --cache-file. Matches by _locID and never calls the API; 'output' "
             "is not required.",
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
        "--audit-spanish",
        type=Path,
        metavar="TRANSLATED_XML",
        help="Audit a Spanish translation for terminology, peninsular forms, untranslated "
             "strings and punctuation. Takes the source XML as the 'input' positional; "
             "'output' is not required and the API is never called.",
    )
    diag_group.add_argument(
        "--repair-from",
        type=Path,
        metavar="GOOD_CACHE_JSON",
        help="With --audit-spanish and --purge-audited: recover STRUCTURALLY BROKEN strings "
             "(slot holding another string, lost <color> markup, still in the source "
             "language) from a known-good cache instead of paying to re-translate them. "
             "Wording problems like terminology are never taken from it -- the audited "
             "cache is newer there. A donor value is refused unless it keeps the "
             "placeholders, has real text and is not broken itself. Never written to.",
    )
    diag_group.add_argument(
        "--purge-audited",
        type=Path,
        metavar="CACHE_JSON",
        help="With --audit-spanish: delete the flagged strings from this cache so the next run "
             "re-translates them with the current prompt and glossary.",
    )
    diag_group.add_argument(
        "--purge-only",
        metavar="LABEL[,LABEL...]",
        help="With --purge-audited: purge only the strings flagged under these audit labels "
             "(a glossary term such as 'skirmisher', or a section: keywords, untranslated, "
             "peninsular, markup, misaligned). Without it every flagged string is purged, which "
             "turns one terminology decision into a re-translation of everything the audit "
             "found. The audit prints the available labels.",
    )
    diag_group.add_argument(
        "--suggest-glossary",
        type=Path,
        metavar="TRANSLATED_XML",
        help="Mine the XML pair for terminology the glossary is missing: the game's own short "
             "label strings are its glossary, so a label translated one way while the "
             "descriptions containing it say something else is a real inconsistency. Prints "
             "ready-to-paste glossary.txt lines. Takes the source XML as 'input'; no API call.",
    )
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

    args = parser.parse_args()
    if args.suggest_glossary is not None:
        if args.input is None:
            parser.error("--suggest-glossary also requires the source XML as 'input'")
    elif args.audit_spanish is not None:
        if args.input is None:
            parser.error("--audit-spanish also requires the source XML as 'input'")
        if args.repair_from is not None and args.purge_audited is None:
            # --purge-audited names the cache that gets repaired; without it there is no target.
            parser.error("--repair-from also requires --purge-audited (the cache to repair)")
    elif args.purge_audited is not None or args.repair_from is not None:
        parser.error("--purge-audited/--repair-from only make sense together with --audit-spanish")
    elif args.build_cache_from is not None:
        # Building a cache needs the source XML (input) but produces no translated XML.
        if args.input is None:
            parser.error("--build-cache-from also requires the source XML as 'input'")
        if args.cache_file is None:
            parser.error("--build-cache-from requires --cache-file (where to write the cache)")
    elif not (args.self_test_quality_gate or args.self_test_merge) and (
            args.input is None or args.output is None):
        parser.error("the following arguments are required: input, output")
    return args


def _audit_pairs(
    source_targets: Sequence[TranslationTarget],
    translated_targets: Sequence[TranslationTarget],
) -> List[Tuple[str, str]]:
    """Align a source XML with its translation by _locID and return the (source, target) pairs."""
    report = merge_by_locid(source_targets, source_targets, translated_targets)
    return [(e.new_source, e.draft) for e in report.entries if e.draft]


def _audit_term_consistency(
    pairs: Sequence[Tuple[str, str]],
    term: str,
    canonical: str,
    variants: Sequence[str],
    source_block: Optional["re.Pattern[str]"] = None,
    source_require: Optional["re.Pattern[str]"] = None,
) -> Tuple[int, Dict[str, List[Tuple[str, str]]]]:
    """Count how a source term was rendered: canonical vs each known-wrong variant.

    ``source_require`` narrows a row to one context. The same English word can be two different
    things -- 'skirmisher' inside the green counter tag is a unit CLASS, outside it is the unit
    NAME -- and each needs its own canon, so a row must be able to say "only when this pattern
    is in the English".
    """
    # Allow the plural but nothing else, so "conscripted" (the verb) is not counted as a
    # variant of the noun "Conscript" -- that inflates the report and the purge list.
    trigger = re.compile(r"\b" + re.escape(term) + r"(?:e?s)?\b", re.IGNORECASE)
    canon_re = re.compile(r"\b" + re.escape(canonical), re.IGNORECASE)
    variant_res = [(v, re.compile(r"\b" + re.escape(v), re.IGNORECASE)) for v in variants]

    ok = 0
    found: Dict[str, List[Tuple[str, str]]] = {}
    for src, tgt in pairs:
        if not trigger.search(src):
            continue
        if source_block is not None and source_block.search(src):
            continue
        if source_require is not None and not source_require.search(src):
            continue
        matched = False
        for name, rx in variant_res:
            if rx.search(tgt):
                found.setdefault(name, []).append((src, tgt))
                matched = True
        if not matched and canon_re.search(tgt):
            ok += 1
    return ok, found


# Digits only, after dropping thousands separators so "20,000" and "20.000" compare equal.
_AUDIT_THOUSANDS_RE = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")
_AUDIT_NUM_RE = re.compile(r"\d+")


def _audit_strip_all(text: str) -> str:
    """Drop markup, placeholders and escapes so only the readable text is measured."""
    cleaned = re.sub(r"<[^>]*>|\{[^}]*\}", " ", text or "")
    cleaned = re.sub(r"%\d*\$?[sdif]|\\[ntr]", " ", cleaned)
    # "20,000" and "20.000" are the same number written two ways; normalize so a formatting
    # difference is never mistaken for a changed quantity.
    cleaned = _AUDIT_THOUSANDS_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _audit_misaligned(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Find strings whose Spanish is not a translation of their English at all.

    These come from a cache rebuilt by POSITION rather than by ``_locID``: if the two files ever
    differed by one element, the whole following block shifted, and a building's name landed in
    its description's slot. Both the XML and the cache then hold the same wrong pairing, so the
    cache cannot be used as ground truth here -- it agrees with the error.

    The rule is deliberately narrow, because a false positive sends a correct string back to the
    API for no reason: the English must READ LIKE A SENTENCE (long, ends in a period) while the
    Spanish READS LIKE A LABEL (much shorter, no final period). A looser rule based on shared
    proper nouns was tried and rejected -- neighbouring entries share names constantly, so it
    produced ten times as many hits and most were correct translations.

    The check is SYMMETRIC, and that matters: these defects come in swapped pairs, so the mirror
    case (a short English name whose slot holds a long Spanish description) has to be caught too.
    Purging only one half would re-translate one slot correctly and leave the other still wrong.
    """
    def sentence_vs_label(a: str, b: str) -> bool:
        # `a` reads like a sentence, `b` like a label sitting where a sentence belongs.
        return (len(a) >= 45 and a.endswith(".")
                and len(b) < len(a) * 0.6 and not b.endswith("."))

    def counts_differ(english: str, spanish: str) -> bool:
        """A short card label whose numbers do not match: '8 Riflemen' -> '6 Fusileros'.

        These are the same shift, but between two short labels, so the length rule cannot see
        them -- and they are the worst kind, because the card promises the player one number of
        units and the game shows another. Restricted to SHORT strings and to strings with digits
        on BOTH sides: that drops the legitimate cases where English spells a number and Spanish
        writes it as a word ('1 Settler Wagon' -> 'una Carreta'), and long narrative text where a
        date or a thousands separator differs in formatting.
        """
        if len(english.split()) > 6:
            return False
        a, b = _AUDIT_NUM_RE.findall(english), _AUDIT_NUM_RE.findall(spanish)
        return bool(a) and bool(b) and sorted(a) != sorted(b)

    found: List[Tuple[str, str]] = []
    for src, tgt in pairs:
        english = _audit_strip_all(src)
        spanish = _audit_strip_all(tgt)
        if not english or not spanish:
            continue
        if (sentence_vs_label(english, spanish) or sentence_vs_label(spanish, english)
                or counts_differ(english, spanish)):
            found.append((src, tgt))
    return found


def _structurally_broken(src: str, value: Optional[str]) -> bool:
    """True when a translation is objectively broken, not merely worded differently.

    The distinction decides what --repair-from may overwrite:

      * STRUCTURAL -- the slot holds another string's text, a ``<color>`` pair was lost, or the
        text is still in the source language. A sound older value is strictly better here, so
        reusing one is safe.
      * WORDING -- terminology, register, punctuation. The audited cache is the newer and better
        translation; the deterministic post-process fixes it for free or it gets re-translated.
        Overwriting it with an older value would be a regression, so these are NOT repaired.

    The "still in the source language" test uses the audit's own threshold of five real words:
    without it the 4463 proper nouns that legitimately survive translation ("Yamabushi") would
    all count as untranslated.
    """
    if not value or not value.strip():
        return True
    if _audit_misaligned([(src, value)]):
        return True
    if not markup_integrity_ok(src, value):
        return True
    if (src.strip() == value.strip() and _has_translatable_text(src)
            and len(re.findall(r"[A-Za-z]{3,}", src)) >= 5):
        return True
    return False


def _audit_label_map(pairs: Sequence[Tuple[str, str]]) -> Dict[str, "collections.Counter[str]"]:
    """Short English label strings (unit/tech/building names) -> how they were translated.

    These labels are the game telling us its own glossary, so they are the reference both for the
    discovery sweep and for the collision check below.
    """
    labels: Dict[str, "collections.Counter[str]"] = {}
    for src, tgt in pairs:
        label = src.strip()
        if 1 <= len(label.split()) <= 3 and re.fullmatch(r"[^\W\d_][\w \-']+", label, re.UNICODE):
            labels.setdefault(label, collections.Counter())[tgt.strip()] += 1
    return labels


def _audit_name_collisions(
    pairs: Sequence[Tuple[str, str]],
    limit: int = 25,
) -> List[Tuple[str, List[str]]]:
    """Several different English names sharing ONE Spanish name.

    The mirror image of ``_audit_discover_candidates``, and the dangerous direction: one English
    word rendered two ways is untidy, but two different units rendered with the SAME word makes
    them look like one thing on screen. That is how "Guerrillero" ended up naming both the
    `Skirmisher` and Wars of Liberty's own `Guerrilla` unit.

    Reports only; which of the two names has to change is a human decision.
    """
    groups: Dict[str, Tuple[str, set]] = {}
    for label, counter in _audit_label_map(pairs).items():
        if len(label) < 5:
            continue
        canonical = counter.most_common(1)[0][0].strip()
        if len(canonical) < 5 or "<" in canonical or "%" in canonical:
            continue
        display, names = groups.setdefault(_fold(canonical), (canonical, set()))
        names.add(label)

    results: List[Tuple[str, List[str]]] = []
    for display, names in groups.values():
        # Singular/plural and casing of one same English name are not a collision.
        if len({_fold(name).rstrip("s") for name in names}) < 2:
            continue
        results.append((display, sorted(names)))
    results.sort(key=lambda item: -len(item[1]))
    return results[:limit]


def _audit_discover_candidates(
    pairs: Sequence[Tuple[str, str]],
    limit: int = 25,
) -> List[Tuple[str, str, int, int, List[str]]]:
    """Heuristic sweep for terms that are NOT in any glossary yet.

    Short English label strings (unit/tech/building names) are treated as the game's own
    glossary; their own translation is the presumed canon, and every longer string containing
    the term is checked for it. This is noisy by nature -- casing, inflection and proper nouns
    all show up -- so callers must present it as "candidates", never as findings.
    """
    def strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text.lower())
        return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

    en_norm = [strip_accents(src) for src, _ in pairs]
    es_norm = [strip_accents(tgt) for _, tgt in pairs]

    index: Dict[str, set] = {}
    for i, text in enumerate(en_norm):
        for word in set(re.findall(r"[a-z]{4,}", text)):
            index.setdefault(word, set()).add(i)

    labels = _audit_label_map(pairs)

    def stem(word: str) -> str:
        word = strip_accents(word)
        if word.endswith("es") and len(word) > 5:
            return word[:-2]
        return word[:-1] if word.endswith("s") and len(word) > 4 else word

    results: List[Tuple[str, str, int, int, List[str]]] = []
    for label, counter in labels.items():
        if len(label) < 5:
            continue
        canonical = counter.most_common(1)[0][0]
        words = [strip_accents(w) for w in label.split() if len(w) >= 4]
        if not words:
            continue
        candidates: Optional[set] = None
        for word in words:
            hits = index.get(word, set())
            candidates = hits if candidates is None else (candidates & hits)
            if not candidates:
                break
        if not candidates or len(candidates) > 3000:
            continue
        stems = [stem(w) for w in canonical.split() if len(w) > 3]
        if not stems:
            continue
        label_norm = strip_accents(label)
        ok = 0
        diverging_sources: List[str] = []
        for i in candidates:
            if pairs[i][0].strip() == label or label_norm not in en_norm[i]:
                continue
            if all(s in es_norm[i] for s in stems):
                ok += 1
            else:
                diverging_sources.append(pairs[i][0])
        if diverging_sources and ok >= 1 and len(diverging_sources) <= 30:
            results.append((label, canonical, ok, len(diverging_sources), diverging_sources))

    results.sort(key=lambda r: -r[3])
    return results[:limit]


def run_audit_spanish_cli(
    args: argparse.Namespace,
    skip_rules: SkipRules,
    user_glossary: Dict[str, str],
) -> None:
    """Handle --audit-spanish: report translation-quality problems. No API calls.

    Works on the XML PAIR (source + translation) rather than a cache file, because most checks
    must be triggered by the English source. Without it we would flag "Tarjeta de Navidad"
    (correct for "Christmas Card") and miss that "Cofre" is right next to "Chest".
    """
    source_path: Path = args.input
    translated_path: Path = args.audit_spanish
    for path, label in ((source_path, "source"), (translated_path, "translated")):
        if not path.exists():
            raise SystemExit(f"File does not exist ({label}): {path}")

    def load(path: Path) -> List[TranslationTarget]:
        tree, _fmt = parse_strings_xml(path)
        return [t for t in iter_translatable_elements(tree.getroot(), skip_rules) if not t.skip]

    pairs = _audit_pairs(load(source_path), load(translated_path))
    print(f"\n📋 Spanish audit — {len(pairs)} aligned string pair(s)\n" + "=" * 72)

    # Flags are kept per label (a glossary term, or a section name) so --purge-only can drop
    # one terminology decision without dragging every other defect into the same re-translation.
    flagged_by_label: Dict[str, List[str]] = {}

    def flag(label: str, sources: Iterable[str]) -> None:
        flagged_by_label.setdefault(label, []).extend(sources)

    # --- 1) Glossary-driven consistency (precise) --------------------------------------
    print("\n1. TERMINOLOGY (from SPANISH_GLOSSARY + glossary.txt)")
    # (term, canonical, wrong variants, block-if-English-matches, require-English-to-match)
    known: List[Tuple[str, str, List[str],
                      Optional[re.Pattern[str]], Optional[re.Pattern[str]]]] = [
        ("hitpoints", "Puntos de Vida",
         ["Puntos de Golpe", "Puntos de Resistencia", "Puntos de Salud"], None, None),
        ("shipment", "Envío", ["Cargamento"], None, None),
        ("settler", "Colono", ["Aldeano"], re.compile(r"\bvillagers?\b", re.IGNORECASE), None),
        ("crate", "Caja", ["Cofre", "Cajones", "Cajón"],
         re.compile(r"\bchests?\b", re.IGNORECASE), None),
        ("outpost", "Avanzada", ["Puesto de Avanzada", "Puesto Avanzado"], None, None),
        ("deck", "Mazo", ["Baraja"], re.compile(r"\bsteel\s+decks?\b", re.IGNORECASE), None),
        ("potato", "papa", ["patata"], None, None),
        # Wars of Liberty names. The variants are the ones actually found in the shipped file.
        ("allotment", "Contingente", ["Parcela", "Reparto", "Asignación"],
         re.compile(r"\breallotments?\b", re.IGNORECASE), None),
        ("boneguard", "Guardia Ósea",
         ["Boneguard", "Guardia de Hueso", "Guardahueso", "Guardaósea"], None, None),
        ("conscript", "Conscripto", ["Recluta", "Conscrito"],
         re.compile(r"\bconscript\s+[A-Z]", re.UNICODE), None),
        # "Skirmisher" is two different things and needs two rows. Inside the green counter tag
        # it is the unit CLASS the engine labels "Ranged Infantry"; outside it, the unit name.
        # One row cannot express that, hence source_require.
        ("skirmisher", "infantería a distancia",
         ["Hostigador", "Escaramuzador", "Escararuzador", "Cazador"], None, GREEN_SKIRMISHER_RE),
        ("skirmisher", "Hostigador",
         # "Guerrillero" is WoL's own Guerrilla unit and "Cazador" its Cacadore: using either
         # here makes two different units look like one on screen.
         ["Guerrillero", "Escaramuzador", "Escararuzador"], GREEN_SKIRMISHER_RE, None),
        ("hand infantry", "cuerpo a cuerpo", ["Infantería de mano", "Infanteria de mano"],
         None, None),
        ("hand cavalry", "cuerpo a cuerpo", ["Caballería de mano", "Caballeria de mano"],
         None, None),
        ("warlord", "Caudillo", ["Señor de la guerra"], None, None),
        ("pasha", "Pasha", ["Pashá", "Bajá"], None, None),
        ("madrasah", "Madrasa", ["Madraza", "Madrasah"], None, None),
        ("hajduk", "Hajduk", ["Hayduk"], None, None),
        ("righteous fighter", "Guerrero Justo",
         ["Combatiente Justo", "Luchador Justiciero"], None, None),
        ("lodge", "Cabaña", ["Logia", "Pabellón"],
         re.compile(r"\bmasonic\s+lodges?\b", re.IGNORECASE), None),
        ("square", "Cuadro", ["Cuadrado", "Plaza"],
         re.compile(r"\b(?:town|city|market|village)\s+squares?\b", re.IGNORECASE), None),
        ("zapotec", "Zapoteca", ["Zapoteco"], None, None),
    ]
    # Anything the user added to glossary.txt that is not already covered above.
    covered = {term.lower() for term, _c, _v, _b, _r in known}
    for term, target_term in sorted(user_glossary.items()):
        if term.lower() not in covered:
            known.append((term, target_term, [], None, None))

    for term, canonical, variants, block, require in known:
        ok, found = _audit_term_consistency(pairs, term, canonical, variants, block, require)
        total_wrong = sum(len(v) for v in found.values())
        if not ok and not total_wrong:
            continue
        status = "✅" if not total_wrong else "⚠️ "
        detail = ", ".join(f"{name}: {len(items)}" for name, items in sorted(found.items()))
        print(f"  {status} {term:<12} {canonical!r}: {ok} ok" + (f" | {detail}" if detail else ""))
        for name, items in sorted(found.items()):
            for src, tgt in items[:2]:
                print(f"        EN: {src[:78]}")
                print(f"        ES: {tgt[:78]}")
            flag(term, (src for src, _ in items))

    # --- 2) Strings that shipped in the source language ---------------------------------
    print("\n2. UNTRANSLATED (shipped in English)")
    untranslated = [
        (src, tgt) for src, tgt in pairs
        if src.strip() == tgt.strip()
        and _has_translatable_text(src)
        and len(re.findall(r"[A-Za-z]{3,}", src)) >= 5
    ]
    print(f"  {len(untranslated)} string(s) with 5+ real words are identical to the English.")
    for src, _tgt in untranslated[:5]:
        print(f"        {src[:78]}")
    flag("untranslated", (src for src, _ in untranslated))

    # --- 3) Peninsular forms and register (report only, never rewritten) ----------------
    print("\n3. PENINSULAR FORMS (reported, never auto-rewritten — see normalize_latam_spanish)")
    peninsular = re.compile(
        r"\b(vosotros|vuestr[oa]s?|sois|ten[ée]is|pod[ée]is|hab[ée]is|est[áa]is|quer[ée]is"
        r"|deb[ée]is|patatas?|ordenador(?:es)?|zumos?)\b", re.IGNORECASE)
    hits = [(src, tgt) for src, tgt in pairs if peninsular.search(tgt)]
    print(f"  {len(hits)} string(s).")
    for src, tgt in hits[:5]:
        print(f"        {tgt[:78]}")
    flag("peninsular", (src for src, _ in hits))

    tu_re = re.compile(r"\b(debes|puedes|tienes|tus|haz|eres|quieres|selecciona|presiona)\b", re.I)
    usted_re = re.compile(r"\b(usted|debe |puede |tiene |haga|seleccione|presione|elija)\b", re.I)
    print(f"  register: {sum(1 for _s, t in pairs if tu_re.search(t))} tú / "
          f"{sum(1 for _s, t in pairs if usted_re.search(t))} usted")

    # --- 4) Missing opening punctuation --------------------------------------------------
    print("\n4. PUNCTUATION")
    punct = [(src, tgt) for src, tgt in pairs
             if normalize_latam_spanish(src, tgt, args.target) != tgt]
    print(f"  {len(punct)} string(s) would be fixed by the LatAm gate (¡/¿ and UI imperatives).")
    for _src, tgt in punct[:5]:
        print(f"        {tgt[:78]}")

    # --- 5) Markup integrity -------------------------------------------------------------
    print("\n5. MARKUP (<color> tags)")
    print("   The coloured word names the unit type a description is about, so a lost tag")
    print("   costs the player the 'strong against' cue. Cannot be repaired automatically.")
    broken_markup = [(src, tgt) for src, tgt in pairs if not markup_integrity_ok(src, tgt)]
    empty_pairs = [(s, t) for s, t in broken_markup if EMPTY_COLOR_RE.search(t)]
    print(f"  {len(broken_markup)} string(s) with broken markup "
          f"({len(empty_pairs)} of them left an empty tag pair).")
    for src, tgt in broken_markup[:5]:
        print(f"        EN: {src[:78]}")
        print(f"        ES: {tgt[:78]}")
    flag("markup", (src for src, _ in broken_markup))

    # --- 6) Unit-class keywords ------------------------------------------------------------
    print("\n6. UNIT-CLASS KEYWORDS (the coloured word names a class, not a decoration)")
    print("   Wars of Liberty codes each unit class with a colour, and the engine prints that")
    print("   same class in its effect tooltips. Naming another class there tells the player the")
    print("   wrong counter; markup_integrity_ok cannot see it because the tags are intact.")
    keyword_bad = [(src, tgt) for src, tgt in pairs
                   if not counter_keyword_ok(src, tgt, args.target)]
    keyword_free = [(src, tgt) for src, tgt in keyword_bad
                    if apply_counter_keyword_fixes(src, tgt, args.target) != tgt]
    print(f"  {len(keyword_bad)} string(s) whose coloured keyword is wrong or empty "
          f"({len(keyword_free)} of them are repaired for free on the next --cache-only export).")
    for src, tgt in keyword_bad[:5]:
        print(f"        EN: {src[:78]}")
        print(f"        ES: {tgt[:78]}")
    flag("keywords", (src for src, _ in keyword_bad))

    # --- 7) Misaligned strings -----------------------------------------------------------
    print("\n7. MISALIGNED (the Spanish belongs to a different string)")
    misaligned = _audit_misaligned(pairs)
    print(f"  {len(misaligned)} string(s) where the Spanish is not a translation of this English.")
    if misaligned:
        print("   Cause: a cache rebuilt by POSITION instead of by _locID, so a block shifted and")
        print("   names landed in description slots. The CACHE holds the same wrong pairing, so")
        print("   re-running alone will not fix it -- you must --purge-audited first, then")
        print("   re-translate. Building a cache with --build-cache-from prevents it happening again.")
    for src, tgt in misaligned[:8]:
        print(f"        EN: {_audit_strip_all(src)[:78]}")
        print(f"        ES: {_audit_strip_all(tgt)[:78]}")
    flag("misaligned", (src for src, _ in misaligned))

    # --- 8) One Spanish name serving several English names --------------------------------
    print("\n8. NAME COLLISIONS (two different units sharing one Spanish name)")
    print("   Reported, never auto-fixed: which of the two names must change is a human call.")
    collisions = _audit_name_collisions(pairs)
    print(f"  {len(collisions)} Spanish name(s) serving more than one English name.")
    for spanish, english_names in collisions[:8]:
        print(f"  · {spanish!r} <- {', '.join(repr(n) for n in english_names[:4])}")

    # --- 9) Discovery (noisy on purpose) -------------------------------------------------
    print("\n9. CANDIDATES TO REVIEW — auto-detected, NOISY")
    print("   Terms not in any glossary whose rendering varies. Expect false positives from")
    print("   casing, inflection and proper nouns. Add the real ones to glossary.txt.")
    for label, canonical, ok, diverging, _sources in _audit_discover_candidates(pairs):
        print(f"  · {label!r} → {canonical!r}: {ok} consistent, {diverging} diverging")

    flagged: List[str] = [src for sources in flagged_by_label.values() for src in sources]

    print("\n" + "=" * 72)
    print(f"Total strings flagged for re-translation: {len(set(flagged))}")
    print("  by label: " + ", ".join(
        f"{label}={len(set(sources))}"
        for label, sources in sorted(flagged_by_label.items()) if sources
    ))
    if args.purge_audited or args.repair_from:
        terms, regex, exclude = _normalize_protection(
            args.protect, compile_regex_list(args.protect_regex), args.acronym_exclude)

        def _key(text: str) -> str:
            key, _tok, _phr = protect_for_cache(text, terms, regex, exclude)
            return key

        # --purge-audited names the cache being fixed; --repair-from is only a read-only donor.
        cache_path: Path = args.purge_audited
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read cache {cache_path}: {exc}")

        current = {src: tgt for src, tgt in pairs}
        repaired = 0
        # Keys just repaired must be exempt from the purge below: they are already correct, and
        # the post-process has nothing to change in them, so the purge loop would delete them.
        repaired_keys: set[str] = set()

        # --- Repair first: purging first would delete what we are about to fix. ------------
        # Only STRUCTURALLY broken strings are repaired (see _structurally_broken): the slot
        # holds another string's text, a <color> pair was lost, or it is still in the source
        # language. Wording problems -- terminology, register, punctuation -- are NOT repaired:
        # there the audited cache is the newer and better translation, and reaching back to an
        # older one would undo recent work.
        by_kind: "collections.Counter[str]" = collections.Counter()
        if args.repair_from:
            try:
                trusted = json.loads(args.repair_from.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Could not read trusted cache {args.repair_from}: {exc}")
            for src, tgt in pairs:
                if not _structurally_broken(src, tgt):
                    continue
                good = trusted.get(_key(src))
                if not good or not good.strip() or good.strip() == (tgt or "").strip():
                    continue
                # The donor is not infallible either: the July cache stores a bare
                # "__PROTECT_0__" as one "translation". Take a value only if it fixes the defect
                # and introduces none of its own.
                if not placeholders_compatible(src, good):
                    continue
                if not _has_translatable_text(good):
                    continue
                if _structurally_broken(src, good):
                    continue
                if has_english_residue(src, good, args.target):
                    continue
                if _audit_misaligned([(src, tgt)]):
                    by_kind["misaligned"] += 1
                elif not markup_integrity_ok(src, tgt):
                    by_kind["markup"] += 1
                else:
                    by_kind["still in source language"] += 1
                cache[_key(src)] = good
                current[src] = good
                repaired_keys.add(_key(src))
                repaired += 1
            detail = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items())) or "none"
            print(f"\n🔧 Repaired {repaired} entry(ies) from {args.repair_from.name} ({detail}).")

        # --- Then purge, but only what is not already fixed for free ----------------------
        removed = free = 0
        if args.purge_audited:
            # --purge-only narrows the purge to one decision. Without it a terminology change
            # would drag every unrelated defect the audit found into the same re-translation.
            if args.purge_only:
                wanted = {name.strip().lower() for name in args.purge_only.split(",") if name.strip()}
                known_labels = {label.lower(): label for label in flagged_by_label}
                unknown = wanted - set(known_labels)
                if unknown:
                    raise SystemExit(
                        "--purge-only: no audit label named "
                        + ", ".join(sorted(unknown))
                        + ". Available: " + ", ".join(sorted(flagged_by_label))
                    )
                purge_targets = {
                    src for label, sources in flagged_by_label.items()
                    if label.lower() in wanted for src in sources
                }
                print(f"   --purge-only {args.purge_only}: {len(purge_targets)} of "
                      f"{len(set(flagged))} flagged string(s) considered.")
            else:
                purge_targets = set(flagged)
            for src in purge_targets:
                key = _key(src)
                if key not in cache or key in repaired_keys:
                    continue
                tgt = current.get(src)
                if tgt is not None:
                    fixed = apply_postprocess_overrides(src, tgt, args.target)
                    fixed = apply_counter_keyword_fixes(src, fixed, args.target)
                    fixed = apply_user_glossary_fixes(src, fixed, user_glossary)
                    fixed = normalize_latam_spanish(src, fixed, args.target)
                    if fixed != tgt:
                        # The deterministic post-process already repairs this one on export;
                        # purging it would throw away a good translation and pay for it again.
                        free += 1
                        continue
                del cache[key]
                removed += 1
            _write_cache_atomic(cache_path, cache)
            print(f"🧹 Purged {removed} entry(ies) from {cache_path.name}; "
                  f"kept {free} that the post-process already fixes for free.")

        if repaired or removed:
            print("   Next: re-export with --cache-only to apply the free fixes, then run a "
                  "normal translation to redo the purged strings.")
    elif flagged:
        print("Use --purge-audited CACHE.json to drop these from the cache and re-translate them,")
        print("and --repair-from GOOD_CACHE.json to recover structurally broken ones without the API.")


def run_suggest_glossary_cli(
    args: argparse.Namespace,
    skip_rules: SkipRules,
    user_glossary: Dict[str, str],
) -> None:
    """Mine the XML pair for terminology the glossary does not cover yet.

    The game ships its own glossary: every unit, tech and building has a short label string, and
    that label's translation is the name the player actually sees on the card. When the long
    descriptions that mention the same English word say something else, one of the two is wrong
    -- and the player sees two names for one thing.

    That is exactly how this got noticed by hand: `_locID 430207` "Ranged Infantry" is labelled
    "Infanteria a Distancia", while 417 descriptions called the same class "hostigador". This
    prints those disagreements as ready-to-paste glossary.txt lines instead of requiring someone
    to spot them.

    Report only: it never edits glossary.txt, because picking the right term needs the English
    read in full (see the Allotment note in that file).
    """
    for path, label in ((args.input, "source"), (args.suggest_glossary, "translated")):
        if not path.exists():
            raise SystemExit(f"File does not exist ({label}): {path}")

    def load(path: Path) -> List[TranslationTarget]:
        tree, _fmt = parse_strings_xml(path)
        return [t for t in iter_translatable_elements(tree.getroot(), skip_rules) if not t.skip]

    pairs = _audit_pairs(load(args.input), load(args.suggest_glossary))
    print(f"\n🔎 Glossary suggestions — {len(pairs)} aligned string pair(s)\n" + "=" * 72)

    already = {term.lower() for term in user_glossary}
    for entry in SPANISH_GLOSSARY:
        already.add(entry.name.replace("-", " "))
        already.add(entry.name.split("-")[0])   # "skirmisher-class" also covers "skirmisher"

    candidates = _audit_discover_candidates(pairs, limit=60)
    fresh = [row for row in candidates if row[0].lower() not in already]
    print(f"\n1. LABEL vs DESCRIPTIONS ({len(fresh)} term(s) not covered by any glossary yet)")
    print("   The label is the name on the card; the count is how many descriptions agree with")
    print("   it. READ THE FULL ENGLISH STRING before pasting a line: picking by frequency is")
    print("   exactly how 'Allotment = Asignacion' went wrong.\n")
    for label, canonical, ok, diverging, _sources in fresh:
        print(f"# {label!r}: the card says {canonical!r}; "
              f"{ok} description(s) agree, {diverging} disagree")
        print(f"{label} = {canonical}")

    collisions = _audit_name_collisions(pairs, limit=40)
    print(f"\n2. NAME COLLISIONS ({len(collisions)}) — one Spanish name, several English names")
    print("   These cannot be fixed by adding a line: one of the two units has to be renamed,")
    print("   and only someone who knows the mod can say which.\n")
    for spanish, english_names in collisions:
        print(f"# {spanish!r} <- {', '.join(repr(n) for n in english_names)}")

    print("\n" + "=" * 72)
    print("Nothing was written. Copy the lines you agree with into glossary.txt.")


def run_build_cache_cli(
    args: argparse.Namespace,
    skip_rules: SkipRules,
    protected_terms: Sequence[str],
    protected_regex: Sequence["re.Pattern[str]"],
    acronym_exclude: Sequence[str],
) -> None:
    """Handle --build-cache-from: pair a source XML with its translation and write the cache.

    Never touches the API. The protection settings are threaded through so the keys match the
    ones translate_strings would read for the same run.
    """
    source_path: Path = args.input
    translated_path: Path = args.build_cache_from
    cache_path: Path = args.cache_file

    for path, label in ((source_path, "source"), (translated_path, "translated")):
        if not path.exists():
            raise SystemExit(f"File does not exist ({label}): {path}")

    def _load(path: Path) -> List[TranslationTarget]:
        tree, _fmt = parse_strings_xml(path)
        return [t for t in iter_translatable_elements(tree.getroot(), skip_rules) if not t.skip]

    source_targets = _load(source_path)
    translated_targets = _load(translated_path)
    print(f"📄 {source_path.name}: {len(source_targets)} translatable string(s)")
    print(f"📄 {translated_path.name}: {len(translated_targets)} translatable string(s)")

    existing: Dict[str, str] = {}
    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
                print(f"💾 Merging into existing cache: {len(existing)} entry(ies)")
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read existing cache %s: %s", cache_path, exc)

    cache, stats = build_cache_from_translation(
        source_targets,
        translated_targets,
        protected_terms=protected_terms,
        protected_regex=protected_regex,
        acronym_exclude=acronym_exclude,
        target_lang=args.target,
        existing_cache=existing,
    )

    _write_cache_atomic(cache_path, cache)
    print(
        f"✅ Cache written to {cache_path} ({len(cache)} total entry(ies); "
        f"{stats.written} from this pair)."
    )
    print(
        f"   reused: {stats.seeded_reused} | kept identical: {stats.seeded_identical} | "
        f"skipped untranslated: {stats.skipped_english} | "
        f"skipped placeholder mismatch: {stats.skipped_placeholder} | "
        f"unmatched: {stats.skipped_unmatched}"
    )


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    if args.self_test_quality_gate:
        self_test_quality_gate()
        self_test_source_casing()
        self_test_glossary()
        self_test_user_glossary()
        self_test_latam_spanish()
        self_test_markup_integrity()
        self_test_counter_keywords()
        self_test_skirmisher_class()
        self_test_misalignment()
        self_test_repair_from()
        return
    if args.self_test_merge:
        self_test_merge()
        self_test_cache_key_parity()
        self_test_build_cache()
        return
    skip_rules = build_skip_rules(args)
    protected_terms = list(DEFAULT_PROTECTED_TERMS)
    if args.protect:
        protected_terms.extend(args.protect)
    protected_regex = compile_regex_list(args.protect_regex)
    acronym_exclude = list(DEFAULT_ACRONYM_EXCLUDE)
    if args.acronym_exclude:
        acronym_exclude.extend([t.strip() for t in args.acronym_exclude if t and t.strip()])

    if args.build_cache_from is not None:
        run_build_cache_cli(args, skip_rules, protected_terms, protected_regex, acronym_exclude)
        return

    if args.suggest_glossary is not None:
        suggest_glossary_path = args.glossary_file
        if suggest_glossary_path is None:
            default_glossary = Path(__file__).with_name("glossary.txt")
            suggest_glossary_path = default_glossary if default_glossary.exists() else None
        run_suggest_glossary_cli(args, skip_rules, load_user_glossary(suggest_glossary_path))
        return

    if args.audit_spanish is not None:
        audit_glossary_path = args.glossary_file
        if audit_glossary_path is None:
            default_glossary = Path(__file__).with_name("glossary.txt")
            audit_glossary_path = default_glossary if default_glossary.exists() else None
        run_audit_spanish_cli(args, skip_rules, load_user_glossary(audit_glossary_path))
        return

    glossary_path = args.glossary_file
    if glossary_path is None:
        default_glossary = Path(__file__).with_name("glossary.txt")
        glossary_path = default_glossary if default_glossary.exists() else None
    user_glossary = load_user_glossary(glossary_path)
    if user_glossary:
        print(f"📖 User glossary: {len(user_glossary)} term(s) from {glossary_path.name}")

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
            # Count the strings that are actually missing, by cache key. The old estimate
            # subtracted the number of cache ENTRIES (unique keys) from the number of text
            # SLOTS (which repeat the same string many times), so it always over-reported --
            # it announced 9624 missing on the WoL table when 166 were missing. That number
            # is what makes people re-translate a whole file for no reason.
            _missing = {
                key for key in (
                    protected_cache_key(
                        text,
                        protected_terms=protected_terms,
                        protected_regex=protected_regex,
                        acronym_exclude=acronym_exclude,
                    )
                    for text in translatable_texts
                )
                if not (_preview.get(key) or "").strip()
            }
            if _missing:
                print(
                    f"   ⚠️  {len(_missing)} unique string(s) are NOT in the cache and will be "
                    f"translated. Likely causes: a new mod version, an interrupted run, batch "
                    f"failures, or entries purged on purpose."
                )
            else:
                print("   ✅ Every string in this file is already in the cache (no API calls).")
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
            temperature=args.temperature,
            seed=args.seed,
            progress_callback=progress_callback,
            protected_terms=protected_terms,
            protected_regex=protected_regex,
            acronym_exclude=acronym_exclude,
            strict_no_english_residue=strict_no_english_residue,
            cache_only=args.cache_only,
            retry_empty_cache=args.retry_empty_cache,
            retranslate_stale_terms=args.retranslate_stale_terms,
            consistency_sweep=args.consistency_sweep,
            api_timeout_seconds=args.api_timeout,
            user_glossary=user_glossary or None,
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
        # These used to happen silently: the string ships in the SOURCE language.
        if stats.quality_rejected:
            summary.append(f"  ⚠️  Quality-rejected: {stats.quality_rejected} (left in source language)")
        if stats.batch_failed:
            summary.append(f"  ⚠️  Batch failures  : {stats.batch_failed} (left in source language)")
        if stats.markup_rejected:
            summary.append(f"  ⚠️  Markup broken   : {stats.markup_rejected} (<color> tags lost; see --audit-spanish)")
        if stats.counter_keyword_rejected:
            summary.append(
                f"  ⚠️  Wrong unit class: {stats.counter_keyword_rejected} "
                "(coloured keyword names the wrong class; see --audit-spanish)")
        if stats.terms_stale:
            summary.append(f"  ♻️  Stale terms     : {stats.terms_stale} (re-translated after a rule change)")
        if stats.consistency_fixed:
            summary.append(f"  🔁 Consistency fix : {stats.consistency_fixed} (aligned with the game's own labels)")
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
