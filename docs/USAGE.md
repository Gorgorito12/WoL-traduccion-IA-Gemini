# Usage Guide — Gemini XML Translator (WoL)

This document contains the full CLI usage for `translate_gemini.py`, including common workflows and advanced flags.

---

## Basic syntax

```bat
python translate_gemini.py "INPUT.xml" "OUTPUT.xml" [options]
```

* `INPUT.xml`: source XML (usually English).
* `OUTPUT.xml`: translated XML output (e.g., Spanish LATAM).

---

## Quick start (most common)

### First translation (creates/updates cache)

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE"
```

### Re-run (uses cache; no API if everything is cached)

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml"
```

---

## Other language pairs (Chinese→English, Italian→English, Japanese→Spanish, …)

The engine is language-agnostic: pass the pair with `--source` / `--target` (defaults:
English → Latin American Spanish). The Spanish-only quality gates (glossary, English-residue
detection) simply switch off for other targets.

```bat
python translate_gemini.py "stringtabley_zh.xml" "stringtabley_en.xml" --api-key "KEY" --source "Chinese (Simplified)" --target "English" --cache-file "wol_zh-en.cache.json"
```

> Use **one cache file per language pair** — the cache key is the source text only, so mixing
> pairs in one file returns wrong-language translations. In the GUI, both languages are dropdowns
> ("Translate from / to"), and when the fields are left empty the automatic output/cache names are
> partitioned per pair (`X_translated_zh-en.xml`, `X_translated_zh-en.xml.cache.json`); the default
> English→Latin American Spanish pair keeps the legacy names (`X_translated.xml`).
> The GUI also sanity-checks the pair: if the file doesn't look like the selected source language
> (e.g. an English file with German→Japanese selected), it warns before translating and offers to
> switch the source in one click.

---

## Protecting game terms (keep original English words untranslated)

To stop Gemini from translating specific game terms (e.g. Asian Dynasties unit names), protect
them — they are masked as `__PROTECT_x__` before the API call and restored verbatim afterwards:

```bat
python translate_gemini.py "input.xml" "output.xml" --api-key "KEY" --protect-regex "\bSepoy\b" --protect-regex "\bAshigaru\b"
```

(`--protect "Sepoy"` also works but matches substrings case-sensitively; `--protect-regex` with
`\b` is safer for single words.)

In the GUI: **Avanzado ▾ → Palabras protegidas** — comma-separated (e.g. `Sepoy, Ashigaru,
Flying Crow`), case-sensitive, whole-word, remembered between sessions, and applied consistently
to translation, cache keys and the cost estimate. Adding new words re-translates the strings that
contain them once (their cache key changes).

---

## Forcing official game terminology (user glossary)

When the model should always use an exact term in the **target** language — e.g. translating
Chinese→English and wanting the official Age of Empires terms ("Home City", "Settings"), not
invented variants — use the user glossary. Create `glossary.txt` next to the scripts (the GUI's
**Avanzado ▾ → Glosario…** button creates it with a template):

```text
# source term = target term
主城 = Home City
设置 = Settings
```

Works for any language pair (entries only fire when the source term appears in a string). Two
layers: the prompt instructs Gemini per batch, and a deterministic pass fixes source terms left
untranslated in the output. CLI: picked up automatically, or pass `--glossary-file "my.txt"`.

Note: the glossary does NOT change cache keys — strings translated before you added a term keep
their old wording until re-translated.

---

## Recommended: use a global cache file (version-friendly)

If you translate multiple mod versions, always reuse the same cache:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

Then for a new version:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

---

## Cache-only mode (0 API calls)

Use cached translations only (no API calls). Great for previewing how much coverage you already have:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --cache-file "wol_es.cache.json" --cache-only
```

---

## Rebuild cache from an existing translated XML (0 API)

If you already have:

* `stringtabley.xml` (English)
* `stringtabley_es_latam.xml` (Spanish)

You can rebuild/populate a cache file without spending API:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --cache-file "wol_es.cache.json" --cache-only
```

> Important: rebuilding requires BOTH input (source) and output (translated). The cache maps `source_string -> translated_string`.

---

## Version update: merge by `_locID` (recommended for new mod versions)

When a new version of the mod ships, match the old translation to the new English by the stable
`_locID` attribute (instead of fragile line-by-line position). This reuses unchanged strings,
flags the ones whose English **changed**, and lists the brand-new ones — so you only translate
what actually moved.

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" ^
  --match-by-locid ^
  --prev-source "stringtabley_old.xml" ^
  --prev-translation "stringtabley_es_latam_old.xml" ^
  --cache-file "wol_es.cache.json" ^
  --api-key "YOUR_API_KEY_HERE" ^
  --report "update_report.json"
```

* `--prev-source`: the **old English** XML the old translation was made from (needed to detect
  what changed).
* `--prev-translation`: the **old translated** XML to carry over.
* `--report PATH`: writes a JSON summary (counts + every string that still needs attention).
* Add `--cache-only` to preview with **zero API calls** (reused strings get applied; new/changed
  ones stay in English until you run with a key).

Safety rules baked in: a reused translation is only trusted when the English is unchanged **and**
its `%s`/`%1$s` placeholders still line up; if the old "translation" is actually still English, or
the English changed, the string is sent for (re)translation instead of carried over blindly.

> Tip: the bulk of your reuse comes from the **cache file**, not from `--prev-translation` if that
> file isn't really translated. Always pass `--cache-file` pointing at your populated cache.

### Self-tests for the merge logic

```bat
python translate_gemini.py --self-test-merge
```

---

## Full options list (most relevant)

### API & performance

* `--api-key "KEY"`: required only when there are uncached strings to translate.
* `--api-timeout N`: per-request API timeout in seconds (default: 120).
* `--max-workers N`: number of concurrent workers (default: 8).
* `--max-budget-bytes N`: batch size budget for requests (default: 4500).

### Languages

* `--source "NAME"` / `--target "NAME"`: the language pair (defaults: English → Latin American
  Spanish). See "Other language pairs" above.

### Prompt modes

* `--compact-prompt`: compact prompt (default).
* `--detailed-prompt`: more verbose prompt (higher token usage).

### Cache controls

* `--cache-file "PATH"`: use a specific cache file instead of the default `<output>.cache.json`.
* `--cache-only`: never call the API; only apply cache.
* `--retry-empty-cache`: retry entries cached as empty (`""`). Use this only if you want to force retries.

### Quality rules (Spanish-target specific)

* `--strict-no-english-residue`: force strict English residue detection for Spanish targets.
* `--no-strict-no-english-residue`: disable English residue detection even for Spanish targets.

### Skip rules

* `--skip-symbol "NAME"`: skip exact symbol names (repeatable).
* `--skip-symbol-contains "TEXT"`: skip symbol names containing substring (repeatable).
* `--skip-symbol-regex "REGEX"`: skip symbol names matching regex (repeatable).
* `--skip-text-regex "REGEX"`: skip element text matching regex (repeatable).
* `--no-path-heuristic`: disable path-like text auto detection.

### Protection rules

* `--protect "PHRASE"`: protect exact phrases from translation (repeatable).
* `--protect-regex "REGEX"`: protect regex matches from translation (repeatable).
* `--acronym-exclude "TOKEN"`: allow specific ALL-CAPS tokens to translate (repeatable).
* `--glossary-file "PATH"`: user glossary forcing official terminology (defaults to
  `glossary.txt` next to the script when it exists). See "Forcing official game terminology".

### Diagnostics & tests

* `--diagnostic`: print encoding/BOM diagnostics after each write.
* `--verbose`: debug-level logging.
* `--self-test-quality-gate`: run the quality-gate, source-casing, glossary and user-glossary
  self-tests and exit (no files needed).
* `--self-test-merge`: run the `_locID`-merge and cache-key-parity self-tests and exit.

---

## Typical commands (copy/paste)

### Spanish LATAM, global cache, compact prompt (default)

```bat
python translate_gemini.py "unithelpstringsy.xml" "unithelpstringsy_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

### Spanish LATAM, detailed prompt

```bat
python translate_gemini.py "unithelpstringsy.xml" "unithelpstringsy_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json" --detailed-prompt
```

### Preview what the cache covers (no API)

```bat
python translate_gemini.py "unithelpstringsy_new.xml" "unithelpstringsy_es_latam_new.xml" --cache-file "wol_es.cache.json" --cache-only
```

### Force retry of empty-cache entries (optional)

```bat
python translate_gemini.py "unithelpstringsy_new.xml" "unithelpstringsy_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json" --retry-empty-cache
```

---

## Notes on outputs

* The script writes the output XML and preserves formatting/encoding as best as possible.
* The cache is JSON and is meant to be reused between versions to reduce API usage.

---

## If something looks wrong

* Run self tests:

```bat
python translate_gemini.py --self-test-quality-gate
```

* Upgrade dependencies:

```bat
pip install --upgrade google-genai tqdm
```

* If you get quota/rate limit errors, reduce workers:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --max-workers 3
```

---

## Related docs

* `docs/CACHE_WORKFLOW.md` — cache strategy, rebuilds, and version updates.
* `docs/TROUBLESHOOTING.md` — common issues and practical fixes.
