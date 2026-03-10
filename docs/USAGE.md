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

## Full options list (most relevant)

### API & performance

* `--api-key "KEY"`: required only when there are uncached strings to translate.
* `--max-workers N`: number of concurrent workers (default: 8).
* `--max-budget-bytes N`: batch size budget for requests (default: 4500).

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

### Diagnostics & tests

* `--diagnostic`: print encoding/BOM diagnostics after each write.
* `--self-test-quality-gate`: run quick quality gate + casing tests and exit.

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
