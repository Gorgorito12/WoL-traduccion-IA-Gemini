# Cache Workflow — Minimal API Usage Across WoL Versions

This document explains the recommended caching strategy to minimize Gemini API usage when new mod versions introduce only a small number of string changes.

---

## 1) What the cache is (conceptually)

The cache is a JSON mapping:

- **key**: source string (after internal protection/tokenization)
- **value**: translated string

If a string is identical in the new version, the script reuses its cached translation and **does not call the API**.

If a string changed (even one character), it is treated as a **new key** and translated once.

---

## 2) The #1 rule: use a single “global cache” per language pair

If you let the script use its default cache naming (`<output>.cache.json`), changing output names per version creates new caches and wastes API.

### Recommended setup

Keep one cache per **language pair** — the cache key is the source text only (no language
dimension), so a cache reused across pairs would silently return wrong-language translations.
For example:

- `wol_es.cache.json` (English→Spanish)
- `wol_pt.cache.json` (English→Portuguese)
- `wol_zh-en.cache.json` (Chinese→English)

Use it every time via `--cache-file`.

> GUI note: when the Output/Cache fields are left empty, the GUI already partitions the automatic
> names per pair (`X_translated_zh-en.xml` + `.cache.json`; the default English→Latin American
> Spanish pair keeps the legacy `X_translated.xml` names). The explicit `--cache-file` global
> cache remains the recommended pattern for version updates.

---

## 3) The standard workflow for mod updates

### Scenario

You have:

- Old version file: `stringtabley_old.xml`
- New version file: `stringtabley_new.xml`
- Existing cache: `wol_es.cache.json`

### Step A — Preview coverage (0 API calls)

This applies cache to the new file without spending API:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --cache-file "wol_es.cache.json" --cache-only
```

Expected summary behavior:

- `Used from cache`: very high
- `Translated with API`: 0
- Some strings may remain in English if they are new (uncached)

### Step B — Translate only new/changed strings (minimal API)

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

Result:

- API is used only for strings not found in the cache
- Cache is updated for the next versions

---

## 4) Rebuilding cache when you lost it (0 API)

If you have:

- `stringtabley.xml` (English)
- `stringtabley_es_latam.xml` (Spanish translated output)

You can recreate/populate a cache file with:

```bat
python translate_gemini.py "stringtabley.xml" --build-cache-from "stringtabley_es_latam.xml" --cache-file "wol_es.cache.json"
```

Important notes:

- You must provide **both** files: the cache key is the *source* string, so the translated file
  alone cannot tell you which English string a translation came from.
- Pairing is by the stable `_locID` attribute, so reordering and count mismatches are fine.
- Pass the same `--protect` / `--protect-regex` / `--target` you translate with, or the keys
  will not match the ones the engine reads.
- Strings whose translation equals the source are kept when they are proper nouns or pure
  markup (*Yamabushi*, `<color=…>`) and refused when they are genuinely untranslated English
  (*The Asian Dynasties*) — the latter would poison the cache. On the WoL Spanish table this
  keeps roughly 7% of the corpus that would otherwise be re-sent to Gemini.
- An existing cache at `--cache-file` is merged into, not overwritten.

> **In the GUI:** the **Generar… / Generate…** button next to the *Cache file* field on the
> Translator tab, or **“Generar caché (sin API)…”** on the Compare tab. Both run the same engine
> function as the CLI.
>
> The legacy `--cache-only` rebuild (passing the translated XML as the `output` positional)
> pairs **by position** and discards everything on a count mismatch. Prefer the command above.

---

## 5) “Empty cache” entries (`""`) and why they exist

Sometimes a cache entry can be stored as empty (`""`). This usually means one of these happened previously:

- A batch failed and the script intentionally left entries retryable.
- A strict quality rule rejected a candidate translation.

### Default behavior (recommended)

By default, empty entries are **not retranslated automatically**, to prevent repeated API spending.
They are counted as “skipped due to empty cache”.

### If you want to force retries

Use:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json" --retry-empty-cache
```

Use this only when you intentionally want to retry those previously rejected items.

---

## 6) Best practices for WoL localization

### Use stable output names OR global cache

- Either keep the same output name always (less flexible), OR
- Use `--cache-file` and name outputs per version freely.

### Commit or backup cache

If you want to avoid losing progress:

- Keep the cache file backed up locally, or
- Store it in a private place (avoid public repo if it contains sensitive content, though generally it’s just strings).

### Reduce rate limit issues

If your API hits quota/429, reduce concurrency:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json" --max-workers 3
```

---

## 7) Checklist for every new mod version

1. Run cache-only preview (0 API)
2. Run translation with API (minimal)
3. Keep using the same cache file for future versions

---

## Example: one-liner “daily driver” command

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```
