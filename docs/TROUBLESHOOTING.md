# Troubleshooting — Gemini XML Translator (WoL)

Use this guide for the most common issues when translating XML files.

---

## 1) `API key required for uncached translations`

### Cause

The cache does not contain every string and the script needs to call Gemini.

### Fix

Provide your API key:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE"
```

Or run cache-only mode if you intentionally want zero API calls:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --cache-only
```

---

## 2) Too many quota / 429 / transient API failures

### Cause

Concurrency is too high for your current quota or burst limits.

### Fix

Lower workers and rerun:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --max-workers 3
```

If needed, reduce batch size too:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --max-workers 3 --max-budget-bytes 2500
```

---

## 3) Output still has English fragments in Spanish target

### Cause

Some strings may pass through unchanged (e.g., cached data, partial failures, or weak generations).

### Fix

Use strict residue detection and retry empty cache entries:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --strict-no-english-residue --retry-empty-cache
```

You can also run the built-in quality test:

```bat
python translate_gemini.py --self-test-quality-gate
```

---

## 4) Cache seems ignored after renaming output files

### Cause

By default, cache filename depends on output filename (`<output>.cache.json`).

### Fix

Pin a shared cache explicitly with `--cache-file`:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

---

## 5) Protected tokens/placeholders were altered

### Cause

Custom text patterns or manual edits may have changed protected placeholders.

### Fix

- Re-run using the original source XML.
- Avoid editing placeholders like `%s`, `%1$s`, `\n`, or `__PROTECT_x__` in translated output.
- Add additional protection rules when needed:

```bat
python translate_gemini.py "input.xml" "output.xml" --api-key "YOUR_API_KEY_HERE" --protect "MyExactToken" --protect-regex "HP_[0-9]+"
```

---

## 6) XML output encoding/BOM issues in game

### Cause

Some tools rewrite encoding or line endings unexpectedly.

### Fix

Use diagnostics mode to inspect write behavior:

```bat
python translate_gemini.py "input.xml" "output.xml" --api-key "YOUR_API_KEY_HERE" --diagnostic
```

Also avoid opening/saving the output in editors that auto-convert encodings.

---

## 7) I only want to validate cache coverage, no API usage

### Fix

Run cache-only mode:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --cache-file "wol_es.cache.json" --cache-only
```

Review the summary counters to see how many strings were reused vs. left pending.

---

## Still blocked?

When reporting an issue, include:

- Command used
- Python version
- Error output snippet
- Whether `--cache-only` succeeds on the same files
