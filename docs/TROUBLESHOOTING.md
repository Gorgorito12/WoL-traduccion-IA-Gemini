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

## 8) `Translator.bat` opens a console window and closes / the GUI never appears

### Cause

Usually Python is not really installed: on Windows 10/11, `python.exe` may resolve to the
Microsoft Store *alias stub* (`...\WindowsApps\python.exe`), which does nothing. Older versions of
the launcher failed silently in that case.

### Fix

`Translator.bat` is now self-sufficient: it detects the Store stub, installs Python 3.13
automatically via `winget` (per-user, no admin), installs the pip dependencies
(`google-genai`, `tqdm`, and optional `tkinterdnd2`), and only then launches the GUI. Any failure
now shows a message and pauses instead of closing silently — just re-run the `.bat` and read the
console output.

If `winget` is not available on your system, install Python manually from
<https://www.python.org/downloads/> (check **"Add python.exe to PATH"**) and re-run the `.bat`.

To see a full traceback when the GUI fails after launch, run it from a console with `python`
(not `pythonw`):

```bat
python translate_gui.py
```

Startup crashes are also shown in an error dialog even under `pythonw`.

---

## 9) The real API cost is higher than the GUI estimate

### Cause (historical)

Older versions underestimated by 4-10× because of a combination of factors:

- **Output tokens cost ~8× more than input** on gemini-2.5-flash ($2.50/M vs $0.30/M), and a
  translation's output is about as long as its input. The old estimate priced everything at the
  input rate.
- **"Thinking" was enabled by default.** gemini-2.5-flash generates hidden reasoning tokens before
  answering and bills them at the output rate. For string translation they add cost, not quality.
- **Per-batch prompt overhead**: the rules template (~400 tokens) is resent with every batch.
- **Retries** (transient errors, the Spanish residue retry) re-send whole batches.

### Current behavior

- The engine now disables thinking (`thinking_budget=0` in `translate_batch_gemini`) — same model,
  same translations, no hidden reasoning bill.
- The GUI estimate now models input + output prices separately, adds the per-batch template
  overhead, and counts CJK characters realistically (~1 token each). Expect the real bill to land
  close to the estimate; retries can still add a little.

If you need it even cheaper: `gemini-2.5-flash-lite` (~8× cheaper) or the Gemini Batch API (50%
discount) are options, but both require code changes (configurable model / async pipeline).

---

## Still blocked?

When reporting an issue, include:

- Command used
- Python version
- Error output snippet
- Whether `--cache-only` succeeds on the same files
