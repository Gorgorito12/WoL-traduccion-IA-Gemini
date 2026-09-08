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

You can also run the built-in quality tests (quality gate, casing, glossaries):

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

- In the GUI, use **Avanzado ▾ → Palabras protegidas** (comma-separated, case-sensitive,
  whole-word) to keep game terms like unit names untranslated.
- Note: adding new protected words changes the cache key of every string containing them, so
  those strings re-translate once on the next run (by design — their old translations may have
  translated the term).
- If a **glossary** term (`glossary.txt`) is not being enforced in some strings, they were likely
  translated (and cached) before you added the entry — the glossary does not change cache keys.
  Remove those entries from the cache file (or re-translate those strings) to refresh them.

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

---

## 10) Strings shipped in English instead of Spanish

**Symptom.** Some strings appear in English in the game even though the run reported no errors.

**Cause.** The English-residue quality gate rejected the translation, so the engine kept the
source string rather than emit something it judged broken. Historically the gate's word list was
too broad: any Spanish sentence mentioning "Age of Empires III" tripped on its `of`, and words
that are ordinary Spanish ("original", "versión") counted as English. 144 strings shipped in
English for that reason alone.

**Fix.** Both the protected-terms list and the stopword list were narrowed. If it still happens:

* The run summary now prints `Quality-rejected` and `Batch failures` counters — these are the
  strings that stayed in the source language. They used to be invisible.
* Find them with `--audit-spanish` (section 2, "UNTRANSLATED").
* Add any product/proper name the gate keeps tripping on via `--protect "Some Name"`, then
  `--purge-audited` the cache and re-run. Protecting a term changes its cache key, so those
  strings are translated once more.

---

## 11) The same English term is translated inconsistently

**Symptom.** One term appears as two or three different Spanish words, sometimes on the same
screen (`Deck` → "Mazo" and "baraja"; `Allotment` → "Parcela", "Reparto" and "Asignación").

**Cause.** Batches are translated in parallel and independently, so nothing made two batches
agree; the model also sampled at temperature 1.0 by default.

**Fix.**

1. `--temperature 0.2` (now the default) sharply reduces the variance.
2. Run `--audit-spanish` to list every term whose rendering varies.
3. Add the term to `glossary.txt` (`Allotment = Asignación`). The rule is injected into the
   prompt only for batches containing the term, and a term the model leaves untranslated is
   replaced deterministically.
4. `--purge-audited CACHE.json` to drop the affected strings, then re-run. A glossary entry
   cannot rewrite a wrong-but-translated synonym, so already-cached wording needs a re-translation.

---

## 12) The colour disappears from unit descriptions

**Symptom.** A unit description that is coloured in English comes out plain in Spanish, or shows
an empty gap where the coloured word should be.

**Cause.** The coloured word is the counter keyword — it tells the player what the unit is strong
against. Spanish reorders adjective and noun ("Nepalese skirmisher" → "Hostigador nepalí"), and
the model moves the word out of its `<color=...>` tag, leaving the pair empty or dropping it.

**Fix.** The engine now detects this (`markup_integrity_ok`) and retries with a stricter prompt,
and sends a preventive markup rule for any batch containing tags. For strings already translated:

1. `--audit-spanish` lists them in section 5, separating the empty-tag cases.
2. `--purge-audited CACHE.json` drops them, then re-run the translation.

It cannot be repaired automatically: once the sentence is reordered there is no way to know where
the word went, so the string has to be translated again.

---

## 13) The same unit has several different names

**Symptom.** A card, a unit and an upgrade that are the same thing in English read as three
different things in Spanish.

**Cause.** Independent parallel batches plus a mod-specific term nobody pinned down. `Allotment`
shipped as *Parcela*, *Reparto* and *Asignación* at once; `Boneguard` had five names.

**Fix.** Add the term to `glossary.txt`. **Read the full English string first** — `Allotment`
looked like a plot of land, but the English says it "musters" and "contains units", so it is a
block of troops (*Contingente*). Then `--audit-spanish` to find the affected strings and
`--purge-audited` to re-translate them.

---

## 14) A string shows a completely different text

**Symptom.** A building's tooltip shows its *name* where the description should be (or the
reverse). The Spanish is well written — it just belongs to a different string.

**Cause.** A cache rebuilt **by position** instead of by `_locID`. If the English and translated
XML ever differed by one element, every entry after that point shifted by one, and the wrong
translation was stored under each key. The XML and the cache then hold the same wrong pairing.

**Why re-running does not fix it.** The cache is poisoned, so the engine finds a "hit" and writes
the same wrong text again. It also means the cache cannot be used to detect the problem: it agrees
with the error.

**Fix.**

1. `--audit-spanish` reports them in section 6. It checks both directions, because these come in
   swapped pairs and fixing only one half leaves the other wrong.
2. `--repair-from GOOD_CACHE.json` recovers them from an older cache that predates the damage —
   free, and better than re-translating, because the old text was already correct. It covers all
   three *structural* defects (wrong slot, lost markup, still in the source language) but never
   terminology, so it cannot undo newer wording. It refuses a donor value that drops a
   placeholder, has no real text, or is broken itself.
3. `--purge-audited CACHE.json` drops whatever could not be repaired, then re-run the translation.
4. Build future caches with `--build-cache-from`, which pairs by `_locID` and cannot shift.

**Watch for the worst variant: shifted numbers.** The same damage between two short labels shows
up as `8 Riflemen` → `6 Fusileros` — the card promises one number of units and the game shows
another. The audit checks for this separately, since the length rule cannot see it.
