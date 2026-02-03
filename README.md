---

# Gemini XML Translator for Age of Empires III: Wars of Liberty

A **Python** script that translates **XML** files used by the **Age of Empires III: Wars of Liberty** mod using **Google Gemini (gemini-2.5-flash)**. It’s designed for **historical video game localization (1789–1916)** with strict rules to keep **tokens/placeholders** intact, and it’s optimized with **multithreading** and **automatic caching** to avoid retranslating strings you already processed.

A **compact prompt is enabled by default** to reduce token usage without losing critical rules. If you need more guidance/context, enable the detailed prompt with `--detailed-prompt`.

---

## Features

* Video game localization-focused output (concise, natural tone).
* Strict protection of **tokens/placeholders** (e.g., `__TOK#__`, `%s`, `%1$s`, `\n`, `\t`, etc.).
* **Multithreading** for faster processing.
* **Automatic cache** to reuse translations and save API cost.
* **Cache-only mode** (no API calls).
* **Global cache file support** via `--cache-file` (recommended for versioned mod updates).
* Optional **retry of empty-cache entries** with `--retry-empty-cache`.
* Quick **self-tests** for quality gate / casing rules.
* **Compact** (default) and **Detailed** (optional) prompt modes.

---

## Requirements

* **Python 3.10+**
* **Command Prompt (CMD) or PowerShell** (Windows)
* A valid **Google Gemini API Key** (only required when translating uncached strings)

---

## Install & Run (CMD / PowerShell)

### 1) Verify Python

```bat
python --version
```

If the command is not found, install Python from:
[https://www.python.org](https://www.python.org)
and make sure to check **“Add Python to PATH”** during installation.

---

### 2) (Optional) Create and activate a virtual environment

```bat
python -m venv venv
venv\Scripts\activate
```

---

### 3) Install dependencies

```bat
pip install google-genai tqdm
```

---

### 4) Go to the script folder

Example:

```bat
cd C:\Users\User\Documents\translator
```

Make sure `translate_gemini.py` is located in this folder.

---

## Basic Usage

### Translate (uses API only if needed)

```bat
python translate_gemini.py "unithelpstringsy.xml" "unithelpstringsy_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --source "English" --target "Latin American Spanish"
```

* **Compact mode** is enabled by default.
* To use a more detailed prompt (higher token usage), add `--detailed-prompt`.
* If **everything is already cached**, the script will reuse cache and you can omit `--api-key`.

### Detailed prompt example

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --source "English" --target "Latin American Spanish" --detailed-prompt
```

---

## Cache Workflow (Recommended for Mod Versions)

### Why use a global cache file?

If you name outputs per version (e.g., `..._1.2.0b.xml`, `..._1.2.0c.xml`), default per-output caching creates multiple caches and wastes API.
Using `--cache-file` lets you keep **one cache per language** and translate **only new/changed strings** across versions.

### Recommended: one cache per language

Example:

* `wol_es.cache.json` (Spanish LATAM)
* `wol_pt.cache.json` (Portuguese)
* `wol_de.cache.json` (German)

---

## Commands You’ll Actually Use

### A) First time on a file (creates/updates cache)

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

### B) New mod version with minimal changes (translate only what’s new)

1. **Preview/Apply cache without spending API (0 calls)**

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --cache-file "wol_es.cache.json" --cache-only
```

2. **Translate only uncached strings (minimal API use)**

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

### C) Rebuild cache from an existing translated XML (0 API)

If you already have:

* `stringtabley.xml` (English)
* `stringtabley_es_latam.xml` (Spanish)

You can regenerate the cache without calling the API:

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --cache-only --cache-file "wol_es.cache.json"
```

> Note: cache rebuild requires BOTH input (source) and output (translated) files so the script can map `source_string -> translated_string`.

### D) Retry items cached as empty (`""`) (optional)

Sometimes a string may be cached as empty due to quality gating or a failed batch. By default, those entries are **skipped** to prevent repeated API usage.
If you want to force retrying them:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json" --retry-empty-cache
```

---

## Parameters

| Parameter                        | Description                                                     |
| -------------------------------- | --------------------------------------------------------------- |
| `input.xml`                      | Source XML file (e.g., English)                                 |
| `output.xml`                     | Translated output XML file                                      |
| `--api-key`                      | Google Gemini API key (**required only for uncached strings**)  |
| `--source`                       | Source language (default: `English`)                            |
| `--target`                       | Target language (default: `Latin American Spanish`)             |
| `--max-workers`                  | Number of concurrent worker threads                             |
| `--max-budget-bytes`             | Batch size budget (bytes)                                       |
| `--compact-prompt`               | Forces compact prompt (default)                                 |
| `--detailed-prompt`              | Uses detailed prompt (more context / more tokens)               |
| `--cache-only`                   | Use cache only; **no API calls**                                |
| `--cache-file PATH`              | Use a specific cache JSON file instead of `<output>.cache.json` |
| `--retry-empty-cache`            | Retry strings cached as empty (`""`)                            |
| `--strict-no-english-residue`    | Enable strict English residue detection for Spanish targets     |
| `--no-strict-no-english-residue` | Disable English residue detection (even for Spanish targets)    |
| `--self-test-quality-gate`       | Run quick self-tests and exit                                   |
| `--diagnostic`                   | Print encoding/BOM diagnostics after each write                 |

---

## Cache Notes

* Default cache location (if `--cache-file` is not provided):
  `"<output>.cache.json"`
* Using `--cache-file` is recommended for versioned workflows to avoid creating a new cache per output file name.
* `--cache-only` guarantees **0 API calls** and can be used to:

  * apply cached translations
  * rebuild the cache from an existing translated output

---

## Quick Troubleshooting

### `ModuleNotFoundError: google.genai` / `No module named ...`

Upgrade/reinstall dependencies:

```bat
pip install --upgrade google-genai tqdm
```

### API errors (401/403/429) or “quota/rate limit”

Confirm your API key is valid, the project has access enabled, and check quotas/limits.

### “Why does it still want to translate some strings?”

Some entries may be cached as empty (`""`) due to a previous quality-gate rejection or a failed batch.
By default, those are skipped (to avoid repeated API usage). Use `--retry-empty-cache` only if you want to force retries.

### Encoding/XML issues

If the game is sensitive to XML encoding, keep the expected format (for example, UTF-16 LE if required) and validate the output XML before testing in-game.

---

## Security (API Key)

Do **not** commit your API key to GitHub or share it publicly.
Use environment variables if you plan to automate runs.

---

## Disclaimer

Review Google Gemini’s policies/terms for API usage (including commercial use and rate limits).

---
