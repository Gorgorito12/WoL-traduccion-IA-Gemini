# Gemini XML Translator for Age of Empires III: Wars of Liberty

A **Python** script to translate **XML localization files** for the **Age of Empires III: Wars of Liberty** mod using **Google Gemini (gemini-2.5-flash)**.

It is designed for **historical video game localization (1789–1916)**, with strict rules to preserve **tokens/placeholders**, and optimized with **multithreading and caching** to minimize API usage when updating mod versions.

---

## Features

* Video game localization–focused output (clear, natural, historically appropriate).
* Strict protection of **placeholders and tokens** (`__TOK#__`, `%s`, `%1$s`, `\n`, etc.).
* **Multithreaded** translation for speed.
* **Automatic caching** to avoid retranslating unchanged strings.
* **Cache-only mode** (0 API calls).
* **Global cache support** for versioned workflows.
* Compact (default) and Detailed prompt modes.
* **Version-update merge by `_locID`**: carry an old translation onto a new English version,
  reusing unchanged strings and flagging what changed/added (CLI `--match-by-locid`, or the
  GUI **Comparar / WinMerge** tab).

---

## Requirements

* **Python 3.10+**
* **CMD or PowerShell** (Windows)
* **Google Gemini API Key** (only required for uncached strings)

---

## Installation

```bat
pip install google-genai tqdm
```

---

## Quick Start

### First translation (creates cache)

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE"
```

### Re-run with cache (no API if nothing changed)

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml"
```

---

## Recommended Workflow (Mod Updates)

### Use a global cache file (recommended)

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

### New mod version (translate only new/changed strings)

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --api-key "YOUR_API_KEY_HERE" --cache-file "wol_es.cache.json"
```

### Apply cache without API (preview / audit)

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --cache-file "wol_es.cache.json" --cache-only
```

---

## Common Options

| Option              | Description                                       |
| ------------------- | ------------------------------------------------- |
| `--api-key`         | Gemini API key (only needed for uncached strings) |
| `--cache-only`      | Use cache only (no API calls)                     |
| `--cache-file PATH` | Use a shared/global cache file                    |
| `--detailed-prompt` | Use a more explicit prompt (higher token usage)   |

---

## Version updates (new mod version)

Match the old translation to the new English by the stable `_locID` attribute, reusing what didn't
change and translating only what's new/changed:

```bat
python translate_gemini.py "stringtabley_new.xml" "stringtabley_es_latam_new.xml" --match-by-locid --prev-source "stringtabley_old.xml" --prev-translation "stringtabley_es_latam_old.xml" --cache-file "wol_es.cache.json" --api-key "YOUR_API_KEY_HERE"
```

In the **GUI**, use the **Comparar / WinMerge** tab to compare the files, see what's missing, edit
translations by hand, auto-translate the gaps, and export the adapted XML. See `docs/USAGE.md` for
details (and `--cache-only`/`--report` for a zero-API preview).

---

## Documentation

**Full documentation and advanced workflows**
See the `/docs` folder:

* `docs/USAGE.md` – all parameters and examples
* `docs/CACHE_WORKFLOW.md` – cache strategy, rebuilds, version updates
* `docs/TROUBLESHOOTING.md` – common issues and fixes

---

## Security

**Never commit your API key to GitHub.**
Use environment variables or local config files for automation.

---

## Disclaimer

Review Google Gemini’s terms and policies regarding API usage, rate limits, and commercial use.

---
