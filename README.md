
---

# Gemini XML Translator for Age of Empires III: Wars of Liberty

A **Python** script that translates **XML** files used by the **Age of Empires III: Wars of Liberty** mod using **Google Gemini (gemini-2.5-flash)**. It’s designed for **historical video game localization (1789–1916)** with strict rules to keep **tokens/placeholders** intact, and it’s optimized with **multithreading** and **automatic caching** to avoid retranslating strings you already processed.

A **compact prompt is enabled by default** to reduce token usage without losing critical rules. If you need more guidance/context, enable the detailed prompt with `--detailed-prompt`.

---

## Features

- Video game localization-focused output (concise, natural tone).
- Strict protection of **tokens/placeholders** (e.g., `__TOK#__`, `%s`, `%1$s`, `\n`, `\t`, etc.).
- **Multithreading** for faster processing.
- **Automatic cache**: detects previously translated strings and skips them to save time and API cost.
- Target-aware glossary: defaults to **“Home City” → “Metrópoli”** when translating to Spanish, and can be extended/overridden for any language with CLI options.
- **Compact** (default) and **Detailed** (optional) prompt modes.

---

## Requirements

- **Python 3.10+**
- **Command Prompt (CMD) or PowerShell** (Windows)
- A valid **Google Gemini API Key**

---

## Install & Run (CMD / PowerShell)

### 1) Verify Python

```bat
python --version
````

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

### 5) Run the script

Example:

```bat
python translate_gemini.py "unithelpstringsy.xml" "unithelpstringsy_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --source "English" --target "Latin American Spanish"
```

* **Compact mode** is enabled by default.
* To use a more detailed prompt (higher token usage), add `--detailed-prompt`.
* If you want to explicitly force compact mode, use `--compact-prompt`.

Example (detailed prompt):

```bat
python translate_gemini.py "stringtabley.xml" "stringtabley_es_latam.xml" --api-key "YOUR_API_KEY_HERE" --source "English" --target "Latin American Spanish" --detailed-prompt
```

---

## Parameters

| Parameter           | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `input.xml`         | Source XML file (e.g., English)                       |
| `output.xml`        | Translated output XML file                            |
| `--api-key`         | **Required.** Google Gemini API key                   |
| `--source`          | Source language (default: `English`)                  |
| `--target`          | Target language (default: `Latin American Spanish`)   |
| `--compact-prompt`  | Forces the compact prompt (default)                   |
| `--detailed-prompt` | Uses the detailed prompt (more context / more tokens) |

---

## Cache Notes

The script generates a cache to reuse translations and **only translate new/changed strings** when it detects updates.
This reduces both runtime and cost when rerunning the script after new mod patches.

---

## Glossary (optional)

- Default: When the target language is Spanish (including “Latin American Spanish”), the term **“Home City”** is forced to **“Metrópoli”** to keep the game’s terminology consistent.
- Custom glossary entries:
  - Inline: `--glossary "Home City=Metrópoli"`
  - From JSON file (object mapping source → translation): `--glossary-json glossary.json`

Combine multiple `--glossary` flags or mix them with `--glossary-json` to support any target language.

---

## Quick Troubleshooting

* **`ModuleNotFoundError: google.genai` / `No module named ...`**
  Upgrade/reinstall dependencies:

  ```bat
  pip install --upgrade google-genai tqdm
  ```

* **API errors (401/403/429) or “quota/rate limit”**
  Confirm your API key is valid, the project has access enabled, and check your quotas/limits.

* **Encoding/XML issues**
  If the game is sensitive to XML encoding, keep the expected format (for example, UTF-16 LE if required) and validate the output XML before testing in-game.

---

## Security (API Key)

Do **not** commit your API key to GitHub or share it publicly.
Use a safer approach (like environment variables) if you plan to automate runs.

---

## Disclaimer

Review Google Gemini’s policies/terms for API usage (including commercial use and rate limits).

---
